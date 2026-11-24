"""Our copy of UNVEIL's training loop, reduced to what the trial run needs (docs/unveil-recipe.md rules 2, 5, 7).

``run_trial`` draws batches from the recipe streams (``streams.StreamSampler``), runs a few hundred optimizer steps and
reports step time and peak memory, so ``epochs`` can be chosen before a long run. The joint presence module exists
only when the recipe configures a ``joint`` pair. Two models plug in: ``stub`` (a tiny CNN, CPU-friendly, used by the
tests) and ``unveil`` (UNVEIL's ``TrainNet`` on SAM3 weights; ``unveil`` is imported lazily, only for this model).
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import checkpoint as C
from .calibrate import calibrate
from .ops import verified_absent
from .streams import build_streams, read_item, StreamSampler


def _loss(logit: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    """``region``: BCE + IoU; ``balanced``: class-balanced BCE + IoU (recipe rule 3)."""
    bce = F.binary_cross_entropy_with_logits(logit, mask, reduction="none")
    if kind == "balanced":
        n_pos = mask.sum().clamp_min(1.0)
        n_neg = (1 - mask).sum().clamp_min(1.0)
        bce = 0.5 * ((bce * mask).sum() / n_pos + (bce * (1 - mask)).sum() / n_neg)
    else:
        bce = bce.mean()
    p = logit.sigmoid()
    inter = (p * mask).sum()
    return bce + 1 - inter / (p + mask - p * mask).sum().clamp_min(1e-6)


# A batch item: (image, mask or None for a verified-absent image, kind, prompt id, presence 1.0 / 0.0).
Item = tuple[torch.Tensor, "torch.Tensor | None", str, int, float]


class StubModel(nn.Module):
    """A tiny CNN standing in for the UNVEIL network: one map channel per category, the presence logit is the map's
    maximum plus a per-category bias. The joint head exists only when a joint pair is configured."""

    def __init__(self, categories: list[str], joint: list[str] | None):
        super().__init__()
        n_categories = len(categories)
        self.joint_ids = [categories.index(j) for j in joint or []]
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, n_categories, 3, padding=1),
        )
        self.bias = nn.Parameter(torch.zeros(n_categories))
        self.joint = nn.Linear(3, len(joint)) if joint else None

    def _maps(self, x: torch.Tensor, prompt: int) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.body(x[None])[0, prompt]
        return logit, logit.amax() + self.bias[prompt]

    def step_loss(self, batch: list[Item]):
        losses, pres = [], []
        for x, y, kind, prompt, presence in batch:
            logit, p = self._maps(x, prompt)
            if y is not None:
                losses.append(_loss(logit, y, kind))
            pres.append(
                F.binary_cross_entropy_with_logits(
                    p, torch.tensor(presence, device=p.device)
                )
            )
        loss = torch.stack(pres).mean()
        if losses:
            loss = loss + torch.stack(losses).mean()
        if (
            self.joint is not None
        ):  # presence of the two joint members, from the image mean colour
            feats = torch.stack([x.mean((1, 2)) for x, *_ in batch])
            loss = loss + F.binary_cross_entropy_with_logits(
                self.joint(feats), self.joint_target(batch, feats.device)
            )
        return loss

    def joint_target(self, batch: list[Item], device=None) -> torch.Tensor:
        """(items, 2): 1 where the item is a positive of that joint member's category, else 0."""
        return torch.tensor(
            [
                [float(presence > 0 and prompt == j) for j in self.joint_ids]
                for _, _, _, prompt, presence in batch
            ],
            device=device,
        )

    def optimizer_groups(self) -> list[dict]:
        return [
            {"params": [p for p in self.parameters() if p.requires_grad], "lr": 1e-3}
        ]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, prompt: int) -> tuple[np.ndarray, float]:
        """Probability map (H, W) and presence logit of one image."""
        logit, p = self._maps(x, prompt)
        return logit.sigmoid().cpu().numpy(), float(p)

    def tensor_groups(self) -> tuple[dict, torch.Tensor | None]:
        """Checkpoint tensors by prefix (``checkpoint.assemble``) and the convention vectors."""
        groups = {
            "heads.": {
                k: v for k, v in self.state_dict().items() if not k.startswith("joint.")
            }
        }
        if self.joint is not None:
            groups["joint.decoder."] = dict(self.joint.state_dict())
        return groups, None


def build_model(name: str, plan: dict, root: Path, device: str, joint) -> nn.Module:
    if name == "stub":
        return StubModel(list(plan["categories"]), joint).to(device)
    if name == "unveil":
        return _unveil_model(plan, root, device, joint)
    raise SystemExit(f"unknown model {name!r}; use model=unveil or model=stub")


def _unveil_model(plan: dict, root: Path, device: str, joint) -> nn.Module:
    """UNVEIL's ``TrainNet`` on SAM3 weights. Needs the ``unveil`` package and ``sam3_checkpoint``.

    NOT TESTED: written against UNVEIL's source (``train/trainer.py``, ``train/network.py``) but never run, because it
    needs a CUDA GPU, the SAM3 weights and UNVEIL installed.
    """
    try:
        from unveil.model.backbone import encode_text, vit_prefix
        from unveil.model.sam3 import build_sam3, normalize, set_tf32
        from unveil.train.losses import total_loss
        from unveil.train.network import LoraConfig, param_groups, TrainNet
    except ImportError as e:
        raise SystemExit(
            f"model=unveil needs the unveil package ({e}); install UNVEIL, or use model=stub for a CPU smoke run"
        )
    if joint:
        raise SystemExit(
            "the joint module is not wired to the real UNVEIL model yet; remove `joint:` from recipe.yaml "
            "or run with model=stub"
        )
    sam3 = build_sam3(device, plan.get("sam3_checkpoint"))
    set_tf32(True)
    text = encode_text(sam3, list(plan["categories"]), device)
    lora = LoraConfig()
    net = TrainNet(sam3, text, lora, device)

    class Real(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def parameters(self, recurse=True):
            return iter(net.trainable())

        def step_loss(self, batch):
            imgs = torch.stack([x for x, *_ in batch]).to(device) * 255
            labels = [
                None if y is None else (y * 255).to(torch.uint8) for _, y, *_ in batch
            ]
            sizes = [None if y is None else tuple(y.shape) for _, y, *_ in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = vit_prefix(sam3, normalize(imgs), lora.first_block)
                ids = torch.arange(len(batch), device=device)
                prompts = torch.tensor([b[3] for b in batch], device=device)
                conv = torch.zeros(len(batch), dtype=torch.bool, device=device)
                logits, pres = net(prefix, ids, prompts, conv, sizes)
            kinds = [b[2] for b in batch]
            groups = ["shadow" if k == "balanced" else "region" for k in kinds]
            target = torch.tensor([b[4] for b in batch], device=device)
            loss, _ = total_loss(logits, labels, kinds, groups, pres, target)
            return loss

        def optimizer_groups(self):
            return param_groups(net, 8e-5, 0.1, 2.5e-4, 0.1, 8e-4)

        @torch.no_grad()
        def predict(self, x, prompt):
            net.eval()
            h, w = x.shape[-2:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = vit_prefix(sam3, normalize(x[None] * 255), lora.first_block)
                zero = torch.zeros(1, dtype=torch.long, device=device)
                logits, pres = net(
                    prefix,
                    zero,
                    zero + prompt,
                    torch.zeros(1, dtype=torch.bool, device=device),
                    [(h, w)],
                )
            return logits[0].float().sigmoid().cpu().numpy(), float(pres[0])

        def tensor_groups(self):
            state = net.state()
            return {"heads.": state["heads"], "lora.": state["lora"]}, state[
                "convention"
            ]

    return Real()


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(
            f"{name} is {value}; use a number of optimizer steps above 0 (`{name}=N`)"
        )


def _peak_gb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**30
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KB on Linux


def negative_pools(plan: dict, index: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Per category, the train images verified absent: annotated in a declared negative category, none of whose labels
    is in the category's own family (recipe rule 4)."""
    pools = {}
    for name in plan["categories"]:
        pools[name] = verified_absent(plan["family_of"], plan["negatives"], name, index)
    return pools


def lr_factor(it: int, steps: int, warmup: int) -> float:
    """Linear warmup, then cosine to zero (the paper's schedule, ``warmup`` scaled down for short runs)."""
    if it < warmup:
        return (it + 1) / warmup
    t = (it - warmup) / max(1, steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def _run_steps(
    plan: dict,
    index: dict[str, list[dict]],
    root: Path,
    steps: int,
    model: str,
    device: str,
    size: int,
    seed: int,
    accumulate: int = 1,
    schedule: bool = False,
    log=None,
) -> tuple[nn.Module, list[float], int, float]:
    """The loop shared by the trial and the training run -> (model, seconds per step, trainable parameters, peak memory in GB).

    Each micro-batch holds ``per_gpu`` positives of every stream plus one verified-absent image per stream (presence 0,
    no mask), so the presence output learns both answers. ``accumulate`` micro-batches make one optimizer step.
    """
    streams = build_streams(plan)
    sampler = StreamSampler(
        streams, {s.name: len(index[s.name]) for s in streams}, seed=seed
    )
    pools = negative_pools(plan, index)
    torch.manual_seed(seed)
    net = build_model(model, plan, root, device, plan["joint"])
    groups = net.optimizer_groups()
    for g in groups:
        g.setdefault("base_lr", g["lr"])
    opt = torch.optim.AdamW(groups, lr=1e-4)
    params = [p for g in groups for p in g["params"] if p.requires_grad]
    warmup = min(300, max(1, steps // 10))
    if device.startswith(
        "cuda"
    ):  # the peak covers the weights and the steps, not the setup before
        torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(steps):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * (lr_factor(it, steps, warmup) if schedule else 1)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(accumulate):
            micro = it * accumulate + k
            rng = np.random.default_rng([seed, 1, micro])
            batch: list[Item] = []
            for s, i in sampler.batch(micro):
                st = streams[s]
                x, y = read_item(root, index[st.name][i], size)
                batch.append((x.to(device), y.to(device), st.kind, st.prompts[0], 1.0))
            for st in streams:
                if pools[st.name]:
                    rec = pools[st.name][int(rng.integers(len(pools[st.name])))]
                    x, _ = read_item(root, rec, size)
                    batch.append((x.to(device), None, st.kind, st.prompts[0], 0.0))
            loss = net.step_loss(batch) / accumulate
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 0.1)
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if log is not None:
            log(
                {
                    "step": it + 1,
                    "loss": total,
                    "lr": opt.param_groups[0]["lr"],
                    "step_ms": round(1000 * times[-1], 2),
                }
            )
    return net, times, sum(p.numel() for p in params), _peak_gb(device)


def run_trial(
    plan: dict,
    index: dict[str, list[dict]],
    root: str | Path,
    steps: int,
    model: str = "unveil",
    device: str = "cpu",
    size: int = 1008,
    seed: int = 0,
) -> dict:
    """Run ``steps`` optimizer steps on the recipe streams; report step time and peak memory."""
    _positive("trial_steps", steps)
    _, times, n_params, peak = _run_steps(
        plan, index, Path(root), steps, model, device, size, seed
    )
    return {
        "model": model,
        "device": device,
        "steps": steps,
        "images_per_step": sum(s.per_gpu for s in build_streams(plan)),
        "step_ms": round(1000 * statistics.median(times), 2),
        "peak_memory_gb": round(peak, 3),
        "trainable_params": n_params,
        "joint": plan["joint"],
    }


def steps_for(
    plan: dict, index: dict[str, list[dict]], epochs: float, accumulate: int
) -> int:
    """Rule 5: epochs x distinct train images / images per optimizer step (one GPU: per_gpu x streams x accumulate)."""
    images = len({r["id"] for rs in index.values() for r in rs})
    per_step = sum(s.per_gpu for s in build_streams(plan)) * accumulate
    return max(1, math.ceil(epochs * images / per_step))


def run_training(
    plan: dict,
    index: dict[str, list[dict]],
    root: str | Path,
    out: str | Path,
    steps: int,
    model: str = "unveil",
    device: str = "cpu",
    size: int = 1008,
    seed: int = 0,
    accumulate: int = 1,
) -> dict:
    """Train for ``steps`` optimizer steps, fit the thresholds on ``val``, write ``<out>/unveil.safetensors``."""
    _positive("steps", steps)
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.jsonl", "w") as f:

        def log(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()

        net, times, n_params, peak = _run_steps(
            plan, index, root, steps, model, device, size, seed, accumulate, True, log
        )
    tau, binarize, report = calibrate(net, plan, root, size, device)
    groups, convention = net.tensor_groups()
    meta = C.make_meta(
        plan["categories"],
        {
            "single": {n: t for n, t in tau.items()},
            "joint": {},
            "free_text": float(np.mean(list(tau.values()))),
            "rule": "per category: max balanced accuracy on the val split, ties -> 0.5; verified-absent val images "
            "come from other families only",
        },
        binarize,
        {
            "steps": steps,
            "kinds": plan["kinds"],
            "custom_joint_pair": plan["joint"],
            "trained_with": model,
        },
    )
    path = out / "unveil.safetensors"
    digest = C.save_checkpoint(path, C.assemble(groups, convention), meta)
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    return {
        "checkpoint": str(path),
        "sha256": digest,
        "steps": steps,
        "model": model,
        "device": device,
        "step_ms": round(1000 * statistics.median(times), 2),
        "peak_memory_gb": round(peak, 3),
        "trainable_params": n_params,
        "thresholds": {"presence": tau, "binarize": binarize},
    }
