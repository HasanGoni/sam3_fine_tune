"""One training stream per recipe category, built from the prepared UNVEIL view (rules 1, 2, 4 of docs/unveil-recipe.md).

Replaces UNVEIL's 12 fixed ``TRUNK_STREAMS``. ``Stream`` and the sampling rule mirror
``unveil/data/streams.py`` (same pass permutation and position formulas, so a resumed run draws the same data); the
UNVEIL package itself is not imported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .ops import record_problems, safe_join, unveil_dir


@dataclass(frozen=True)
class Stream:
    name: str  # the category, also the key of its index
    kind: str  # region | balanced, declared in the recipe
    prompts: tuple  # prompt id of the category
    per_gpu: int  # images per GPU per step
    perm_key: int  # distinct per stream, seeds the pass permutations


def build_streams(plan: dict) -> tuple[Stream, ...]:
    """One stream per recipe category, in recipe order, each with ``plan["per_gpu"]`` images per step."""
    return tuple(
        Stream(name, plan["kinds"][name], (plan["prompts"][name],), plan["per_gpu"], k)
        for k, name in enumerate(plan["categories"])
    )


def read_item(root: Path, item: dict, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Image (3, size, size) float in 0..1 and mask (size, size) float in {0, 1} of one index record."""
    img = Image.open(safe_join(root, item["image"])).convert("RGB")
    gt = Image.open(safe_join(root, item["gt"])).convert("L")
    x = torch.from_numpy(np.asarray(img.resize((size, size)), dtype=np.float32) / 255)
    y = torch.from_numpy(np.asarray(gt.resize((size, size)), dtype=np.float32) / 255)
    return x.permute(2, 0, 1), (y >= 0.5).float()


def read_split(root: str | Path, plan: dict, split: str) -> dict[str, list[dict]]:
    """Records of each category from ``prepared/unveil/index/<split>``; images annotated for nothing are dropped.

    A missing key or a path leaving the root is a ``ValueError`` naming the file and the entry."""
    view = unveil_dir(root)
    slugs = json.loads((view / "prompts.json").read_text())["slugs"]
    out, problems = {}, []
    for name in plan["categories"]:
        path = view / "index" / split / f"{slugs.get(name, name)}.jsonl"
        lines = path.read_text().splitlines() if path.is_file() else []
        records = [json.loads(x) for x in lines if x.strip()]
        problems += record_problems(f"{split} index {path.name} ({name!r})", records)
        out[name] = [r for r in records if isinstance(r, dict) and r.get("categories")]
    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in the prepared index:\n- "
            + "\n- ".join(problems)
        )
    return out


def load_index(root: str | Path, plan: dict) -> dict[str, list[dict]]:
    """Train records of each category's stream; images annotated for no category are never kept."""
    return read_split(root, plan, "train")


class StreamSampler:
    """For iteration ``it``, this GPU's (stream index, item index) draws: ``per_gpu`` from each stream."""

    def __init__(self, streams, sizes, rank: int = 0, world: int = 1, seed: int = 0):
        empty = [s.name for s in streams if not sizes.get(s.name)]
        if empty:
            raise ValueError(f"empty training streams: {empty}")
        self.streams, self.sizes = streams, sizes
        self.rank, self.world, self.seed = rank, world, seed

    def batch(self, it: int) -> list[tuple[int, int]]:
        out = []
        for s, st in enumerate(self.streams):
            n = self.sizes[st.name]
            for k in range(st.per_gpu):
                pos = (it * st.per_gpu + k) * self.world + self.rank
                q, i = divmod(pos, n)
                perm = np.random.default_rng([self.seed, st.perm_key, q]).permutation(n)
                out.append((s, int(perm[i])))
        return out


def batch_composition(plan: dict, index: dict[str, list[dict]], steps: int) -> list:
    """Image ids per stream for the first ``steps`` iterations (one GPU), to check what a batch holds."""
    streams = build_streams(plan)
    sampler = StreamSampler(streams, {s.name: len(index[s.name]) for s in streams})
    out = []
    for it in range(steps):
        step: dict[str, list[int]] = {s.name: [] for s in streams}
        for s, i in sampler.batch(it):
            step[streams[s].name].append(index[streams[s].name][i]["id"])
        out.append(step)
    return out
