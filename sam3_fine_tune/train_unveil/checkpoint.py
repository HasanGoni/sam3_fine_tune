"""The checkpoint ``sam3-train-unveil`` writes: UNVEIL's own format (``unveil/checkpoint.py``), without importing it.

One safetensors file (tensors plus a JSON ``unveil`` metadata entry) and a ``.sha256`` file next to it, so
``unveil.load()`` opens it. safetensors holds no pickled objects, so reading is weights-only safe by construction;
nothing here calls ``torch.load``. Tensor groups: ``heads.``, ``lora.``, ``joint.decoder.`` and ``joint.convention``.
The metadata holds the presence thresholds (``thresholds.single`` keyed by the category name, which is the prompt) and
the binarization thresholds (``binarize``, one 8-bit threshold per category).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT, FORMAT_VERSION, META_KEY = "unveil-checkpoint", 1, "unveil"
LORA_DEFAULT = {
    "r": 16,
    "alpha": 32.0,
    "first_block": 24,
    "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
}
CONVENTION_SHAPE = (2, 256)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def assemble(groups: dict[str, dict], convention: torch.Tensor | None) -> dict:
    """Prefix -> tensors into the flat fp32 CPU dict of the file; ``joint.convention`` is zeros when there is none."""
    out = {}
    for prefix in ("heads.", "lora.", "joint.decoder."):
        for k, v in sorted(groups.get(prefix, {}).items()):
            out[prefix + k] = v.detach().float().cpu().contiguous()
    conv = torch.zeros(CONVENTION_SHAPE) if convention is None else convention
    out["joint.convention"] = conv.detach().float().cpu().contiguous()
    return out


def make_meta(
    categories: list[str], thresholds: dict, binarize: dict, extra: dict
) -> dict:
    """``categories`` are the prompts, in prompt-id order; ``thresholds`` as UNVEIL's ``make_meta`` documents them."""
    try:  # the SAM3 commit UNVEIL expects, when UNVEIL is installed (lazy: tests run without it)
        from unveil.model.sam3 import SAM3_COMMIT
    except ImportError:
        SAM3_COMMIT = None
    return {
        "model": "UNVEIL",
        "sam3": {"repo": "facebook/sam3", "file": "sam3.pt", "commit": SAM3_COMMIT},
        "lora": dict(LORA_DEFAULT),
        "prompts": list(categories),
        "joint_pair": [],
        "thresholds": thresholds,
        "binarize": binarize,
        **extra,
    }


def save_checkpoint(path: str | Path, tensors: dict, meta: dict) -> str:
    """Write the file atomically and its ``.sha256`` -> the digest. The same inputs give the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT, "format_version": FORMAT_VERSION, **meta}
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp), metadata={META_KEY: json.dumps(meta, sort_keys=True)})
    os.replace(tmp, path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def read_checkpoint(path: str | Path) -> tuple[dict, dict]:
    """-> (tensors on the CPU, metadata). Checks the format and the ``.sha256`` file when there is one."""
    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(META_KEY)
    if raw is None:
        raise ValueError(
            f"{path} is not an UNVEIL checkpoint (no '{META_KEY}' metadata)"
        )
    meta = json.loads(raw)
    if (
        meta.get("format") != FORMAT
        or int(meta.get("format_version", -1)) > FORMAT_VERSION
    ):
        raise ValueError(f"unsupported checkpoint format {meta.get('format')}")
    side = path.with_name(path.name + ".sha256")
    if side.exists():
        words = side.read_text().split()
        if not words:
            raise ValueError(
                f"{side} is empty or truncated; delete it to skip the check, or copy the checkpoint again"
            )
        if sha256_file(path) != words[0]:
            raise ValueError(f"sha256 mismatch for {path}")
    return load_file(str(path), device="cpu"), meta


def read_meta(path: str | Path) -> dict | None:
    """The ``unveil`` metadata of a checkpoint without loading its tensors; ``None`` when the file has none."""
    try:
        with safe_open(str(path), framework="pt") as f:
            raw = (f.metadata() or {}).get(META_KEY)
    except Exception:  # not a safetensors file: the model loader reports it
        return None
    return None if raw is None else json.loads(raw)


def _norm(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def cutoffs(
    texts: list[str], checkpoint: str | Path | None, fallback: float
) -> dict[str, dict]:
    """Per prompt text, the concept-map cut-off: the checkpoint's 8-bit ``binarize`` t (the map is kept where the
    rounded 8-bit value is >= t, so the float cut-off is (t - 0.5) / 255) when it has one for the prompt, else
    ``fallback``. -> {text: {"value", "source"}}. The presence (gate) thresholds are applied by UNVEIL's ``predict``."""
    meta = read_meta(checkpoint) if checkpoint else None
    binarize = {_norm(k): v for k, v in ((meta or {}).get("binarize") or {}).items()}
    out = {}
    for text in texts:
        t = binarize.get(_norm(text))
        if isinstance(t, (int, float)) and 1 <= t <= 255:
            out[text] = {
                "value": (float(t) - 0.5) / 255,
                "source": f"checkpoint binarize t={int(t)}",
            }
        else:
            out[text] = {"value": fallback, "source": "threshold"}
    return out
