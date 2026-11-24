"""Score models on the ``test`` split of a prepared root with one metric: pooled union-mask IoU per category.

For each category the masks of one model (the union of the instances above the threshold) and the ground truth are
pooled over every scored image: ``IoU = sum(intersection) / sum(union)``. A category is scored only on the images that
are annotated for it; an image listed in ``not_annotated_category_ids`` for it is skipped, never a negative or a
positive. A category with no scored image is ``None`` (shown as ``n/a``), not 0.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np
from PIL import Image

from sam3_fine_tune.train_unveil.checkpoint import cutoffs as checkpoint_cutoffs

UNVEIL_KINDS = ("unveil", "unveil-sam3-path")
KINDS = ("sam3-base", "sam3-finetuned", "predictions", *UNVEIL_KINDS)
SPLIT = Path("prepared") / "sam3" / "test"
ANNOTATION = "annotation_merged.json"

SUSPECT_BELOW = 0.01  # a score under this on a trained category is marked suspect


def _norm(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def prompt_for(category: str, trained: list[str] | None) -> str:
    """The text prompt for a category: the spelling the model was trained on when it differs only in case or
    separators. The text is the label, so ``thin_wire_panda`` asked of a model trained on ``thin wire panda``
    finds nothing (0 instances even at threshold 0.01); that was the 0.000 of #18."""
    for name in trained or []:
        if _norm(name) == _norm(category):
            return name
    return category


def flag_for(
    entry: dict[str, Any], category: str, iou: float | None, below: float
) -> str | None:
    """``not trained`` when the entry lists its training categories and this is not one of them; ``suspect`` for a
    near-zero score on a category the model may have been trained on (a prompt or threshold problem, not a weakness)."""
    trained = entry.get("trained_categories")
    if trained is not None and _norm(category) not in {_norm(t) for t in trained}:
        return "not trained"
    if (
        iou is not None
        and iou < below
        and entry["kind"] not in ("sam3-base", "unveil-sam3-path")
    ):
        return "suspect"
    return None


def _shape(image: Path) -> tuple[int, int]:
    """(height, width) from the file header, without decoding the pixels."""
    with Image.open(image) as im:
        return im.size[1], im.size[0]


# a mask source answers "the union mask of this model for this category on this image", or None when it has none
MaskSource = Callable[[Path, str], "np.ndarray | None"]


def predictions_source(entry: dict[str, Any]) -> MaskSource:
    """Saved ``sam3-infer`` masks: ``<path>/masks/<prompt slug>/<image stem>.png`` (several prompts)."""
    from sam3_fine_tune.inference.run import PromptSpec

    path = entry.get("path")
    if not path or not (Path(path) / "masks").is_dir():
        raise ValueError(
            f"model {entry['name']!r}: kind predictions needs path=<sam3-infer output folder> "
            f"holding masks/ (got {path!r}); run sam3-infer with several prompts first"
        )
    masks = Path(path) / "masks"

    def source(image: Path, category: str) -> np.ndarray | None:
        folder = masks / PromptSpec(category, (0.5,)).slug
        if not folder.is_dir():
            return None  # the model was not asked for this category
        f = folder / f"{image.stem}.png"
        if not f.is_file():
            raise ValueError(
                f"model {entry['name']!r}: no mask {f}; predict every test image (sam3-infer coco_split=...)"
            )
        return np.asarray(Image.open(f).convert("L")) > 127

    return source


def model_source(
    entry: dict[str, Any],
    categories: list[str],
    threshold: float,
    sam3_checkpoint: Any,
    bpe_path: Any,
    device: str,
    trust_checkpoint: bool = False,
) -> MaskSource:
    """Real SAM3 (base or fine-tuned): the union of the instances above the threshold, cached for one image."""
    import torch

    from sam3_fine_tune.inference.run import make_predictor, PromptSpec, resolve_device

    kind, checkpoint = entry["kind"], entry.get("checkpoint")
    if not sam3_checkpoint or not bpe_path:
        raise ValueError(
            f"model {entry['name']!r}: kind {kind} needs sam3_checkpoint=<sam3.pt> and bpe_path=<bpe .gz>"
        )
    if kind == "sam3-finetuned" and not checkpoint:
        raise ValueError(
            f"model {entry['name']!r}: kind sam3-finetuned needs checkpoint=<path>"
        )
    if trust_checkpoint and kind == "sam3-finetuned":
        print(
            f"sam3-compare: WARNING trust_checkpoint=true: unpickling {checkpoint} can run arbitrary code; "
            "only use it for checkpoints you trained yourself",
            file=sys.stderr,
        )
    predictor = make_predictor(
        sam3_checkpoint,
        bpe_path,
        resolve_device(device, torch.cuda.is_available()),
        checkpoint if kind == "sam3-finetuned" else None,
        trusted=trust_checkpoint,
    )
    trained = entry.get("trained_categories")
    specs = {c: PromptSpec(prompt_for(c, trained), (threshold,)) for c in categories}
    last: dict[str, Any] = {}

    def source(image: Path, category: str) -> np.ndarray | None:
        if (
            last.get("image") != image
        ):  # the image is encoded once, categories are segmented on request
            last.update(
                image=image,
                handle=predictor.encode(image),
                shape=_shape(image),
                masks={},
            )
        if category not in last["masks"]:
            union = np.zeros(last["shape"], bool)
            for m in predictor.segment(last["handle"], specs[category])[0]:
                union |= np.asarray(m, bool)
            last["masks"][category] = union
        found: np.ndarray = last["masks"][category]
        return found

    return source


def gate_setting(entry: dict[str, Any]) -> str | None:
    """The presence-gate setting of an UNVEIL column (``on``/``off``, ``none`` for the SAM3 path); else ``None``."""
    if entry["kind"] == "unveil-sam3-path":
        return "none"
    if entry["kind"] == "unveil":
        return "off" if entry.get("gate", True) is False else "on"
    return None


def unveil_source(
    entry: dict[str, Any],
    categories: list[str],
    threshold: float,
    sam3_checkpoint: Any,
    bpe_path: Any,
    loaded: dict[Any, Any],
    device: str = "auto",
    cutoffs_out: dict[str, dict] | None = None,
) -> MaskSource:
    """UNVEIL (or its frozen SAM3 semantic path) through ``sam3-infer-unveil``'s predictor: the concept map of each
    category above its cut-off, cached for one image. The cut-off is the checkpoint's per-category ``binarize`` when
    it has one, else ``threshold``; ``cutoffs_out`` receives what was used per category. The presence gate thresholds
    are applied by UNVEIL's own ``predict`` (``gate``). Imports are lazy; a loaded model is shared by entries with
    the same checkpoint (a gated and an ungated column)."""
    import torch
    from omegaconf import OmegaConf

    from sam3_fine_tune.infer_unveil import cli as infer_unveil
    from sam3_fine_tune.inference.run import PromptSpec, resolve_device

    if not sam3_checkpoint or not bpe_path:
        raise ValueError(
            f"model {entry['name']!r}: kind {entry['kind']} needs sam3_checkpoint=<sam3.pt> and bpe_path=<bpe .gz>"
        )
    checkpoint = entry.get("checkpoint")
    if entry["kind"] == "unveil-sam3-path" and checkpoint:
        raise ValueError(
            f"model {entry['name']!r}: kind unveil-sam3-path runs the frozen SAM3 path and takes no checkpoint"
        )
    if checkpoint not in loaded:
        loaded[checkpoint] = infer_unveil.load_unveil(
            OmegaConf.create(
                {
                    "sam3_checkpoint": str(sam3_checkpoint),
                    "bpe_path": str(bpe_path),
                    "checkpoint": checkpoint,
                    "device": resolve_device(device, torch.cuda.is_available()),
                }
            )
        )
    trained = entry.get("trained_categories")
    texts = {c: prompt_for(c, trained) for c in categories}
    cut = checkpoint_cutoffs(list(texts.values()), checkpoint, threshold)
    spec_of = {c: PromptSpec(texts[c], (cut[texts[c]]["value"],)) for c in categories}
    if cutoffs_out is not None:
        cutoffs_out.update({c: cut[texts[c]] for c in categories})
    specs = list(spec_of.values())
    predictor = infer_unveil.make_predictor(
        loaded[checkpoint],
        specs,
        gate_setting(entry) == "on",
        "sam3" if entry["kind"] == "unveil-sam3-path" else "unveil",
    )
    last: dict[str, Any] = {}

    def source(image: Path, category: str) -> np.ndarray | None:
        if last.get("image") != image:
            last.update(image=image, state=predictor.encode(image), shape=_shape(image))
        masks = predictor.segment(last["state"], spec_of[category])[0]
        return np.asarray(masks[0], bool) if masks else np.zeros(last["shape"], bool)

    return source


def _load_split(
    root: Path,
) -> tuple[list[dict], dict[int, str], dict[int, dict[int, list]]]:
    f = root / SPLIT / ANNOTATION
    if not f.is_file():
        raise ValueError(
            f"{f} not found; run sam3-prepare root={root} first (it writes the test split)"
        )
    coco = json.loads(f.read_text())
    problems = []
    for key, needed in (
        ("images", ("id", "file_name", "width", "height")),
        ("categories", ("id", "name")),
        ("annotations", ("image_id", "category_id", "segmentation")),
    ):
        rows = coco.get(key) if isinstance(coco, dict) else None
        if not isinstance(rows, list):
            problems.append(f"{f.name} has no `{key}` list")
            continue
        for n, r in enumerate(rows):
            missing = [k for k in needed if not isinstance(r, dict) or k not in r]
            if missing:
                problems.append(f"{f.name} {key}[{n}] lacks key(s) {missing}")
        if key == "images":
            for n, r in enumerate(rows):
                name = r.get("file_name") if isinstance(r, dict) else None
                if name is not None:
                    parts = PurePosixPath(str(name)).parts
                    if (
                        not parts
                        or PurePosixPath(str(name)).is_absolute()
                        or ".." in parts
                    ):
                        problems.append(
                            f"{f.name} images[{n}] file_name {name!r} is absolute or contains `..`; "
                            "it must be inside the images folder (run sam3-prepare again)"
                        )
    if problems:
        shown = "\n- ".join(problems[:20])
        raise ValueError(
            f"{len(problems)} problem(s) in {f}:\n- {shown}"
            + ("\n- ..." if len(problems) > 20 else "")
        )
    names = {c["id"]: c["name"] for c in coco["categories"]}
    anns: dict[int, dict[int, list]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(a)
    return coco["images"], names, anns


def _ground_truth(entry: dict, annotations: list) -> np.ndarray:
    from pycocotools import mask as mask_utils

    h, w = entry["height"], entry["width"]
    gt = np.zeros((h, w), bool)
    for a in annotations:
        seg = a["segmentation"]
        if isinstance(seg, list):
            seg = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg.get("counts"), list):
            seg = mask_utils.frPyObjects(seg, h, w)
        gt |= mask_utils.decode(seg) > 0
    return gt


def validate_models(models: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list of {name, kind, ...} entries")
    names = [m.get("name") if isinstance(m, dict) else None for m in models]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"every model entry needs a unique name, got {names}")
    for m in models:
        if m.get("kind") not in KINDS:
            raise ValueError(
                f"model {m['name']!r}: kind must be one of {list(KINDS)}, got {m.get('kind')!r}"
            )
    return models


def compare(
    root: str | Path,
    models: Any,
    subset: int | None = None,
    seed: int = 0,
    threshold: float = 0.5,
    sam3_checkpoint: Any = None,
    bpe_path: Any = None,
    device: str = "auto",
    trust_checkpoint: bool = False,
    suspect_below: float = SUSPECT_BELOW,
) -> dict[str, Any]:
    models = validate_models(models)
    if subset is not None and subset < 1:
        raise ValueError(
            f"subset must be a positive number of images or null, got {subset}"
        )
    root = Path(root)
    images, cat_names, anns = _load_split(root)
    images = sorted(images, key=lambda i: i["file_name"])
    if subset is not None and subset < len(images):
        images = sorted(
            random.Random(seed).sample(images, subset), key=lambda i: i["file_name"]
        )
    categories = [cat_names[k] for k in sorted(cat_names)]
    unveil_models: dict[Any, Any] = {}
    cutoffs: dict[str, dict[str, dict]] = {}
    sources: dict[str, MaskSource] = {
        m["name"]: predictions_source(m)
        if m["kind"] == "predictions"
        else unveil_source(
            m,
            categories,
            threshold,
            sam3_checkpoint,
            bpe_path,
            unveil_models,
            device,
            cutoffs.setdefault(m["name"], {}),
        )
        if m["kind"] in UNVEIL_KINDS
        else model_source(
            m,
            categories,
            threshold,
            sam3_checkpoint,
            bpe_path,
            device,
            trust_checkpoint,
        )
        for m in models
    }
    inter = {n: {c: 0 for c in categories} for n in sources}
    union = {n: {c: 0 for c in categories} for n in sources}
    unprompted: dict[str, set[str]] = {n: set() for n in sources}
    count = {c: 0 for c in categories}
    for img in images:
        flagged = set(img.get("not_annotated_category_ids", []))
        path = root / SPLIT / "images" / img["file_name"]
        for cid, cat in sorted(cat_names.items()):
            if cid in flagged or cid not in anns.get(img["id"], {}):
                continue  # scored only where the category is annotated
            count[cat] += 1
            gt = _ground_truth(img, anns[img["id"]][cid])
            for name, source in sources.items():
                pred = source(path, cat)
                if pred is None:
                    unprompted[name].add(cat)
                    continue
                if pred.shape != gt.shape:
                    raise ValueError(
                        f"model {name!r}, {img['file_name']}: prediction {pred.shape} does not match image {gt.shape}"
                    )
                inter[name][cat] += int((pred & gt).sum())
                union[name][cat] += int((pred | gt).sum())
    ious = {
        name: {
            c: (
                inter[name][c] / union[name][c]
                if union[name][c] and c not in unprompted[name]
                else None
            )
            for c in categories
        }
        for name in sources
    }
    return {
        "metric": "pooled union-mask IoU per category",
        "split": "test",
        "subset": None if subset is None else {"images": len(images), "seed": seed},
        "threshold": threshold,
        "suspect_below": suspect_below,
        "cutoffs": cutoffs,
        "settings": {
            m["name"]: {"kind": m["kind"], "gate": gate_setting(m)} for m in models
        },
        "models": {
            m["name"]: {
                c: {
                    "images": count[c],
                    "iou": ious[m["name"]][c],
                    "flag": flag_for(m, c, ious[m["name"]][c], suspect_below),
                }
                for c in categories
            }
            for m in models
        },
    }


def format_table(result: dict[str, Any]) -> str:
    names = list(result["models"])
    cats = list(result["models"][names[0]])
    width = max(len(c) for c in cats + ["category"])

    def cell(n: str, c: str) -> str:
        r = result["models"][n][c]
        if r["flag"] == "not trained":
            return "not trained"
        text = "n/a" if r["iou"] is None else f"{r['iou']:.3f}"
        return f"{text} suspect" if r["flag"] == "suspect" else text

    cw = {n: max([12, len(n)] + [len(cell(n, c)) for c in cats]) for n in names}
    head = f"{'category'.ljust(width)}  {'images':>6}" + "".join(
        f"  {n:>{cw[n]}}" for n in names
    )
    rows = [head, "-" * len(head)]
    for c in cats:
        rows.append(
            f"{c.ljust(width)}  {result['models'][names[0]][c]['images']:>6}"
            + "".join(f"  {cell(n, c):>{cw[n]}}" for n in names)
        )
    flags = {r["flag"] for n in names for r in result["models"][n].values()}
    legend = []
    if "suspect" in flags:
        legend.append(
            f"suspect: below {result['suspect_below']} on a category the model may be trained on; "
            "check the prompt spelling and the threshold before trusting it"
        )
    if "not trained" in flags:
        legend.append(
            "not trained: not in the model's trained_categories; not a weakness"
        )
    gates = {n: s["gate"] for n, s in result.get("settings", {}).items() if s["gate"]}
    if gates:
        rows.insert(
            2,
            "gate".ljust(width + 8)
            + "".join(f"  {gates.get(n, '-'):>{cw[n]}}" for n in names),
        )
        rows += [
            "",
            "UNVEIL presence gate: " + ", ".join(f"{n}={g}" for n, g in gates.items()),
        ]
    cut = result.get("cutoffs", {})
    for n, per in cut.items():
        used = {c: v["source"] for c, v in per.items() if v["source"] != "threshold"}
        rows.append(
            f"UNVEIL cut-off {n}: "
            + (
                ", ".join(f"{c} {src}" for c, src in used.items())
                + f"; other categories threshold {result['threshold']}"
                if used
                else f"threshold {result['threshold']} for every category"
            )
        )
    sub = result["subset"]
    note = (
        "full test split"
        if sub is None
        else f"random subset of {sub['images']} test images, seed {sub['seed']}"
    )
    return "\n".join(rows + ["", f"pooled union-mask IoU, {note}", *legend])


def write_result(result: dict[str, Any], out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
