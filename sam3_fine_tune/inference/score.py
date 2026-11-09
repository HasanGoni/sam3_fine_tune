"""Score ``sam3-infer`` predictions against a COCO split and compare models.

The scoring reuses ``finetune.coco_eval`` (ground truth = rasterised COCO masks, same ``CategoryScores`` and the same
``multiclass_foreground_iou_summary.json`` layout as ``sam3-finetune``), so numbers match ``sam3-finetune --eval-only``
for the same checkpoint, threshold and images. Prompts are matched to COCO category names by exact text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sam3_fine_tune.finetune.coco_eval import write_result
from sam3_fine_tune.sambaunet.data import CocoSemanticDataset
from sam3_fine_tune.sambaunet.metrics import CategoryScores

from .run import PromptSpec


@dataclass(frozen=True)
class Match:
    scored: list[
        tuple[PromptSpec, int]
    ]  # (prompt, category index in the dataset), dataset order
    prompts_without_category: list[str]
    categories_without_prompt: list[str]


def match_prompts(specs: list[PromptSpec], category_names: list[str]) -> Match:
    by_name = {name: k for k, name in enumerate(category_names)}
    scored = sorted(
        ((s, by_name[s.text]) for s in specs if s.text in by_name), key=lambda p: p[1]
    )
    used = {s.text for s, _ in scored}
    return Match(
        scored=scored,
        prompts_without_category=[s.text for s in specs if s.text not in by_name],
        categories_without_prompt=[n for n in category_names if n not in used],
    )


class CocoScorer:
    """Collects the combined mask per (image, prompt) during ``run_inference`` and scores them against COCO."""

    def __init__(
        self, split_dir: str | Path, ann_file: str, specs: list[PromptSpec]
    ) -> None:
        split_dir = Path(split_dir)
        if not (split_dir / ann_file).is_file():
            raise ValueError(f"coco_split annotation not found: {split_dir / ann_file}")
        swept = [s.text for s in specs if s.sweep]
        if swept:
            raise ValueError(
                f"scoring against coco_split needs one threshold per prompt, but {swept} has a list (sweep)"
            )
        self.ds = CocoSemanticDataset(split_dir, ann_file=ann_file)
        self.ann_file = ann_file
        self.match = match_prompts(specs, self.ds.category_names)
        if not self.match.scored:
            raise ValueError(
                f"no prompt matches a category of {split_dir / ann_file}: prompts "
                f"{self.match.prompts_without_category}, categories {self.ds.category_names}"
            )
        self.preds: dict[str, dict[str, np.ndarray]] = {}

    @property
    def image_paths(self) -> list[Path]:
        return [self.ds.image_path(i) for i in range(len(self.ds))]

    def mismatch(self) -> dict[str, list[str]]:
        return {
            "prompts_without_category": self.match.prompts_without_category,
            "categories_without_prompt": self.match.categories_without_prompt,
        }

    def collect(self, path: Path, spec: PromptSpec, union: np.ndarray) -> None:
        self.preds.setdefault(path.name, {})[spec.text] = union

    def finish(self, out_dir: str | Path, meta: dict[str, Any]) -> dict[str, Any]:
        """Score the collected masks, write the summary file and reset for the next model."""
        idxs = [k for _, k in self.match.scored]
        texts = [s.text for s, _ in self.match.scored]
        scores = CategoryScores([self.ds.category_names[k] for k in idxs])
        for i in range(len(self.ds)):
            name = self.ds.image_path(i).name
            gt = self.ds.original_mask(i)[idxs]
            pred = np.stack([self.preds[name][t] for t in texts])
            if pred.shape != gt.shape:
                raise ValueError(
                    f"{name}: prediction {pred.shape[1:]} does not match COCO mask {gt.shape[1:]}"
                )
            scores.update(pred, gt, name)
        self.preds = {}  # ready for the next model
        return write_result(
            scores,
            num_images=len(self.ds),
            latency=None,
            meta={**self.mismatch(), "annotation": self.ann_file, **meta},
            out_dir=out_dir,
        )


def format_comparison(evaluations: dict[str, dict[str, Any]]) -> str:
    """Side-by-side foreground IoU / precision / recall (percent) per category and model.

    ``evaluations``: model label -> the ``summary`` of its ``multiclass_foreground_iou_summary.json``.
    """
    labels = list(evaluations)
    cats = list(dict.fromkeys(c for e in evaluations.values() for c in e["categories"]))
    rows = [["category", *[f"{m} {k}" for m in labels for k in ("IoU", "P", "R")]]]

    def cells(entry: dict[str, Any], keys: tuple[str, str, str]) -> list[str]:
        return [f"{entry[k]:.2f}" if k in entry else "-" for k in keys]

    for c in cats:
        row = [c]
        for m in labels:
            row += cells(
                evaluations[m]["categories"].get(c, {}),
                ("overall_foreground_iou", "pixel_precision", "pixel_recall"),
            )
        rows.append(row)
    rows.append(
        [
            "ALL",
            *[
                v
                for m in labels
                for v in cells(
                    evaluations[m]["overall"], ("foreground_iou", "precision", "recall")
                )
            ],
        ]
    )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(r, widths))
        )
        for r in rows
    )
