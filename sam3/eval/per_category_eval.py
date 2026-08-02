# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""(Ha) Added 2026-07-19: automatic per-category evaluation + overlay snapshots.

`CocoEvaluatorOfflineWithPredFileEvaluators` (coco_eval_offline.py) only reports
ONE aggregate metric set across ALL categories combined -- during the
2026-07-19 investigation into weak thick_wire/thin_wire/gate_pad AP (see
CHANGELOG.md and nbs/11_debug_per_category_ap_investigation.ipynb), a
standalone one-off script (.github/scripts/per_category_coco_eval.py) had to
be run manually after training just to see per-category numbers.

This module makes that automatic: `PerCategoryEvalWithOverlays` is a second
`pred_file_evaluators` entry that runs every validation epoch as part of the
normal training loop, and:

1. Computes AP/AR per category, logged as
   `coco_eval_<iou_type>_<METRIC>_<category_name>` meters (e.g.
   `coco_eval_segm_AP_thick_wire`) -- picked up automatically by the same
   TensorBoard/logging path as the existing aggregate metrics.
2. Saves one JPEG "poster" overlay per category: a single, randomly-picked
   test image containing that category, with GT (green) and top-scoring
   predictions (red, with score labels) drawn on top -- NOT the
   anomalous/bad-loss debug dumps (`trainer.py`'s `_maybe_dump_anomalous_batch`,
   which are a completely separate, crash/zero-loss-triggered mechanism).
3. ALSO combines every category's overlay into ONE single poster image
   (`all_categories_poster.jpg`) -- a grid/contact-sheet with one labeled cell
   per category, so the whole dataset's per-category sanity check can be
   reviewed at a glance in a single file.

Both the metric breakdown and the category list for overlays are derived
DYNAMICALLY from `COCO(gt_path).getCatIds()` -- if a new category is added to
the dataset later, it is automatically picked up with zero code/config changes
required.
"""

import io
import logging
import random
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sam3.eval.coco_eval_offline import COCO_METRICS
from sam3.train.utils.distributed import is_main_process

GT_COLOR = (0, 200, 0)
PRED_COLOR = (255, 40, 40)
POSTER_THUMB_SIZE = (480, 360)
POSTER_LABEL_HEIGHT = 28
POSTER_PADDING = 8
POSTER_BG_COLOR = (20, 20, 20)


def _draw_annotation(image: Image.Image, draw: ImageDraw.ImageDraw, ann: dict, color: tuple, label: Optional[str] = None):
    """Draw one annotation's segmentation (polygon or RLE) + bbox on `image`/`draw`."""
    seg = ann.get("segmentation")
    if seg:
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) >= 6:
                    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                    draw.line(pts + [pts[0]], fill=color, width=3)
        elif isinstance(seg, dict) and "counts" in seg:
            try:
                binary_mask = mask_utils.decode(seg).astype(bool)
                overlay_arr = np.zeros((image.height, image.width, 4), dtype=np.uint8)
                overlay_arr[binary_mask] = (*color, 90)
                overlay = Image.fromarray(overlay_arr, mode="RGBA")
                blended = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
                image.paste(blended)
            except Exception as e:  # pragma: no cover - best-effort visualization only
                logging.warning(f"(Ha) PerCategoryEvalWithOverlays: failed to decode RLE mask: {e}")

    bbox = ann.get("bbox")
    if bbox:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        if label:
            draw.text((x, max(0, y - 12)), label, fill=color)


class PerCategoryEvalWithOverlays:
    """See module docstring. Add as an extra entry in a meter's
    `pred_file_evaluators` list alongside `CocoEvaluatorOfflineWithPredFileEvaluators`.
    """

    def __init__(
        self,
        gt_path: str,
        img_folder: str,
        overlay_dir: str,
        iou_type: str = "segm",
        num_overlay_samples: int = 1,
        max_overlay_dets: int = 20,
        score_threshold: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.gt_path = gt_path
        self.img_folder = img_folder
        self.overlay_dir = overlay_dir
        self.iou_type = iou_type
        self.num_overlay_samples = num_overlay_samples
        self.max_overlay_dets = max_overlay_dets
        self.score_threshold = score_threshold
        self.seed = seed

    def evaluate(self, dumped_file) -> dict:
        if not is_main_process():
            return {}

        logging.info("(Ha) PerCategoryEvalWithOverlays: loading groundtruth")
        gt = COCO(self.gt_path)
        cocoDt = gt.loadRes(str(dumped_file))

        # (Ha) DYNAMIC category discovery -- works automatically for any new
        # category added to the dataset, no hardcoded category list.
        cat_ids = sorted(gt.getCatIds())
        cat_id_to_name = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}

        outs = {}
        overlay_paths = {}
        for cat_id in cat_ids:
            cat_name = cat_id_to_name[cat_id]
            n_gt = len(gt.getAnnIds(catIds=[cat_id]))
            if n_gt == 0:
                # e.g. a category with a train/test split gap (0 test instances) --
                # can't be scored or visualized, skip rather than crash.
                logging.info(
                    f"(Ha) PerCategoryEvalWithOverlays: category '{cat_name}' has 0 "
                    "ground-truth instances in this split, skipping eval + overlay."
                )
                continue

            coco_eval = COCOeval(gt, cocoDt, iouType=self.iou_type)
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            buf = io.StringIO()
            with redirect_stdout(buf):
                coco_eval.summarize()
            for i, value in enumerate(coco_eval.stats):
                outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}_{cat_name}"] = value

            try:
                saved = self._save_category_overlay(gt, cocoDt, cat_id, cat_name)
                if saved:
                    overlay_paths[cat_name] = saved[0]
            except Exception as e:  # pragma: no cover - best-effort visualization only
                logging.warning(
                    f"(Ha) PerCategoryEvalWithOverlays: overlay generation failed for "
                    f"category '{cat_name}': {e}"
                )

        if overlay_paths:
            try:
                self._save_combined_poster(overlay_paths)
            except Exception as e:  # pragma: no cover - best-effort visualization only
                logging.warning(
                    f"(Ha) PerCategoryEvalWithOverlays: combined poster generation failed: {e}"
                )

        return outs

    def _save_category_overlay(self, gt: COCO, cocoDt: COCO, cat_id: int, cat_name: str) -> list:
        img_ids = sorted(gt.getImgIds(catIds=[cat_id]))
        if not img_ids:
            return []

        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        chosen_img_ids = rng.sample(img_ids, min(self.num_overlay_samples, len(img_ids)))

        out_dir = Path(self.overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, img_id in enumerate(chosen_img_ids):
            img_info = gt.loadImgs([img_id])[0]
            img_path = Path(self.img_folder) / img_info["file_name"]
            if not img_path.exists():
                logging.warning(
                    f"(Ha) PerCategoryEvalWithOverlays: image not found for overlay: {img_path}"
                )
                continue

            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
            for ann in gt_anns:
                _draw_annotation(image, draw, ann, GT_COLOR)

            pred_ann_ids = cocoDt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
            pred_anns = cocoDt.loadAnns(pred_ann_ids)
            pred_anns = sorted(pred_anns, key=lambda a: -a.get("score", 0.0))[: self.max_overlay_dets]
            for ann in pred_anns:
                score = ann.get("score", 0.0)
                if score < self.score_threshold:
                    continue
                _draw_annotation(image, draw, ann, PRED_COLOR, label=f"{score:.2f}")

            suffix = f"_{idx}" if self.num_overlay_samples > 1 else ""
            out_path = out_dir / f"{cat_name}{suffix}.jpg"
            image.convert("RGB").save(out_path, quality=90)
            logging.info(f"(Ha) PerCategoryEvalWithOverlays: saved overlay {out_path}")
            saved_paths.append(out_path)

        return saved_paths

    def _save_combined_poster(self, overlay_paths: dict):
        """(Ha) Added 2026-07-19: combine every category's individual overlay
        JPEG into ONE single poster image (a grid/contact-sheet, one cell per
        category, labeled with the category name) -- so the whole dataset's
        per-category sanity check can be reviewed at a glance in one file,
        instead of opening N separate JPEGs.
        """
        from PIL import ImageDraw as _ImageDraw  # local import to avoid confusion with module-level draw usage
        from PIL import ImageFont

        cat_names = sorted(overlay_paths.keys())
        n = len(cat_names)
        if n == 0:
            return

        n_cols = max(1, round(n ** 0.5))
        n_rows = (n + n_cols - 1) // n_cols

        cell_w = POSTER_THUMB_SIZE[0] + 2 * POSTER_PADDING
        cell_h = POSTER_THUMB_SIZE[1] + POSTER_LABEL_HEIGHT + 2 * POSTER_PADDING

        poster = Image.new("RGB", (n_cols * cell_w, n_rows * cell_h), color=POSTER_BG_COLOR)
        draw = _ImageDraw.Draw(poster)
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None

        for i, cat_name in enumerate(cat_names):
            row, col = divmod(i, n_cols)
            x0 = col * cell_w + POSTER_PADDING
            y0 = row * cell_h + POSTER_PADDING

            thumb = Image.open(overlay_paths[cat_name]).convert("RGB")
            thumb.thumbnail(POSTER_THUMB_SIZE)
            # center the thumbnail within the fixed-size cell
            paste_x = x0 + (POSTER_THUMB_SIZE[0] - thumb.width) // 2
            paste_y = y0 + (POSTER_THUMB_SIZE[1] - thumb.height) // 2
            poster.paste(thumb, (paste_x, paste_y))

            label_y = y0 + POSTER_THUMB_SIZE[1] + 4
            draw.text((x0, label_y), cat_name, fill=(255, 255, 255), font=font)

        out_path = Path(self.overlay_dir) / "all_categories_poster.jpg"
        poster.save(out_path, quality=90)
        logging.info(f"(Ha) PerCategoryEvalWithOverlays: saved combined poster {out_path}")
