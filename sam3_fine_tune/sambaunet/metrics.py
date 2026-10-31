"""Per-category foreground IoU / precision / recall, in the exact JSON layout of
``sam3_fine_tune/evaluation/multi_eval_pipe.py`` (``summary.categories`` / ``summary.overall`` /
``per_category_details``), so SAMBA-UNet and SAM3 results can be compared key-by-key.

IoU = TP / (TP + FP + FN); an image with no foreground in both prediction and ground truth scores 1.0 (as in SAM3 eval).
"""
from __future__ import annotations

import numpy as np

SUMMARY_FILENAME = "multiclass_foreground_iou_summary.json"  # same name the SAM3 pipeline writes


def _ratio(num: float, den: float, empty: float = 0.0) -> float:
    return num / den if den > 0 else empty


class CategoryScores:
    def __init__(self, names: list[str]):
        self.names = list(names)
        self.files: list[str] = []
        self.rows: list[np.ndarray] = []  # per image: (K,4) -> tp, fp, fn, tn

    def update(self, pred: np.ndarray, gt: np.ndarray, file_name: str = "") -> None:
        """pred, gt: (K,H,W) bool/uint8 at the same resolution."""
        p, g = pred.astype(bool), gt.astype(bool)
        tp, fp, fn = (p & g).sum((1, 2)), (p & ~g).sum((1, 2)), (~p & g).sum((1, 2))
        tn = p[0].size - tp - fp - fn
        self.rows.append(np.stack([tp, fp, fn, tn], 1).astype(np.int64))
        self.files.append(file_name)

    def _cat(self, k: int) -> np.ndarray:  # (N,4) for category k
        return np.stack([r[k] for r in self.rows]) if self.rows else np.zeros((0, 4), np.int64)

    def summary(self) -> dict:
        cats, g = {}, np.zeros(3, np.int64)
        for k, name in enumerate(self.names):
            a = self._cat(k); n = len(a)
            cat = {"total_images": n, "errors": 0}
            if n:
                tp, fp, fn, tn = (int(a[:, i].sum()) for i in range(4))
                iou_img = [_ratio(r[0], r[0] + r[1] + r[2], 1.0) for r in a]
                prec = _ratio(tp, tp + fp); rec = _ratio(tp, tp + fn)
                best, worst = int(np.argmax(iou_img)), int(np.argmin(iou_img))
                cat.update({
                    "overall_foreground_iou": round(100 * _ratio(tp, tp + fp + fn), 2),
                    "mean_foreground_iou": round(100 * float(np.mean(iou_img)), 2),
                    "pixel_precision": round(100 * prec, 2), "pixel_recall": round(100 * rec, 2),
                    "pixel_f1": round(100 * _ratio(2 * prec * rec, prec + rec), 2),
                    "total_TP_pixels": tp, "total_FP_pixels": fp, "total_FN_pixels": fn, "total_TN_pixels": tn,
                    "best_sample": {"file_name": self.files[best], "fg_iou": round(iou_img[best], 4)},
                    "worst_sample": {"file_name": self.files[worst], "fg_iou": round(iou_img[worst], 4)},
                })
                g += (tp, fp, fn)
            cats[name] = cat
        tp, fp, fn = (int(x) for x in g)
        overall = {"foreground_iou": round(100 * _ratio(tp, tp + fp + fn), 2),
                   "precision": round(100 * _ratio(tp, tp + fp), 2), "recall": round(100 * _ratio(tp, tp + fn), 2),
                   "total_TP_pixels": tp, "total_FP_pixels": fp, "total_FN_pixels": fn}
        done = [c["overall_foreground_iou"] for c in cats.values() if "overall_foreground_iou" in c]
        # macro_foreground_iou is extra (not in the SAM3 file): mean over categories of the pixel-aggregated IoU
        return {"categories": cats, "overall": overall,
                "macro_foreground_iou": round(float(np.mean(done)), 2) if done else 0.0}

    def details(self) -> dict:
        out = {}
        for k, name in enumerate(self.names):
            a = self._cat(k)
            rows = []
            for f, (tp, fp, fn, _tn) in zip(self.files, a):
                rows.append({"file_name": f, "foreground_iou": round(_ratio(tp, tp + fp + fn, 1.0), 4),
                             "precision": round(_ratio(tp, tp + fp), 4), "recall": round(_ratio(tp, tp + fn), 4),
                             "TP_pixels": int(tp), "FP_pixels": int(fp), "FN_pixels": int(fn)})
            out[name] = {"per_image_results": rows, "errors": []}
        return out
