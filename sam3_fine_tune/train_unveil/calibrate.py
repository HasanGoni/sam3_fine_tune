"""Per-category thresholds, fitted on the ``val`` split only (docs/unveil-recipe.md rule 8; the ``test`` split is never read).

For every category: the presence threshold (maximum balanced accuracy on the sigmoid of the presence logit over the
grid 0.001..0.999, ties to the value closest to 0.5, as UNVEIL's ``evaluation.thresholds``) from the val positives
and the verified-absent val images (annotated only in other families); then the binarization threshold t in 1..255 of
the presence-gated 8-bit map on the val positives: ``region`` maximizes the mean per-image IoU, ``balanced`` minimizes
the balanced error rate (BER), ties to the t closest to 128, as UNVEIL's ``evaluation.binarize``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .ops import verified_absent
from .streams import read_item, read_split

GRID = np.round(np.arange(1, 1000) / 1000, 3)
T_GRID = np.arange(1, 256)


def load_val(root: str | Path, plan: dict) -> dict[str, list[dict]]:
    """Val records of each category from ``prepared/unveil/index/val``; images annotated for nothing are dropped."""
    return read_split(root, plan, "val")


def best_tau(score: np.ndarray, y: np.ndarray) -> dict:
    tpr = np.array([(score[y == 1] >= t).mean() for t in GRID])
    tnr = np.array([(score[y == 0] < t).mean() for t in GRID])
    ba = (tpr + tnr) / 2
    best = np.flatnonzero(ba == ba.max())
    k = best[np.argmin(np.abs(GRID[best] - 0.5))]
    return {
        "tau": float(GRID[k]),
        "balanced_acc": float(ba[k]),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def best_binarization(fg: np.ndarray, bg: np.ndarray, kind: str) -> tuple[int, float]:
    """fg / bg: (n, 256) per-image histograms of the 8-bit map over GT foreground / background -> (t, objective)."""
    tp = np.cumsum(fg[:, ::-1], 1)[:, ::-1][:, T_GRID]  # pixels >= t
    fp = np.cumsum(bg[:, ::-1], 1)[:, ::-1][:, T_GRID]
    npos, nneg = fg.sum(1, keepdims=True), bg.sum(1, keepdims=True)
    if kind == "balanced":  # BER summed over the set, minimized
        p, n = max(npos.sum(), 1), max(nneg.sum(), 1)
        curve = 100 * (1 - 0.5 * (tp.sum(0) / p + (n - fp.sum(0)) / n))
        best = curve.min()
    else:  # mean per-image IoU, maximized
        den = tp + fp + (npos - tp)
        curve = np.where(den > 0, tp / np.maximum(den, 1), 1.0).mean(0)
        best = curve.max()
    cand = T_GRID[np.flatnonzero(curve == best)]
    t = int(cand[np.lexsort((cand, np.abs(cand - 128)))][0])
    return t, float(best)


def _hist(map8: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fg = np.bincount(map8[gt > 0], minlength=256)
    bg = np.bincount(map8[gt == 0], minlength=256)
    return fg, bg


@torch.no_grad()
def calibrate(
    model, plan: dict, root: str | Path, size: int, device: str
) -> tuple[dict, dict, dict]:
    """-> (presence thresholds per category, binarization thresholds per category, the report written as JSON)."""
    root = Path(root)
    val = load_val(root, plan)
    if not any(val.values()):
        raise SystemExit(
            "the prepared view has no val images, so thresholds cannot be fitted (they are never fitted on test): "
            "run `sam3-prepare root=... val_fraction=0.1` (or more), then train again"
        )
    model.eval()

    def run(rec: dict, name: str, keep_map: bool) -> tuple[float, tuple | None]:
        """Presence logit of one image and, for a positive, the histograms of its 8-bit map (256 bins each over the
        ground-truth foreground and background): the full-resolution map is dropped at once, so memory stays flat."""
        x, y = read_item(root, rec, size)
        prob, logit = model.predict(x.to(device), plan["prompts"][name])
        gt = y.numpy()
        if not keep_map:
            return float(logit), None
        return float(logit), _hist(np.rint(prob * 255).astype(np.int64), gt)

    tau, binarize, report = {}, {}, {"split": "val", "presence": {}, "binarize": {}}
    for name in plan["categories"]:
        pos = val[name]
        neg = verified_absent(plan["family_of"], plan["negatives"], name, val)
        outs = [run(r, name, True) for r in pos] + [run(r, name, False) for r in neg]
        y = np.array([1] * len(pos) + [0] * len(neg))
        score = 1 / (1 + np.exp(-np.array([o[0] for o in outs], dtype=np.float64)))
        if len(pos) and len(neg):
            fit = best_tau(score, y)
            fit["source"] = "val"
        else:
            fit = {
                "tau": 0.5,
                "source": "default 0.5: no val positives and/or no verified-absent val images",
                "n_pos": len(pos),
                "n_neg": len(neg),
            }
        tau[name] = fit["tau"]
        report["presence"][name] = fit
        hists = []
        for logit, (fg_h, bg_h) in outs[: len(pos)]:
            if 1 / (1 + np.exp(-logit)) >= fit["tau"]:
                hists.append((fg_h, bg_h))
            else:  # gated off: every pixel reads 0
                zero = np.zeros(256, np.int64)
                hists.append(
                    (
                        np.where(np.arange(256) == 0, fg_h.sum(), zero),
                        np.where(np.arange(256) == 0, bg_h.sum(), zero),
                    )
                )
        if hists:
            fg, bg = (np.stack(h) for h in zip(*hists))
            t, obj = best_binarization(fg, bg, plan["kinds"][name])
            binarize[name] = t
            report["binarize"][name] = {
                "t": t,
                "objective": obj,
                "kind": plan["kinds"][name],
                "n": len(hists),
            }
        else:
            binarize[name] = 128
            report["binarize"][name] = {
                "t": 128,
                "source": "default 128: no val positives",
                "n": 0,
            }
    return tau, binarize, report
