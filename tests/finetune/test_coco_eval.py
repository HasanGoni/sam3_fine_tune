import json

import numpy as np

from sam3_fine_tune.finetune.coco_eval import evaluate_coco
from sam3_fine_tune.sambaunet.data import CocoSemanticDataset


def _oracle(ds):
    """A predictor that returns the ground truth of whichever image it is shown (looked up by file name)."""
    by_name = {ds.image_path(i).name: ds.original_mask(i) for i in range(len(ds))}
    names = ds.category_names

    def predict(image_path, prompt):
        return by_name[image_path.name][names.index(prompt)].astype(bool)
    return predict


def test_perfect_predictor_scores_100_and_writes_the_sam3_layout(coco_root, tmp_path):
    ds = CocoSemanticDataset(coco_root / "test")
    out = evaluate_coco(coco_root / "test", "annotation.json", _oracle(ds), tmp_path)
    s = json.loads((tmp_path / "multiclass_foreground_iou_summary.json").read_text())
    assert out["summary"]["overall"]["foreground_iou"] == 100.0
    assert set(s["summary"]["categories"]) == {"bond wire", "pin"}  # category_name == the text prompt
    assert s["summary"]["categories"]["pin"]["overall_foreground_iou"] == 100.0
    assert s["meta"]["model"] == "sam3" and s["meta"]["num_images"] == 2


def test_empty_predictor_scores_zero_on_present_categories(coco_root, tmp_path):
    ds = CocoSemanticDataset(coco_root / "test")
    shapes = {ds.image_path(i).name: ds.original_mask(i).shape[1:] for i in range(len(ds))}
    out = evaluate_coco(coco_root / "test", "annotation.json", lambda p, prompt: np.zeros(shapes[p.name], bool), tmp_path)
    assert out["summary"]["overall"]["foreground_iou"] == 0.0
    assert out["summary"]["overall"]["recall"] == 0.0


def test_predictor_receives_the_category_name_as_prompt(coco_root, tmp_path):
    seen = set()
    ds = CocoSemanticDataset(coco_root / "test")

    def spy(p, prompt):
        seen.add(prompt)
        return np.zeros(ds.original_mask(0).shape[1:], bool)
    evaluate_coco(coco_root / "test", "annotation.json", spy, tmp_path, max_images=1)
    assert seen == {"bond wire", "pin"}
