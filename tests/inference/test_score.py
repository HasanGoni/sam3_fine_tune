import json

import pytest
from omegaconf import OmegaConf

from sam3_fine_tune.finetune.coco_eval import evaluate_coco
from sam3_fine_tune.inference.cli import run
from sam3_fine_tune.inference.run import (
    model_label,
    normalize_checkpoints,
    Predictor,
)
from sam3_fine_tune.inference.score import format_comparison, match_prompts
from sam3_fine_tune.sambaunet.data import CocoSemanticDataset

SUMMARY = "multiclass_foreground_iou_summary.json"


def oracle(ds, empty=False):
    """Predictor that returns the COCO ground truth of the image (or nothing), looked up by file name."""
    by_name = {ds.image_path(i).name: ds.original_mask(i) for i in range(len(ds))}
    names = ds.category_names

    def segment(path, spec):
        if spec.text not in names:
            return [], []
        gt = by_name[path.name][names.index(spec.text)].astype(bool)
        if empty or not gt.any():
            return [], []
        return [gt], [0.9]

    return Predictor(lambda p: p, segment)


def cfg(tmp_path, coco_root, **kw):
    (tmp_path / "sam3.pt").write_bytes(b"x")
    (tmp_path / "bpe.gz").write_bytes(b"x")
    base = dict(
        images=None,
        coco_split=str(coco_root / "test"),
        coco_ann="annotation.json",
        prompt=None,
        prompts=[{"text": "bond wire"}, {"text": "pin"}],
        out=str(tmp_path / "out"),
        sam3_checkpoint=str(tmp_path / "sam3.pt"),
        bpe_path=str(tmp_path / "bpe.gz"),
        checkpoint=None,
        threshold=0.5,
        device="cpu",
    )
    return OmegaConf.create({**base, **kw})


def test_scores_match_the_finetune_evaluation_of_the_same_predictions(
    coco_root, tmp_path
):
    ds = CocoSemanticDataset(coco_root / "test")
    s = run(cfg(tmp_path, coco_root), predictor=oracle(ds))
    ours = json.loads((tmp_path / "out" / SUMMARY).read_text())
    by_name = {ds.image_path(i).name: ds.original_mask(i) for i in range(len(ds))}
    ref = evaluate_coco(  # the sam3-finetune scoring path, fed the same (ground-truth) predictions
        coco_root / "test",
        "annotation.json",
        lambda p, prompt: by_name[p.name][ds.category_names.index(prompt)].astype(bool),
        tmp_path / "ref",
    )
    assert ours["summary"] == ref["summary"]
    assert ours["per_category_details"] == ref["per_category_details"]
    assert ours["summary"]["overall"]["foreground_iou"] == 100.0
    assert set(ours) == {"summary", "per_category_details", "meta"}
    assert ours["meta"]["model"] == "sam3" and ours["meta"]["variant"] == "base"
    assert s["evaluation"] == ours["summary"]


def test_prompt_and_category_mismatches_are_reported_not_skipped(
    coco_root, tmp_path, capsys
):
    ds = CocoSemanticDataset(coco_root / "test")
    c = cfg(tmp_path, coco_root, prompts=[{"text": "bond wire"}, {"text": "gizmo"}])
    s = run(c, predictor=oracle(ds))
    err = capsys.readouterr().err
    assert "gizmo" in err and "pin" in err and "not scored" in err
    assert s["coco"] == {
        "prompts_without_category": ["gizmo"],
        "categories_without_prompt": ["pin"],
    }
    on_disk = json.loads((tmp_path / "out" / SUMMARY).read_text())
    assert list(on_disk["summary"]["categories"]) == ["bond wire"]
    assert on_disk["meta"]["prompts_without_category"] == ["gizmo"]


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(prompts=[{"text": "gizmo"}]), "no prompt matches a category"),
        (dict(images="x"), "not both"),
        (
            dict(prompts=[{"text": "pin", "threshold": [0.2, 0.5]}]),
            "one threshold per prompt",
        ),
        (dict(coco_ann="nope.json"), "annotation not found"),
    ],
)
def test_bad_coco_config_fails_before_model_load(coco_root, tmp_path, kw, msg):
    with pytest.raises(SystemExit, match=msg):
        run(cfg(tmp_path, coco_root, **kw))  # predictor=None would load the real model
    assert not (tmp_path / "out").exists()


def test_checkpoint_list_with_base_gives_comparison_and_one_summary_per_model(
    coco_root, tmp_path
):
    ds = CocoSemanticDataset(coco_root / "test")
    ft = tmp_path / "exp1" / "checkpoints" / "checkpoint.pt"
    ft.parent.mkdir(parents=True)
    ft.write_bytes(b"x")
    seen = []

    def factory(ck):
        seen.append(ck)
        return oracle(ds, empty=ck is None)  # base finds nothing, fine-tuned is perfect

    s = run(cfg(tmp_path, coco_root, checkpoint=[None, str(ft)]), predictor=factory)
    assert seen == [None, ft]
    assert list(s["models"]) == ["base", "exp1_checkpoint"]
    for label in s["models"]:
        assert (tmp_path / "out" / label / SUMMARY).is_file()
        assert (tmp_path / "out" / label / "summary.json").is_file()
    ev = {m: x["evaluation"] for m, x in s["models"].items()}
    assert ev["base"]["overall"]["foreground_iou"] == 0.0
    assert ev["exp1_checkpoint"]["overall"]["foreground_iou"] == 100.0
    table = format_comparison(ev)
    lines = table.splitlines()
    assert "base IoU" in lines[0] and "exp1_checkpoint R" in lines[0]
    assert lines[-1].split()[0] == "ALL" and "100.00" in lines[-1] and "0.00" in lines[-1]
    assert any(l.startswith("bond wire") for l in lines)


def test_single_checkpoint_keeps_the_flat_layout(coco_root, tmp_path):
    ds = CocoSemanticDataset(coco_root / "test")
    s = run(cfg(tmp_path, coco_root), predictor=oracle(ds))
    assert "models" not in s and (tmp_path / "out" / SUMMARY).is_file()


def test_checkpoint_list_validation_and_labels(tmp_path):
    a = tmp_path / "e" / "checkpoints" / "checkpoint.pt"
    a.parent.mkdir(parents=True)
    a.write_bytes(b"x")
    assert model_label(None) == "base" and model_label(a) == "e_checkpoint"
    assert normalize_checkpoints([None, str(a)]) == [None, a]
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_checkpoints([])
    with pytest.raises(ValueError, match="share an output name"):
        normalize_checkpoints([None, None])
    with pytest.raises(ValueError, match="not found"):
        normalize_checkpoints([None, str(tmp_path / "nope.pt")])


def test_match_prompts_orders_by_dataset_category():
    from sam3_fine_tune.inference.run import PromptSpec

    m = match_prompts(
        [PromptSpec("pin", (0.5,)), PromptSpec("x", (0.5,))], ["bond wire", "pin"]
    )
    assert [(s.text, k) for s, k in m.scored] == [("pin", 1)]
    assert m.prompts_without_category == ["x"]
    assert m.categories_without_prompt == ["bond wire"]
