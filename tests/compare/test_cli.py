"""`sam3-compare` run as a user would: a prepared root plus saved predictions in, a table and a JSON file out."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).parents[2]
SIZE = 64


def box(x0, y0, x1, y1):
    m = np.zeros((SIZE, SIZE), bool)
    m[y0:y1, x0:x1] = True
    return m


def polygon(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def save_mask(folder, stem, mask):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{stem}.png")


# test split: file -> ({category: ground-truth box}, categories flagged not_annotated)
TEST_IMAGES = {
    "a": ({"bond wire": (10, 10, 30, 30), "pin": (40, 40, 50, 50)}, []),
    "b": ({"bond wire": (0, 0, 20, 20)}, []),
    "c": ({"pin": (10, 10, 20, 20)}, ["bond wire"]),
    "d": ({"bond wire": (30, 30, 50, 50)}, ["pin"]),
}
CATEGORIES = ["bond wire", "pin", "crack"]  # crack has no test image


@pytest.fixture()
def prepared(tmp_path):
    """`<root>/prepared/sam3/test/` as sam3-prepare writes it, built by hand so the flags are exact."""
    root = tmp_path / "root"
    split = root / "prepared" / "sam3" / "test"
    (split / "images").mkdir(parents=True)
    ids = {n: k + 1 for k, n in enumerate(CATEGORIES)}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for n, i in ids.items()],
    }
    for k, (stem, (gt, flagged)) in enumerate(TEST_IMAGES.items(), 1):
        Image.fromarray(np.zeros((SIZE, SIZE, 3), np.uint8)).save(
            split / "images" / f"{stem}.png"
        )
        entry = {"id": k, "file_name": f"{stem}.png", "width": SIZE, "height": SIZE}
        if flagged:
            entry["not_annotated_category_ids"] = [ids[n] for n in flagged]
        coco["images"].append(entry)
        for cat, b in gt.items():
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": k,
                    "category_id": ids[cat],
                    "segmentation": [polygon(*b)],
                    "area": 1.0,
                    "bbox": [0, 0, 1, 1],
                    "iscrowd": 0,
                }
            )
    (split / "annotation_merged.json").write_text(json.dumps(coco))
    (root / "prepared" / "manifest.json").write_text(
        json.dumps({"version": 1, "categories": ids})
    )
    return root


def write_predictions(folder, per_image):
    """sam3-infer layout with several prompts: masks/<slug>/<stem>.png. per_image: {stem: {slug: mask}}."""
    for stem, cats in per_image.items():
        for slug, mask in cats.items():
            save_mask(folder / "masks" / slug, stem, mask)
    return folder


@pytest.fixture()
def partial_model(tmp_path):
    """Exact on bond wire in a and b, half of d; a bogus mask on c (not annotated for bond wire) must be ignored.
    Pin: exact in a, shifted by half in c, a bogus mask on d (not annotated for pin)."""
    return write_predictions(
        tmp_path / "pred_partial",
        {
            "a": {"bond_wire": box(10, 10, 30, 30), "pin": box(40, 40, 50, 50)},
            "b": {"bond_wire": box(0, 0, 20, 20), "pin": box(0, 0, 0, 0)},
            "c": {"bond_wire": box(0, 0, 20, 20), "pin": box(15, 10, 25, 20)},
            "d": {"bond_wire": box(30, 30, 50, 40), "pin": box(0, 0, 20, 20)},
        },
    )


@pytest.fixture()
def empty_model(tmp_path):
    return write_predictions(
        tmp_path / "pred_empty",
        {s: {"bond_wire": box(0, 0, 0, 0), "pin": box(0, 0, 0, 0)} for s in "abcd"},
    )


def run_compare(root, models, out, *overrides):
    spec = (
        "["
        + ",".join(f"{{name: {n}, kind: predictions, path: {p}}}" for n, p in models)
        + "]"
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.compare",
            f"root={root}",
            f"models={spec}",
            f"out={out}",
            *overrides,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_pooled_iou_per_category_with_na_and_not_annotated_skip(
    prepared, partial_model, empty_model, tmp_path
):
    out = tmp_path / "result.json"
    p = run_compare(prepared, [("partial", partial_model), ("empty", empty_model)], out)
    assert p.returncode == 0, p.stderr[-2000:]
    result = json.loads(out.read_text())["models"]
    wire, pin, crack = (result["partial"][c] for c in ("bond wire", "pin", "crack"))
    # bond wire is scored on a, b, d (c is not_annotated): inter 400+400+200, union 400+400+400
    assert wire["images"] == 3 and wire["iou"] == pytest.approx(1000 / 1200)
    # pin is scored on a, c (d is not_annotated): inter 100+50, union 100+150
    assert pin["images"] == 2 and pin["iou"] == pytest.approx(150 / 250)
    # no test image for crack: not scored, not 0
    assert crack == {"images": 0, "iou": None, "flag": None}
    assert result["empty"]["bond wire"]["iou"] == 0.0
    assert result["empty"]["crack"]["iou"] is None
    assert "0.833" in p.stdout and "0.600" in p.stdout
    crack_row = [line for line in p.stdout.splitlines() if "crack" in line][0]
    assert crack_row.split().count("n/a") == 2  # one per model
    wire_row = [line for line in p.stdout.splitlines() if "bond wire" in line][0]
    assert " 3 " in f"{wire_row} "  # the image count is in the table


def test_subset_is_seeded_and_the_full_split_is_selectable(
    prepared, partial_model, tmp_path
):
    def counts(*overrides):
        out = tmp_path / "r.json"
        p = run_compare(prepared, [("m", partial_model)], out, *overrides)
        assert p.returncode == 0, p.stderr[-2000:]
        data = json.loads(out.read_text())
        return data, {c: v["images"] for c, v in data["models"]["m"].items()}

    full_data, full = counts()
    assert full == {"bond wire": 3, "pin": 2, "crack": 0}
    assert full_data["subset"] is None
    first, n1 = counts("subset=2", "seed=1")
    again, n2 = counts("subset=2", "seed=1")
    assert first == again and first["subset"] == {"images": 2, "seed": 1}
    assert sum(n1.values()) < sum(full.values())
    # some seed picks a different pair of images
    others = [counts("subset=2", f"seed={s}")[0]["models"] for s in range(2, 8)]
    assert any(o != first["models"] for o in others)


def test_bad_model_entry_fails_with_the_fix(prepared, partial_model, tmp_path):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.compare",
            f"root={prepared}",
            "models=[{name: x, kind: nonsense}]",
            f"out={tmp_path / 'r.json'}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert p.returncode != 0 and "predictions" in p.stderr and "kind" in p.stderr


def run_compare_spec(root, spec, out):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.compare",
            f"root={root}",
            f"models={spec}",
            f"out={out}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_near_zero_on_a_trained_category_is_suspect_and_untrained_is_marked(
    prepared, empty_model, partial_model, tmp_path
):
    """empty scores 0.000 on bond wire and pin but was trained on bond wire (and crack) only: its bond wire zero is
    suspect (a prompt or threshold problem, not a weakness) and pin is not trained. partial declares no training
    categories and scores well, so it carries no mark."""
    out = tmp_path / "r.json"
    spec = (
        f"[{{name: empty, kind: predictions, path: {empty_model}, "
        f"trained_categories: [bond wire, crack]}},"
        f"{{name: partial, kind: predictions, path: {partial_model}}}]"
    )
    p = run_compare_spec(prepared, spec, out)
    assert p.returncode == 0, p.stderr[-2000:]
    models = json.loads(out.read_text())["models"]
    assert models["empty"]["bond wire"]["flag"] == "suspect"
    assert models["empty"]["pin"]["flag"] == "not trained"
    assert models["empty"]["crack"]["flag"] is None  # trained, no test image: n/a
    assert models["partial"]["bond wire"]["flag"] is None
    rows = {
        c: [line for line in p.stdout.splitlines() if line.startswith(c)][0]
        for c in ("bond wire", "pin", "crack")
    }
    assert "suspect" in rows["bond wire"] and "not trained" in rows["pin"]
    assert "suspect" not in rows["crack"] and "not trained" not in rows["crack"]


def test_trained_spelling_is_used_as_the_prompt():
    """thin_wire_panda scored 0.000 because the model was trained on the text 'thin wire panda'."""
    from sam3_fine_tune.compare.ops import prompt_for

    trained = ["et_pin", "thin wire panda"]
    assert prompt_for("thin_wire_panda", trained) == "thin wire panda"
    assert prompt_for("et_pin", trained) == "et_pin"
    assert prompt_for("crack", trained) == "crack"
    assert prompt_for("thin_wire_panda", None) == "thin_wire_panda"
