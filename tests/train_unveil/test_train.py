"""`sam3-train-unveil train=true`: a real training loop, val-only thresholds and a safetensors checkpoint (stub model)."""

import json

import pytest
import yaml
from safetensors import safe_open

from sam3_fine_tune.train_unveil.checkpoint import read_checkpoint
from tests.prepare.conftest import BOX, Root, run_prepare, square
from tests.train_unveil.test_validate import run_train_unveil, write_recipe

TRAIN = (
    "validate_only=false",
    "train=true",
    "model=stub",
    "device=cpu",
    "image_size=48",
    "steps=6",
)
CATS = ("chip", "substrate", "wire")


def val_root(tmp_path):
    """Chip + substrate (one family) and wire (another): 14 train pictures, a val split, no test image file."""
    root = Root(tmp_path / "ds")
    for i in range(1, 9):
        root.add(
            "train", f"c{i}.png", i, {"chip": BOX, "substrate": square(4, 4, 60, 60)}
        )
    for i in range(9, 15):
        root.add("train", f"w{i}.png", i, {"wire": BOX})
    root.add("test", "t.png", 99, {"chip": BOX}, write_file=False)
    path = root.write()
    assert run_prepare(path, "val_fraction=0.3").returncode == 0
    write_recipe(path, epochs=1)
    return path


def without_steps():
    return [a for a in TRAIN if not a.startswith("steps")]


def train(path, out, *extra):
    p = run_train_unveil(path, *TRAIN, f"out={out}", *extra)
    assert p.returncode == 0, p.stderr
    return yaml.safe_load(p.stdout)["train"]


def val_records(path, name):
    f = path / "prepared/unveil/index/val" / f"{name}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x]


def test_a_short_run_ends_with_a_checkpoint_and_a_log(tmp_path):
    path = val_root(tmp_path)
    result = train(path, tmp_path / "run")
    assert result["steps"] == 6
    assert (tmp_path / "run" / "unveil.safetensors").is_file()
    lines = (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    log = [json.loads(x) for x in lines]
    assert [r["step"] for r in log] == list(range(1, 7))
    assert all(r["loss"] == r["loss"] for r in log)  # no NaN


def test_the_checkpoint_is_safetensors_with_thresholds_per_category(tmp_path):
    path = val_root(tmp_path)
    train(path, tmp_path / "run")
    ckpt = tmp_path / "run" / "unveil.safetensors"
    tensors, meta = read_checkpoint(ckpt)
    assert meta["format"] == "unveil-checkpoint"
    assert set(meta["thresholds"]["single"]) == set(CATS)
    assert all(0 < t < 1 for t in meta["thresholds"]["single"].values())
    assert set(meta["binarize"]) == set(CATS)
    assert all(1 <= t <= 255 for t in meta["binarize"].values())
    assert any(k.startswith("heads.") for k in tensors)
    assert "joint.convention" in tensors
    assert (tmp_path / "run" / "unveil.safetensors.sha256").is_file()
    with safe_open(str(ckpt), framework="pt") as f:
        assert "unveil" in f.metadata()


def test_thresholds_come_from_the_val_split_never_the_test_split(tmp_path):
    path = val_root(
        tmp_path
    )  # the test image file does not exist: reading it would crash
    train(path, tmp_path / "run")
    report = json.loads((tmp_path / "run" / "calibration.json").read_text())
    assert report["split"] == "val"
    for name in CATS:
        assert report["presence"][name]["n_pos"] == len(val_records(path, name))
    # wire images are verified absent for the chip family and the other way round
    assert report["presence"]["chip"]["n_neg"] == len(val_records(path, "wire"))
    chip_ids = {r["id"] for n in ("chip", "substrate") for r in val_records(path, n)}
    assert report["presence"]["wire"]["n_neg"] == len(chip_ids)


def test_a_root_without_val_images_stops_with_the_fix(tmp_path):
    root = Root(tmp_path / "ds")
    root.add("train", "a.png", 1, {"chip": BOX}).add("train", "b.png", 2, {"wire": BOX})
    path = root.write()
    assert run_prepare(path, "val_fraction=0").returncode == 0
    write_recipe(
        path,
        categories={"chip": {"kind": "region"}, "wire": {"kind": "balanced"}},
        families={"c": ["chip"], "w": ["wire"]},
        epochs=1,
    )
    p = run_train_unveil(path, *TRAIN, f"out={tmp_path / 'run'}")
    assert p.returncode != 0
    assert "val" in p.stderr and "val_fraction" in p.stderr


def test_epochs_from_the_recipe_set_the_number_of_steps(tmp_path):
    path = val_root(tmp_path)
    p = run_train_unveil(path, *without_steps(), f"out={tmp_path / 'r'}")
    assert p.returncode == 0, p.stderr
    out = yaml.safe_load(p.stdout)["train"]
    # 1 epoch x distinct train images / images per step (8 per category x 3 = 24)
    assert out["epochs"] == 1 and out["steps"] >= 1


def test_without_steps_or_epochs_the_run_stops_with_the_fix(tmp_path):
    path = val_root(tmp_path)
    write_recipe(path)  # no epochs
    p = run_train_unveil(path, *without_steps(), f"out={tmp_path / 'r'}")
    assert p.returncode != 0
    assert "epochs" in p.stderr and "steps" in p.stderr


def test_the_same_seed_gives_the_same_checkpoint(tmp_path):
    path = val_root(tmp_path)
    train(path, tmp_path / "a")
    train(path, tmp_path / "b")
    a, _ = read_checkpoint(tmp_path / "a" / "unveil.safetensors")
    b, _ = read_checkpoint(tmp_path / "b" / "unveil.safetensors")
    assert all((a[k] == b[k]).all() for k in a)


def test_unveils_own_loader_opens_the_checkpoint(tmp_path):
    unveil_ckpt = pytest.importorskip("unveil.checkpoint")
    path = val_root(tmp_path)
    train(path, tmp_path / "run")
    tensors, meta, _ = unveil_ckpt.load(tmp_path / "run" / "unveil.safetensors")
    assert set(meta["thresholds"]["single"]) == set(CATS)
    assert "joint.convention" in tensors


def test_steps_must_be_positive(tmp_path):
    path = val_root(tmp_path)
    args = [a for a in TRAIN if not a.startswith("steps")]
    p = run_train_unveil(path, *args, "steps=0", f"out={tmp_path / 'run'}")
    assert p.returncode != 0
    assert "steps" in p.stderr and "above 0" in p.stderr


def test_peak_memory_is_read_before_calibration(tmp_path, monkeypatch):
    from sam3_fine_tune.train_unveil import calibrate as cal, trainer
    from sam3_fine_tune.train_unveil.ops import validate_recipe
    from sam3_fine_tune.train_unveil.streams import load_index

    path = val_root(tmp_path)
    plan = validate_recipe(path)
    index = load_index(path, plan)
    events = []
    real_peak, real_cal = trainer._peak_gb, cal.calibrate
    monkeypatch.setattr(trainer, "_peak_gb", lambda d: events.append("peak") or 1.0)
    monkeypatch.setattr(
        trainer, "calibrate", lambda *a, **k: events.append("cal") or real_cal(*a, **k)
    )
    trainer.run_training(plan, index, path, tmp_path / "r", 2, "stub", "cpu", 48)
    assert events == ["peak", "cal"]
    assert real_peak("cpu") > 0
