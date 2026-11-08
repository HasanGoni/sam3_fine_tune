from pathlib import Path

import pytest

from sam3_fine_tune.finetune import cli


@pytest.fixture()
def fake(monkeypatch, coco_root, tmp_path):
    calls = {}

    def fake_train(cfg_path, num_gpus, accelerator):
        calls["train"] = (Path(cfg_path).read_text(), num_gpus, accelerator)
        calls["cfg_existed"] = Path(cfg_path).exists()
        ck = Path(calls["exp"]) / "checkpoints"; ck.mkdir(parents=True)
        (ck / "checkpoint.pt").write_text("x")

    def fake_eval(**kw):
        calls["eval"] = kw
        return {"summary": {"overall": {"foreground_iou": 12.5}}}

    monkeypatch.setattr(cli, "run_training", fake_train)
    monkeypatch.setattr(cli, "run_evaluation", fake_eval)
    args = ["--data-root", str(coco_root), "--sam3-checkpoint", "/w/sam3.pt", "--bpe-path", "/w/bpe.txt.gz",
            "--experiment-dir", str(tmp_path / "exp"), "--epochs", "1", "--accelerator", "cpu"]
    calls["exp"] = tmp_path / "exp"
    return calls, args


def test_train_then_evaluate_by_default(fake):
    calls, args = fake
    assert cli.main(args) == 0
    assert "train" in calls and calls["eval"]["checkpoint"].name == "checkpoint.pt"
    assert calls["eval"]["split_dir"].name == "test" and calls["eval"]["accelerator"] == "cpu"
    assert (Path(calls["exp"]) / "config_used.yaml").exists()  # the generated config is kept with the run


def test_generated_config_is_not_left_in_the_configs_dir(fake):
    calls, args = fake
    cli.main(args)
    assert calls["cfg_existed"] is True
    assert not list(cli.CONFIGS_DIR.glob("_gen_*.yaml"))


def test_skip_eval_only_trains(fake):
    calls, args = fake
    cli.main(args + ["--skip-eval"])
    assert "train" in calls and "eval" not in calls


def test_eval_only_uses_an_existing_run(fake, tmp_path):
    calls, args = fake
    ck = tmp_path / "exp" / "checkpoints"; ck.mkdir(parents=True); (ck / "checkpoint.pt").write_text("x")
    cli.main(args + ["--eval-only"])
    assert "train" not in calls and "eval" in calls


def test_bad_dataset_fails_before_training(fake, tmp_path):
    calls, args = fake
    args[1] = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        cli.main(args)
    assert "train" not in calls
