import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).parents[2]


def test_cli_adds_a_category_with_hydra_overrides(raw, tmp_path):
    imgs, masks = raw
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.dataset",
            f"images={imgs}",
            f"masks={masks}",
            "category=crack",
            f"into={tmp_path / 'ds'}",
            "test_fraction=0.25",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    assert (tmp_path / "ds" / "train" / "annotation.json").exists()
    assert "mode: new" in p.stdout and not (tmp_path / "outputs").exists()


def test_cli_dry_run_changes_nothing(raw, tmp_path):
    imgs, masks = raw
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sam3_fine_tune.dataset",
            f"images={imgs}",
            f"masks={masks}",
            "category=crack",
            f"into={tmp_path / 'ds'}",
            "dry_run=true",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert p.returncode == 0 and not (tmp_path / "ds").exists()


def test_console_script_is_registered_in_pyproject():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["sam3-dataset"] == "sam3_fine_tune.dataset.cli:main"


def test_cli_prints_the_duplicate_image_warning_on_stderr(raw, tmp_path):
    imgs, masks = raw

    def run(category):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "sam3_fine_tune.dataset",
                f"images={imgs}",
                f"masks={masks}",
                f"category={category}",
                f"into={tmp_path / 'ds'}",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )

    assert "sam3-prepare" not in run("substrate").stderr
    p = run("chip")
    assert p.returncode == 0 and "sam3-prepare" in p.stderr
