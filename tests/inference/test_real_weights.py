"""Opt-in check against the real SAM3 weights. Skipped unless both env vars are set (e.g. on the HPC):

    SAM3_CHECKPOINT=/path/sam3.pt SAM3_BPE=/path/bpe_simple_vocab_16e6.txt.gz \
        .venv/bin/python -m pytest tests/inference/test_real_weights.py -q
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam3_fine_tune.inference.run import (
    make_predictor,
    parse_prompts,
    resolve_device,
    run_inference,
)

CKPT, BPE = os.environ.get("SAM3_CHECKPOINT"), os.environ.get("SAM3_BPE")
pytestmark = pytest.mark.skipif(
    not (CKPT and BPE and Path(CKPT).is_file() and Path(BPE).is_file()),
    reason="set SAM3_CHECKPOINT and SAM3_BPE to run against real weights (local copies: see CLAUDE.md, Local weights)",
)


def test_two_prompts_with_different_thresholds_on_real_model(tmp_path):
    import torch

    img = tmp_path / "imgs" / "x.png"
    img.parent.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(img)
    device = resolve_device("auto", torch.cuda.is_available())
    predictor = make_predictor(CKPT, BPE, device)
    encodes = []
    real_encode = predictor.encode
    predictor = predictor._replace(
        encode=lambda p: (encodes.append(p), real_encode(p))[1]
    )
    specs = parse_prompts(
        None, [{"text": "pin", "threshold": 0.3}, {"text": "crack"}], 0.5
    )
    summary = run_inference([img], specs, tmp_path / "out", predictor)
    assert len(encodes) == 1  # backbone once for both prompts
    rows = summary["images"][0]["prompts"]
    assert [r["threshold"] for r in rows] == [0.3, 0.5]
    for r in rows:  # kept instances must respect each prompt's own threshold
        assert all(s > r["threshold"] for s in r["scores"])
    assert (tmp_path / "out" / "masks" / "pin" / "x.png").is_file()
    assert (tmp_path / "out" / "masks" / "crack" / "x.png").is_file()


def test_threshold_sweep_on_real_model_is_monotonic(tmp_path):
    import torch

    img = tmp_path / "x.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(img)
    predictor = make_predictor(
        CKPT, BPE, resolve_device("auto", torch.cuda.is_available())
    )
    specs = parse_prompts(None, [{"text": "pin", "threshold": [0.05, 0.2, 0.5]}], 0.5)
    summary = run_inference([img], specs, tmp_path / "out", predictor)
    counts = [lvl["instances"] for lvl in summary["images"][0]["sweep"]]
    assert counts == sorted(counts, reverse=True)
    assert (tmp_path / "out" / "masks" / "thr_0.05" / "x.png").is_file()


def test_scoring_matches_the_finetune_eval_path_on_the_real_base_model(
    coco_root, tmp_path
):
    """`sam3-infer coco_split=...` vs the `sam3-finetune --eval-only` predictor logic, same weights/threshold/images."""
    import torch
    from omegaconf import OmegaConf

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3_fine_tune.finetune.coco_eval import evaluate_coco
    from sam3_fine_tune.inference.cli import run

    device = resolve_device("auto", torch.cuda.is_available())
    cfg = OmegaConf.create(
        dict(
            images=None,
            coco_split=str(coco_root / "test"),
            coco_ann="annotation.json",
            prompt=None,
            prompts=[{"text": "bond wire"}, {"text": "pin"}],
            out=str(tmp_path / "out"),
            sam3_checkpoint=CKPT,
            bpe_path=BPE,
            checkpoint=None,
            threshold=0.5,
            device=device,
        )
    )
    ours = run(cfg)["evaluation"]

    model = build_sam3_image_model(
        bpe_path=BPE, device=device, eval_mode=True, load_from_HF=False,
        checkpoint_path=CKPT,
    )  # fmt: skip
    processor = Sam3Processor(model, confidence_threshold=0.5)

    @torch.no_grad()
    def reference(
        image_path, prompt
    ):  # same steps as finetune.coco_eval.make_sam3_predictor
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        out = np.zeros((image.height, image.width), bool)
        for m in state.get("masks", []):
            out |= m.cpu().numpy().squeeze() > 0.5
        return out

    ref = evaluate_coco(
        coco_root / "test", "annotation.json", reference, tmp_path / "ref"
    )
    assert ours == ref["summary"]
