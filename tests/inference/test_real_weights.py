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
    reason="set SAM3_CHECKPOINT and SAM3_BPE to run against real weights",
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
