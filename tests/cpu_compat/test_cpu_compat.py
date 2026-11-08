"""Device handling must work on machines without a GPU. Run with CUDA_VISIBLE_DEVICES="" to prove it for real."""
import types

import pytest
import torch

from sam3.train.utils.train_utils import MemMeter, resolve_accelerator


def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_resolve_accelerator_auto_follows_availability(monkeypatch):
    _no_cuda(monkeypatch)
    assert resolve_accelerator("auto") == "cpu"
    _with_cuda(monkeypatch)
    assert resolve_accelerator("auto") == "cuda"


def test_resolve_accelerator_keeps_explicit_choices(monkeypatch):
    _with_cuda(monkeypatch)
    assert resolve_accelerator("cpu") == "cpu"  # cpu on a GPU machine is allowed (debugging)
    assert resolve_accelerator("cuda") == "cuda"


def test_cuda_requested_without_gpu_fails_early_with_a_clear_message(monkeypatch):
    _no_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match=r"accelerator.*cpu.*auto"):
        resolve_accelerator("cuda")


def test_unknown_accelerator_is_rejected():
    with pytest.raises(ValueError, match="mps"):
        resolve_accelerator("mps")


def test_nccl_backend_falls_back_to_gloo_on_cpu():
    from sam3.train.trainer import Trainer

    conf = types.SimpleNamespace(backend="nccl")
    Trainer._infer_distributed_backend_if_none(None, conf, "cpu")
    assert conf.backend == "gloo"
    conf = types.SimpleNamespace(backend="nccl")
    Trainer._infer_distributed_backend_if_none(None, conf, "cuda")
    assert conf.backend == "nccl"  # GPU runs are untouched
    conf = types.SimpleNamespace(backend=None)
    Trainer._infer_distributed_backend_if_none(None, conf, "cpu")
    assert conf.backend == "gloo"


def test_mem_meter_does_not_touch_cuda_on_cpu(monkeypatch):
    _no_cuda(monkeypatch)
    m = MemMeter("Mem (GB)", torch.device("cpu"), ":.2f")
    m.update(reset_peak_usage=True)  # used to raise: torch.cuda.max_memory_allocated needs a GPU
    assert m.val == 0 and m.peak == 0
    assert "Mem (GB)" in str(m)


def test_object_gather_device_is_cpu_without_gpu(monkeypatch):
    from sam3.train.utils.distributed import _object_gather_device

    _no_cuda(monkeypatch)
    assert _object_gather_device(cpu_group=None) == "cpu"
    _with_cuda(monkeypatch)
    assert _object_gather_device(cpu_group=None) == "cuda"
    assert _object_gather_device(cpu_group=object()) == "cpu"


def test_dummy_loss_default_device_works_without_gpu(monkeypatch):
    from sam3.train.loss.sam3_loss import DummyLoss

    _no_cuda(monkeypatch)
    out = DummyLoss()()
    assert next(iter(out.values())).device.type == "cpu"


@pytest.mark.parametrize("reduce,multimask", [(True, False), (True, True), (False, False)])
def test_sigmoid_focal_loss_runs_on_cpu_and_matches_the_formula(reduce, multimask):
    from sam3.train.loss.loss_fns import sigmoid_focal_loss

    torch.manual_seed(0)
    shape = (2, 3, 4, 5) if multimask else (4, 6)
    x, y = torch.randn(*shape), (torch.rand(*shape) > 0.5).float()  # CPU tensors: used to hit the Triton kernel
    got = sigmoid_focal_loss(x, y, num_boxes=2, alpha=0.25, gamma=2, loss_on_multimask=multimask, reduce=reduce)
    p = x.sigmoid(); pt = p * y + (1 - p) * (1 - y)
    ref = 0.25 * y + 0.75 * (1 - y)
    ref = ref * torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction="none") * (1 - pt) ** 2
    assert torch.isfinite(got).all()
    if not reduce:
        torch.testing.assert_close(got, ref)
    elif multimask:
        torch.testing.assert_close(got, ref.flatten(2).mean(-1) / 2)
    else:
        torch.testing.assert_close(got, ref.mean(1).sum() / 2)
