"""Offline unit tests: a tiny randomly initialised SAM encoder replaces facebook/sam-vit-base."""
import pytest
import torch

pytest.importorskip("transformers")

from sam3_fine_tune.sambaunet.model import (  # noqa: E402
    SAMBAUNet,
    bce_dice_loss,
    selective_scan_assoc,
    selective_scan_ref,
    selective_scan_seq,
    tiny_sam_config,
)

DIMS = (8, 16, 32, 64)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SAMBAUNet(n_classes=3, dims=DIMS, sam_config=tiny_sam_config())


def test_chunked_scan_matches_sequential_for_any_length():
    torch.manual_seed(0)
    b, d, n = 2, 4, 3
    for L in (16, 64, 100, 256):
        u, delta = torch.randn(b, d, L), torch.rand(b, d, L) * 0.1
        A = -torch.rand(d, n) - 0.1
        B, C, D = torch.randn(b, n, L), torch.randn(b, n, L), torch.randn(d)
        torch.testing.assert_close(
            selective_scan_ref(u, delta, A, B, C, D),
            selective_scan_seq(u, delta, A, B, C, D),
            atol=1e-4,
            rtol=1e-4,
        )


def test_forward_returns_one_logit_channel_per_category(model):
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)


def test_sam_branch_is_identity_to_vanilla_sam_at_init(model):
    x = torch.rand(1, 3, 64, 64)
    x = (x - model.sam_mean) / model.sam_std
    with_lora = model.sam(x)
    # strip LoRA/adapters by zeroing is already the init state -> compare with a pristine copy
    pristine = model.sam.pristine_forward(x)
    for k in with_lora:
        torch.testing.assert_close(with_lora[k], pristine[k])


def test_only_cose_mamba_and_hcem_are_trainable(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing trainable"
    for n in trainable:
        assert not n.startswith("sam.enc.") or any(
            t in n for t in ("down_", "up_", "adapter")
        ), n


def test_backward_reaches_lora_adapter_hcem_and_mamba(model):
    model.zero_grad()
    x, y = torch.rand(2, 3, 64, 64), (torch.rand(2, 3, 64, 64) > 0.5).float()
    bce_dice_loss(model(x), y).backward()
    # LoRA "up" and adapter "up" start at zero, so only they (and everything downstream) get gradient first
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "adapter.up" in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.hcem.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.enc.named_parameters())
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)


def test_bce_dice_loss_is_lower_for_a_perfect_prediction():
    y = (torch.rand(1, 2, 8, 8) > 0.5).float()
    perfect, wrong = (y * 20 - 10), (-(y * 20 - 10))
    assert bce_dice_loss(perfect, y) < bce_dice_loss(wrong, y)


def test_parallel_scan_matches_sequential_for_all_four_directions():
    torch.manual_seed(0)
    b, K, d, n, L = 2, 4, 3, 5, 100
    u, delta = torch.randn(b, K, d, L), torch.rand(b, K, d, L) * 0.2
    A, B, C, D = -torch.rand(K, d, n) - 0.1, torch.randn(b, K, n, L), torch.randn(b, K, n, L), torch.randn(K, d)
    got = selective_scan_assoc(u, delta, A, B, C, D)
    for k in range(K):
        torch.testing.assert_close(got[:, k], selective_scan_seq(u[:, k], delta[:, k], A[k], B[:, k], C[:, k], D[k]), atol=1e-4, rtol=1e-4)


def test_gradient_checkpointing_gives_identical_gradients():
    torch.manual_seed(0)
    plain = SAMBAUNet(n_classes=2, dims=DIMS, sam_config=tiny_sam_config())
    ckpt = SAMBAUNet(n_classes=2, dims=DIMS, sam_config=tiny_sam_config(), grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    x, y = torch.rand(1, 3, 64, 64), (torch.rand(1, 2, 64, 64) > 0.5).float()
    for m in (plain, ckpt):
        m.train(); bce_dice_loss(m(x), y).backward()
    a = dict(plain.named_parameters())
    for n, p in ckpt.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(p.grad, a[n].grad, atol=1e-5, rtol=1e-4)
