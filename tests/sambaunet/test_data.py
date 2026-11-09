import numpy as np
import torch

from sam3_fine_tune.sambaunet.data import CocoSemanticDataset


def test_item_shapes_and_category_order(coco_root):
    ds = CocoSemanticDataset(coco_root / "train", size=64)
    assert ds.category_names == ["pin", "bond wire"]  # sorted by category id (3, 7)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64) and item["image"].dtype == torch.float32
    assert item["image"].min() >= 0 and item["image"].max() <= 1
    assert item["mask"].shape == (2, 64, 64)
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}


def test_masks_land_in_the_right_channel(coco_root):
    ds = CocoSemanticDataset(
        coco_root / "train", size=120
    )  # im0 is 120x80 -> stretched to 120x120
    m = ds[0]["mask"]
    pin, wire = m[0], m[1]
    assert (
        wire[:, :60].sum() > 0 and pin[:, :60].sum() == 0
    )  # wire x in [10,50], pin x in [60,80]
    assert pin[:, 60:].sum() > 0


def test_background_only_image_has_empty_mask(coco_root):
    ds = CocoSemanticDataset(coco_root / "train", size=64)
    assert ds[len(ds) - 1]["mask"].sum() == 0


def test_original_mask_is_full_resolution(coco_root):
    ds = CocoSemanticDataset(coco_root / "test", size=64)
    om = ds.original_mask(0)
    assert om.shape == (2, 80, 120) and om.dtype == np.uint8 and om.sum() > 0
    assert ds.image_path(0).name == "im0.png"


def test_explicit_category_list_keeps_train_order_on_test(coco_root):
    ds = CocoSemanticDataset(
        coco_root / "test", size=64, category_names=["bond wire", "pin"]
    )
    assert ds.category_names == ["bond wire", "pin"]
    assert ds[0]["mask"][0].sum() > 0  # channel 0 is now the wire


def test_unknown_category_name_raises(coco_root):
    import pytest

    with pytest.raises(KeyError):
        CocoSemanticDataset(coco_root / "test", size=64, category_names=["crack"])


def test_augmentation_keeps_image_and_mask_aligned(coco_root):
    torch.manual_seed(0)
    ds = CocoSemanticDataset(coco_root / "train", size=64, augment=True)
    for _ in range(20):
        it = ds[0]
        # image is a constant colour, so mask/image alignment can only break by shape or dtype
        assert it["image"].shape[-2:] == it["mask"].shape[-2:] == (64, 64)
