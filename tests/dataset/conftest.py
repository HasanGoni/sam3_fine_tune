import numpy as np
import pytest
from PIL import Image

from tests.sambaunet.conftest import coco_root  # noqa: F401  (existing dataset: categories "bond wire"=7, "pin"=3)


@pytest.fixture()
def raw(tmp_path):
    """Flat folders of images + binary masks (matched by file stem); 8 pairs, one image without foreground."""
    imgs, masks = tmp_path / "raw" / "imgs", tmp_path / "raw" / "masks"
    imgs.mkdir(parents=True); masks.mkdir(parents=True)
    for i in range(8):
        w, h = (100, 80) if i % 2 else (90, 90)
        Image.fromarray(np.full((h, w, 3), 20 * i, np.uint8)).save(imgs / f"s{i}.png")
        m = np.zeros((h, w), np.uint8)
        if i != 7:
            m[10:30, 10:50] = 255
        Image.fromarray(m).save(masks / f"s{i}.png")
    return imgs, masks
