import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _rect(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


@pytest.fixture()
def coco_root(tmp_path: Path) -> Path:
    """<root>/{train,test}/{images,annotation.json}; 2 categories, image sizes differ, one image has no annotation."""
    cats = [{"id": 7, "name": "bond wire"}, {"id": 3, "name": "pin"}]
    for split, n in (("train", 4), ("test", 2)):
        (tmp_path / split / "images").mkdir(parents=True)
        images, anns = [], []
        for i in range(n):
            w, h = (120, 80) if i % 2 == 0 else (96, 96)
            Image.fromarray(np.full((h, w, 3), 40 * (i + 1), np.uint8)).save(
                tmp_path / split / "images" / f"im{i}.png"
            )
            images.append({"id": i, "file_name": f"im{i}.png", "width": w, "height": h})
            if i != n - 1:  # last image: background only
                anns.append(
                    {
                        "id": 2 * i,
                        "image_id": i,
                        "category_id": 7,
                        "segmentation": _rect(10, 10, 50, 30),
                        "area": 800,
                        "bbox": [10, 10, 40, 20],
                        "iscrowd": 0,
                    }
                )
                anns.append(
                    {
                        "id": 2 * i + 1,
                        "image_id": i,
                        "category_id": 3,
                        "segmentation": _rect(60, 40, 80, 70),
                        "area": 600,
                        "bbox": [60, 40, 20, 30],
                        "iscrowd": 0,
                    }
                )
        (tmp_path / split / "annotation.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": cats})
        )
    return tmp_path
