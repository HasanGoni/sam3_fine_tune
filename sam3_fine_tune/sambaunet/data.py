"""COCO ``annotation.json`` -> multi-hot semantic masks for SAMBA-UNet.

Layout (same as the SAM3 pipeline): ``<split_dir>/images/*`` + ``<split_dir>/annotation.json``.
Channel k of the mask is category k; ``category_name`` is the same string used as the SAM3 text prompt.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoSemanticDataset(Dataset):
    def __init__(
        self,
        split_dir,
        size: int = 256,
        augment: bool = False,
        category_names: list[str] | None = None,
        ann_file: str = "annotation.json",
    ):
        self.dir, self.size, self.augment = Path(split_dir), size, augment
        self.coco = COCO(str(self.dir / ann_file))
        by_name = {c["name"]: c["id"] for c in self.coco.dataset["categories"]}
        if category_names is None:  # default: sorted by COCO category id, deterministic
            category_names = [
                c["name"]
                for c in sorted(self.coco.dataset["categories"], key=lambda c: c["id"])
            ]
        self.category_names = list(category_names)
        self.cat_ids = [
            by_name[n] for n in self.category_names
        ]  # KeyError on unknown names (on purpose)
        self.img_ids = sorted(self.coco.imgs)

    def __len__(self) -> int:
        return len(self.img_ids)

    def image_path(self, idx: int) -> Path:
        return self.dir / "images" / self.coco.imgs[self.img_ids[idx]]["file_name"]

    def original_mask(self, idx: int) -> np.ndarray:
        """(K,H,W) uint8 at the image's native resolution (used for evaluation)."""
        info = self.coco.imgs[self.img_ids[idx]]
        out = np.zeros((len(self.cat_ids), info["height"], info["width"]), np.uint8)
        for k, cid in enumerate(self.cat_ids):
            for ann in self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=info["id"], catIds=[cid])
            ):
                out[k] |= self.coco.annToMask(ann)
        return out

    def __getitem__(self, idx: int) -> dict:
        img = np.asarray(Image.open(self.image_path(idx)).convert("RGB"))
        image = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        mask = torch.from_numpy(self.original_mask(idx)).float().unsqueeze(0)
        image = F.interpolate(
            image, (self.size, self.size), mode="bilinear", align_corners=False
        )[0]
        mask = F.interpolate(mask, (self.size, self.size), mode="nearest")[0]
        if self.augment:
            image, mask = _paired_augment(image, mask)
        return {"image": image.clamp(0, 1), "mask": mask, "index": idx}


def _paired_augment(image: torch.Tensor, mask: torch.Tensor):
    """Random flips + 90-degree rotations applied identically to image and mask."""
    if random.random() < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)
    if random.random() < 0.5:
        image, mask = image.flip(-2), mask.flip(-2)
    k = random.randint(0, 3)
    return torch.rot90(image, k, (-2, -1)), torch.rot90(mask, k, (-2, -1))
