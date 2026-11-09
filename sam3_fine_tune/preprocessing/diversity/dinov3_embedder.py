"""DINOv3 (ConvNeXt) image embedder used for diversity-based dataset selection.

Uses the locally available DINOv3 ConvNeXt checkpoints at
``/home/hpc_storage/Models/dinov3_models`` via ``timm``. ConvNeXt is used
instead of the DINOv3 ViT variants because its checkpoint naming convention
matches timm's built-in ``convnext`` checkpoint filter cleanly (verified: only
two unused ``norms.3.*`` keys are dropped, all backbone weights load).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_MODEL_NAME = "convnext_tiny"
DEFAULT_CHECKPOINT = (
    "/home/hpc_storage/Models/dinov3_models/"
    "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv3Embedder:
    """Loads a DINOv3 ConvNeXt backbone and extracts L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
        image_size: int = 224,
    ):
        import timm
        from timm.models.convnext import checkpoint_filter_fn

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)  # unsafe-load-ok: external DINOv3 weights, unverified as tensor-only
        filtered = checkpoint_filter_fn(ckpt, self.model)
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys loading {checkpoint_path}: {missing}")

        self.model.eval().to(self.device)
        self.embed_dim = self.model.num_features

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _load_one(self, p: str | Path) -> torch.Tensor:
        img = Image.open(p).convert("RGB")
        return self.transform(img)

    @torch.no_grad()
    def embed_batch(
        self,
        image_paths: list[str | Path],
        batch_size: int = 64,
        num_io_workers: int = 8,
    ) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(image_paths), embed_dim).

        Images are loaded from a network-mounted filesystem, which is the
        dominant cost per batch (not the GPU forward pass). Loading a chunk's
        images concurrently via a thread pool (I/O-bound, so the GIL isn't a
        bottleneck) both speeds up throughput and keeps the GPU from sitting
        idle for long, uninterrupted stretches — this cluster auto-kills GPU
        jobs whose utilization stays below ~5% for more than ~5 minutes.
        """
        all_embeds = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=num_io_workers) as pool:
                tensors = list(pool.map(self._load_one, chunk))
            batch = torch.stack(tensors).to(self.device)
            feats = self.model(batch)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_embeds.append(feats.cpu().numpy())
        if not all_embeds:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return np.concatenate(all_embeds, axis=0).astype(np.float32)
