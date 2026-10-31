"""Hydra training CLI:  python -m sam3_fine_tune.sambaunet.train data.root=/path/to/dataset [key=value ...]"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import build_model, save_checkpoint
from .data import CocoSemanticDataset
from .metrics import CategoryScores
from .model import bce_dice_loss

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


@torch.no_grad()
def validate(model, loader, device, names, threshold=0.5) -> dict:
    model.eval()
    scores = CategoryScores(names)
    for batch in loader:
        pred = (torch.sigmoid(model(batch["image"].to(device))) > threshold).cpu().numpy()
        gt = batch["mask"].numpy() > 0.5
        for p, g in zip(pred, gt):
            scores.update(p, g)
    return scores.summary()


def fit(cfg: DictConfig, out_dir: Path) -> dict:
    out_dir = Path(out_dir); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.trainer.device)

    root = Path(cfg.data.root)
    train_ds = CocoSemanticDataset(root / cfg.data.train_split, cfg.data.size, cfg.data.augment, ann_file=cfg.data.ann_file)
    val_ds = CocoSemanticDataset(root / cfg.data.val_split, cfg.data.size, False, train_ds.category_names, cfg.data.ann_file)
    kw = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) >= cfg.data.batch_size, **kw)
    val_dl = DataLoader(val_ds, **kw)
    log.info("categories (= text prompts): %s | train %d | val %d", train_ds.category_names, len(train_ds), len(val_ds))

    model = build_model(cfg.model, len(train_ds.category_names)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    log.info("trainable params: %.2fM of %.2fM", sum(p.numel() for p in params) / 1e6, sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.trainer.epochs * len(train_dl)), eta_min=cfg.optim.min_lr)

    start_epoch, best = 0, -1.0
    if cfg.trainer.resume:
        ck = torch.load(cfg.trainer.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        log.info("resumed from %s at epoch %d", cfg.trainer.resume, start_epoch)

    tb = SummaryWriter(str(out_dir / "tb"))
    step = start_epoch * len(train_dl)
    for epoch in range(start_epoch, cfg.trainer.epochs):
        model.train()
        for i, batch in enumerate(train_dl):
            loss = bce_dice_loss(model(batch["image"].to(device)), batch["mask"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is {loss.item()} at epoch {epoch} step {i}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % cfg.trainer.log_every == 0:
                tb.add_scalar("train/loss", loss.item(), step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        res = validate(model, val_dl, device, train_ds.category_names, cfg.trainer.threshold)
        miou = res["macro_foreground_iou"]
        tb.add_scalar("val/macro_foreground_iou", miou, epoch)
        for n, c in res["categories"].items():
            tb.add_scalar(f"val/iou/{n}", c["overall_foreground_iou"], epoch)
        log.info("epoch %d loss %.4f val macro FG IoU %.2f", epoch, loss.item(), miou)
        if miou > best:
            best = miou
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, cfg, train_ds.category_names, epoch=epoch, val=res)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, cfg, train_ds.category_names, epoch=epoch, val=res,
                        opt=opt.state_dict(), sched=sched.state_dict(), best=best)
    tb.close()
    return {"best_mean_foreground_iou": best, "out_dir": str(out_dir)}


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    fit(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
