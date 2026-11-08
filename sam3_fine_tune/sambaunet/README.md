# SAMBA-UNet (arXiv 2610.40029) for side-by-side comparison with the SAM3 fine-tune

SAM ViT-B (frozen + LoRA/Adapters) guiding a Mamba U-Net through HCEM. No official code exists; this is our port of
`quarto_blog_hasan/notebooks/paper-sambaunet/sambaunet.py`. Paper-silent choices are marked `# ASSUMPTION` in `model.py`.

Differs from SAM3: **no text prompts, no instances**. One sigmoid channel per COCO category (the category name is the
SAM3 prompt), input 256x256, semantic masks only.

```bash
uv pip install -e ".[sambaunet]"          # adds transformers

# train (any config key can be overridden; see conf/config.yaml)
python -m sam3_fine_tune.sambaunet.train data.root=/path/to/dataset trainer.epochs=100 data.batch_size=8 model.grad_checkpoint=true

# evaluate -> multiclass_foreground_iou_summary.json (same layout as the SAM3 multi-class evaluation)
python -m sam3_fine_tune.sambaunet.evaluate checkpoint=experiments/sambaunet/<run>/checkpoints/best.pt data.root=/path/to/dataset save_overlays=true

pytest tests/sambaunet                    # offline, tiny random SAM encoder
```

Dataset layout is the standard one: `<root>/{train,test}/{images,annotation.json}`.

## Cost (measured once on a shared GB10, 256px, 2 categories, real SAM ViT-B weights)

| setting | s / image (fwd+bwd) | peak GPU memory |
|---|---|---|
| parallel scan | 1.07 | 25 GB at batch 2 |
| parallel scan + `model.grad_checkpoint=true` | 1.4 | 5.6 GB at batch 2, 10.4 GB at batch 4 |

The selective scan is pure PyTorch (log-depth `associative_scan`); the paper's CUDA kernel (`mamba_ssm`) would be much faster.

## Comparing with SAM3
Both evaluations write `multiclass_foreground_iou_summary.json`; compare `summary.categories.<name>.overall_foreground_iou`,
`pixel_precision`, `pixel_recall` and `summary.overall`. Use the same `test/` split. Caveat: both trainers pick the best
checkpoint on that split, so neither number is a clean held-out estimate.

Worked example (Python API + Hydra CLI, runs on CPU with synthetic data): `nbs/14_sambaunet_api_and_cli.ipynb`.
