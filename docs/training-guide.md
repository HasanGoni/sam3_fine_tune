# Training guide

Reference for editing training configs, launching training (HPC and CPU-only) and debugging training crashes. Moved out
of `CLAUDE.md`, which points here. The HPC steps start at `.github/skills/configure-repo/SKILL.md`.

## Training config gotchas

- **`category_chunk_size`** (two occurrences in the YAML: train and val) is
  a **GPU-memory lever**, not just "number of categories" —
  `sam3/train/data/coco_json_loaders.py`'s `COCO_FROM_JSON` groups this many
  categories' queries together per step; GPU memory scales with this value,
  and setting it below the total category count just makes epochs longer
  (categories processed in multiple chunks), not incorrect. Don't blindly
  set it to the new total category count when adding a category — check
  memory headroom first (see `finetune-merged-dataset` skill for the
  validated OOM history and fix order).
- **`scratch.resolution` is fixed at `1008`.** The ViT's RoPE `freqs_cis` is
  precomputed for this size; changing it trips an assertion in
  `sam3/model/vitdet.py` and needs model-code changes, not a config edit.
- **`gradient_accumulation_steps` requires exact divisibility**:
  `num_gpus × train_batch_size × gradient_accumulation_steps` must evenly
  divide the training set size, or the trainer's collator asserts
  (`Expected N batches, got M`) at the end of the first epoch.
- Configs expect `<pin_dir_root>/{train,test}/annotation_merged.json` —
  create it as a copy of `annotation.json` if a merged dataset doesn't have
  one yet.
- `sam3/train/train.py`'s CLI only accepts `--config`/`--num-gpus`/
  `--num-nodes` — no Hydra-style key=value overrides. For a one-off run, copy
  `full_train_config.yaml` to a new file and edit that instead.

## Training commands

One-command train + test-set eval: `python -m sam3_fine_tune.finetune --data-root ... --sam3-checkpoint ... --bpe-path ...`
(`sam3_fine_tune/finetune/`, see README Step 5). It writes a temporary `_gen_*.yaml` into `sam3/train/configs/` because
`train.py` composes configs by name from that directory.

Run from `sam3/train/`:

```bash
python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1
```

On the Acme HPC (via `hpc-gpu-submit` skill), submitted as a batch job
(not `-Is`) so the SSH call returns immediately:

```bash
bsub -q gpu_compute -gpu "num=1:j_exclusive=yes" -R "osrel=80 && ui==aiml-python" -P ma_ai -n 8 \
  -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
  "bash -c \"source ~/.bashrc && source .venv/bin/activate && module load <cuda_module> && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    cd sam3/train && python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1\""
```

`gpu_compute` (single GPU, higher memory) is preferred over `gpu`
(multi-GPU, `num=3`) — more GPUs don't reduce per-GPU memory (`train_batch_size`
is per-GPU), so OOM fixes are `train_batch_size` / `val_batch_size` /
`category_chunk_size`, not GPU count.

## CPU-only runs (no GPU)

`trainer.accelerator` accepts `auto` (cuda if a GPU is visible, else cpu), `cuda` or `cpu`. The shipped configs say
`cuda`, so on a machine without a GPU they fail immediately with a readable error rather than an NCCL/CUDA stack trace.
To run any config on CPU (debugging only, a step on the 848M-parameter model takes minutes):

```bash
cd sam3/train && CUDA_VISIBLE_DEVICES="" python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1 --accelerator cpu
```

On CPU the trainer switches `distributed.backend: nccl` to `gloo`, autocast follows the real device, the focal loss uses
its pure-PyTorch branch instead of the Triton kernel, and the GPU memory meter reads 0. Set `pin_memory: False` and
small batch sizes / worker counts in the config to keep it usable. Image inference (`build_sam3_image_model(device="cpu")`),
the evaluation helpers and the SAMBA-UNet CLIs (`trainer.device=cpu` / `device=cpu`) also run on CPU.
Prove it with `CUDA_VISIBLE_DEVICES="" pytest tests`. The video predictor/tracker (`sam3/model/sam3_video_*`,
`sam3_tracking_predictor.py`, `io_utils.py` video loaders) is intentionally CUDA-only and is not used by this pipeline.

## Debugging training crashes

`FloatingPointError: Loss is nan`, or loss stuck at exactly `0.00000`:

1. Inspect `<experiment_log_dir>/dumps/anomalous_batches/*.pt`
   (`Trainer._maybe_dump_anomalous_batch`, rate-limited to 3 dumps/run —
   contains both the post-augmentation batch and, if available, the
   pre-augmentation raw source) with
   `.github/scripts/inspect_anomalous_batch.py`.
2. `<experiment_log_dir>/checkpoints/checkpoint_val_pin_detection_coco_eval_segm_AP.pt`
   is the permanently-kept best-validation-AP checkpoint — fall back to it if
   a later epoch collapses.
3. Read `.github/skills/loss-debugging/SKILL.md` first — it documents exactly
   which loss terms in `sam3/train/loss/loss_fns.py` are already NaN-guarded.

Also: this HPC cluster auto-kills jobs whose GPU utilization stays below ~5%
for 5+ minutes (`TERM_OWNER`, exit `143`) — easy to mistake for a manual
kill; check for a slow setup phase or high `Data Time` steps before assuming
a bug.
