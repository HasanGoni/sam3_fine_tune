# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fine-tunes Meta's SAM3 (Segment Anything Model 3 — ViT vision backbone + text
encoder + cross-attention transformer decoder) for open-vocabulary
segmentation of Acme industrial defect/component categories (bond wires,
pins, cracks, chips) driven by text prompts. Full pipeline: mask→COCO
conversion → dataset merge → training → inference → evaluation →
visualization/error analysis.

## Environment: where things run

This directory is the git clone: `git`, `gh` and pushes to `origin` (GitHub, `HasanGoni/sam3_fine_tune`) work here.
Training, preprocessing and evaluation on the real data run on the Acme HPC (GPU/CPU `bsub` queues), reached via
`ssh hpc_muc`. That host does not resolve from every machine (it failed here with "Could not resolve hostname"), so run
`.github/skills/ssh-hpc-connect/SKILL.md` first. When the HPC is unreachable, work with the local copies under "Local
weights and data" below. HPC steps start at `.github/skills/configure-repo/SKILL.md`; there, check that a change has
propagated to the HPC side of the network mount before running anything (it can lag).

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Requires Python 3.12, PyTorch 2.7, CUDA 12.6. `uv` pulls from a private
Acme Artifactory index (configured in `pyproject.toml`).

## Repository layout

- **`sam3/`** — vendored/forked copy of Meta's SAM3 model + training
  framework (`model/`, `train/`, `eval/`, `sam/`, `agent/`, `perflib/`). This
  is the code that actually runs: training entrypoint is
  `sam3/train/train.py`, driven by `sam3/train/trainer.py` and configs under
  `sam3/train/configs/`.
- **`sam3_fine_tune/`** — this project's own package layered on top:
  - `preprocessing/` — `coco_format.py` (masks → COCO), dataset merge/append
    helpers backing `nbs/02_merge_dataset.ipynb`, patch/tiling dataset
    builder, DINOv3 diversity-based rebalancing (`diversity/`).
  - `evaluation/` — single/multi-class eval pipelines.
  - `inference/` — decoder inference helper.
  - `config/` — copies of the training YAMLs (kept in sync with
    `sam3/train/configs/`).
  - `modified_script/` — **reference-only** snapshots of hand-patched
    `sam3/` files (`train.py`, `loss_fns.py`, `matcher.py`, `box_ops.py`,
    `train_utils.py`, `freeze_utils.py`), kept for diffing against upstream
    history. Nothing here is imported or executed — the live versions are
    under `sam3/train/`.
- **`nbs/`** — nbdev notebooks, numbered as pipeline stages (`01_coco_format`
  … `13_dinov3_embedding_dedup`). Some are the source of truth for logic that
  helper scripts extract directly (see next point) rather than import.
  `14_sambaunet_api_and_cli` is documentation only (SAMBA-UNet Python API +
  Hydra CLI on synthetic data); it exports nothing.
- **`.github/skills/`** + **`.github/scripts/`** — the HPC runbooks and the
  helper scripts they invoke (single-round-trip SSH validators/submitters).
- **`scripts/`** — standalone analysis/poster scripts, several specific to
  the `thin_wire_panda` category, plus `run_post_training_analysis.sh`
  orchestrating the standard post-training artifact set.
- **`experiments/`** — per-run `experiment_log_dir`s: checkpoints,
  tensorboard logs, and (on a crash) `dumps/anomalous_batches/`.
- **`scratch/`** — convention for ad-hoc/one-off exploration scripts; keep
  the repo root clean.

**nbdev quirk:** `settings.ini` (nbdev config) was never migrated to
`pyproject.toml`, so the real `nbdev_export` CLI does not work in this repo.
Where a notebook's `#| export` cells need to run outside the notebook,
helper scripts parse the notebook JSON directly and call the exported
functions in-memory (e.g. `.github/scripts/run_append_dataset.sh` reads
`nbs/02_merge_dataset.ipynb`) instead of relying on a generated module.

## Full fine-tuning vs. decoder-only

Controlled by `trainer.optim.train_full_model` in
`sam3/train/configs/full_train_config.yaml` (`Trainer._maybe_freeze_non_decoder_params`
in `sam3/train/trainer.py`) — **do not use the separate
`decoder_train_config.yaml`**, it predates fixes that only live in
`full_train_config.yaml`.

| Component | Decoder-only | Full fine-tune |
|---|---|---|
| `backbone.vision_backbone.*` (ViT) | frozen | trained |
| `backbone.language_backbone.*` (text encoder) | frozen | trained |
| `transformer.encoder.*` (cross-attention) | frozen | trained |
| `transformer.decoder.*`, `segmentation_head.*`, `geometry_encoder.*`, `dot_prod_scoring.*` | trained | trained |

Full fine-tuning is recommended (per README benchmarks) and converges
earlier.

## Building / extending a dataset

`sam3-dataset images=... masks=... category=... into=...` (`sam3_fine_tune/dataset/`, Hydra, console script registered in
`pyproject.toml`; `uv pip install -e . --no-deps` or `uv sync` installs the script) converts masks to COCO,
splits train/test and creates / extends a category locally. See README "Shortcut: build or extend a dataset".

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

## Local weights and data

`~/workspace/projects/Models/models--facebook--sam3/` holds `sam3.pt` and `bpe_simple_vocab_16e6.txt.gz` (the tokenizer is
not in this repo). Check there before downloading anything. `data/` is gitignored scratch for downloads. Real-weights
test: `SAM3_CHECKPOINT=<sam3.pt> SAM3_BPE=<bpe.gz> .venv/bin/python -m pytest tests/inference/test_real_weights.py -q`.

`data/sam3.pt` is a second copy of the weights. The vendored `build_sam3_image_model` defaults `bpe_path` to
`assets/bpe_simple_vocab_16e6.txt.gz`, which does not exist: always pass `bpe_path` (it fails with `FileNotFoundError`
otherwise).

Local datasets live under `~/workspace/projects/data/sam3_data/` (for example the 12-category
`10cat_..._wholechips_v1`, 6,091 test images). They are not the HPC's `9cat_separated_wires_v3`, and part of the images
are absent on disk (2,253 of 12,435 train images in that set), so check that image files exist before using a split.

## Formatting / lint / tests

Plain `python` is not on PATH here; use `.venv/bin/python`.

```bash
.venv/bin/python -m pytest tests                      # full suite (~65 s, CPU only)
.venv/bin/python -m pytest tests/test_repo_standards.py   # mechanical repo rules, instant
uvx pre-commit run --all-files      # mypy on the sam3-infer files + ufmt on exactly the files CI formats (skips the AUTOGENERATED decoder_inference.py)
```

Format and check the files you changed with `uvx pre-commit run --files <files>`. Never call `ufmt` directly: it needs the
hook's pinned dependencies (`ruff-api` …) and crashes without them. `.pre-commit-config.yaml` holds the exclusions.

`.github/workflows/standards.yml` and `.pre-commit-config.yaml` (`uvx pre-commit install`, once) run the repo rules and
`ufmt` on the hand-written code: `tests/`, `sam3_fine_tune/{dataset,finetune,sambaunet}` and the `sam3-infer` files in
`sam3_fine_tune/inference/`. Vendored `sam3/`, `scripts/` and the nbdev-generated modules are not formatted
(`workflows/format.yml` at the repo root is outside `.github/` and never runs). Line length is 88.

After implementing, run `/code-review` for correctness bugs; the standards/spec review does not hunt them.
Review rules that need judgement live in `CODING_STANDARDS.md`; domain vocabulary in `GLOSSARY.md`.

## Adding a Hydra CLI

Copy the layout of `sam3_fine_tune/dataset/`: `cli.py` (`@hydra.main(config_path="conf", ...)`), `__main__.py`,
`conf/<name>.yaml` **plus an empty `conf/__init__.py`**, a `[project.scripts]` entry and a
`[tool.setuptools.package-data]` entry in `pyproject.toml`, and a `tests/<name>/` folder. `tests/test_repo_standards.py`
fails if the `__init__.py` or package-data entry is missing.

## Data layout convention

```
dataset/
├── train/
│   ├── images/
│   └── annotation.json        # COCO polygons, produced from images/+masks/
└── test/
    ├── images/
    └── annotation.json
```

`category_name` in the COCO annotation is the text prompt used at both train
and inference time — it must match exactly.

## Agent skills

### Pull requests

When writing a PR body, use the `/pr` skill.

### Issue tracker

GitHub Issues on `HasanGoni/sam3_fine_tune`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
