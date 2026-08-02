---
name: finetune-merged-dataset
description: 'Update the SAM3 training config for a merged dataset (paths, category_chunk_size) and submit a full fine-tuning job on the Acme HPC GPU queue, then validate with evaluation. Use after merge-category-dataset (or with any already-merged dataset) — does not touch dataset merging itself.'
argument-hint: '[--dataset-dir <path>] [--num-categories <n>] [--experiment-name <name>]'
user-invocable: true
disable-model-invocation: false
---
# Finetune Merged Dataset

Update `full_train_config.yaml` for a merged multi-category dataset and
submit a fine-tuning job on the Acme HPC. This is Step 4/5/7 of the
[README](../../../README.md) pipeline. Purely a training step — it assumes
the dataset is already merged (use `merge-category-dataset` first if it
isn't yet).

## When To Use

- A merged dataset (`train/annotation.json` + `test/annotation.json` with
  all categories already combined) exists and is ready to train on.
- You just ran `merge-category-dataset` and want to train on the updated
  combined dataset.
- You want to re-run/resume fine-tuning on an existing merged dataset without
  changing its category set.

## Prerequisites

Run the `configure-repo` skill first to resolve `REPO_ROOT` and `CUDA_MODULE`
in one round-trip.

## Inputs

Ask for these up front:

- `dataset_dir`: root of the merged dataset (`paths.pin_dir_root`) — contains
  `train/` and `test/` with `images/` + `annotation.json`.  The config also
  expects `annotation_merged.json` (it uses `ann_file:
  ${paths.pin_dir_root}/train/annotation_merged.json`); create the file as a
  copy/rename of `annotation.json` if it does not already exist.
- `num_categories`: total category count in the merged dataset. If unknown,
  get it in the same round-trip as validation (step 1 below) rather than a
  separate call.
- `experiment_name`: short name for this training run, used to build
  `experiment_log_dir`. Default to something under the user's own project
  space, not a colleague's, if the dataset is shared.
- `checkpoint_path`: base `sam3.pt` (default, full fine-tune from scratch —
  recommended per README) or a previous fine-tuned `.pt` to resume from. Ask
  the user which, since this materially changes training behavior.
- `train_full_model`: **ask the user** whether to fine-tune the FULL model or
  ONLY the decoder head, and recommend full fine-tuning by default:

  | Component | Decoder-only | Full fine-tune |
  |---|---|---|
  | `backbone.vision_backbone.*` (ViT image encoder) | FROZEN | trained |
  | `backbone.language_backbone.*` (text encoder) | FROZEN | trained |
  | `transformer.encoder.*` (cross-attention encoder) | FROZEN | trained |
  | `transformer.decoder.*` (object query decoder) | trained | trained |
  | `segmentation_head.*` (mask prediction head) | trained | trained |
  | `geometry_encoder.*` | trained | trained |
  | `dot_prod_scoring.*` (classification scores) | trained | trained |

  **Recommendation: full fine-tuning**, per the [README](../../../README.md#option-2-fine-tune-on-a-new-category)
  ("full finetuning has performed better and even converged earlier") and
  this repo's own validated run (job 238099599: full fine-tune from base
  `sam3.pt` took thick_wire/thin_wire/gate_pad AP from 0.19-0.37 to
  0.57-0.90 in 10 epochs). Only suggest decoder-only if the user explicitly
  wants faster/cheaper iteration (fewer trainable params, lower memory) and
  is OK with likely-worse final quality, e.g. quick sanity-checking a new
  category's data before committing to a full run.

  **Do NOT use the separate `sam3/train/configs/decoder_train_config.yaml`
  file for this** — it predates and was never updated with this repo's
  since-validated fixes (`annotation_merged.json` split-photo fix, the
  bf16/GradScaler zero-loss-collapse fix, `PerCategoryEvalWithOverlays`,
  clDice loss option). Using it as-is would silently re-introduce already-fixed
  bugs. Instead, use `full_train_config.yaml`'s own
  `trainer.optim.train_full_model` flag (added 2026-07-20 in `trainer.py`'s
  `Trainer._maybe_freeze_non_decoder_params()`), which freezes the same 3
  components via `requires_grad=False` while inheriting every current fix:
  ```yaml
  trainer:
    optim:
      train_full_model: True   # False = decoder-only (freezes vision/language
                                # backbone + transformer.encoder)
  ```

## Procedure

### 1. Validate the merged dataset (single round-trip)

```bash
ssh hpc_muc '
for split in train test; do
  F="<dataset_dir>/$split/annotation.json"
  python3 -c "
import json
d = json.load(open(\"$F\"))
print(\"$split:\", len(d[\"images\"]), \"images,\", len(d[\"annotations\"]), \"annotations, categories=\", [c[\"name\"] for c in d[\"categories\"]])
"
done
'
```

Use the printed category count as `num_categories`.

Also ensure `annotation_merged.json` exists (the config hard-codes that
filename for both `train` and `test`).  Create it if necessary:

```bash
ssh hpc_muc '
for split in train test; do
  D="<dataset_dir>/$split"
  if [ ! -f "$D/annotation_merged.json" ] && [ -f "$D/annotation.json" ]; then
    cp "$D/annotation.json" "$D/annotation_merged.json"
    echo "created $split/annotation_merged.json"
  fi
done
'
```

### 2. Update the training config

In `sam3/train/configs/full_train_config.yaml` (recommended over decoder-only
per the [README](../../../README.md#overview-full-fine-tuning-vs-decoder-only-fine-tuning)):

- `paths.pin_dir_root` → `<dataset_dir>`.
- `paths.experiment_log_dir` → a new experiment directory for this run (e.g.
  under the user's own project space: `.../experiments/<experiment_name>`).
- `paths.bpe_path` → unchanged unless told otherwise.
- `trainer.model.checkpoint_path` → base `sam3.pt` or a previous fine-tuned
  `.pt`, per the user's answer in Inputs.
- `trainer.optim.train_full_model` → `True` (full fine-tune, default/
  recommended) or `False` (decoder-only), per the user's answer in Inputs.
  Leave `decoder_only_trainable_patterns` at its default unless the user
  wants a non-standard set of trainable components.
- `category_chunk_size` (there are **two** occurrences in the file — train
  and val): **do not blindly set this to `num_categories`.** Read
  `sam3/train/data/coco_json_loaders.py`'s `COCO_FROM_JSON.__init__` first —
  `category_chunk_size` controls how many categories' queries (positive +
  negative) are processed **together, per step, per image**, and GPU memory
  scales with this number, not with total category count:
  ```python
  self.category_chunks = [
      self._sorted_cat_ids[i : i + self.category_chunk_size]
      for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
  ]
  # getDatapointIds():
  return list(range(len(self._raw_data) * len(self.category_chunks)))
  ```
  If `category_chunk_size < num_categories`, categories split into multiple
  chunks and each image is visited once per chunk per epoch — so **epoch
  length grows** by `ceil(num_categories / category_chunk_size)` instead of
  memory growing. This is the lever to use when adding categories to an
  already-memory-tight config:
  - **Best case:** if the new total category count is **≤ the last
    validated memory-safe value**, set `category_chunk_size` **equal to
    `num_categories`** so every training step sees all categories together
    (maximum negative-example pressure, no extra epoch overhead). On
    `gpu_compute` hosts with H200-class memory (`gmem=80G`), you can
    usually do this *and* raise batch size — see the H200-specific config
    in step 3 below.
  - **Safe default:** set `category_chunk_size` to the **last validated
    memory-safe value** for the current `train_batch_size`/`resolution`
    (e.g. `8`, confirmed stable with `train_batch_size=4` on a 44GB card —
    see repo memory `sam3_fine_tune-hpc.md`), NOT to the new total category
    count, if the new total exceeds that safe value.
  - Only scale `category_chunk_size` up to match `num_categories` (for
    maximum negative-example pressure) if you've confirmed there's enough
    GPU memory headroom to absorb it — otherwise you're trading a guaranteed
    OOM for a bit more epoch-time, when you could keep the config working
    and just accept longer epochs instead.
  - When you do raise `category_chunk_size`, lower `val_batch_size` to the
    same value as `train_batch_size` (or lower). Validation uses the same
    per-image category chunking, and the default `val_batch_size: 10` can
    OOM independently of training (observed 2026-07-21: `val_batch_size=10`
    + `category_chunk_size=9` OOM'd during the resume validation epoch on
    44GB cards while training itself had been stable).
  - Concrete lesson (2026-07-17): going from `category_chunk_size=5` (5
    categories) to `category_chunk_size=8` (8 categories, matching the new
    total) OOM'd at `train_batch_size=6` on a 44GB card — the exact same
    batch size that was stable at `category_chunk_size=5`. Had to reduce
    `train_batch_size` to `4` to fit. If instead `category_chunk_size` had
    stayed at a memory-safe value and NOT scaled to match total categories,
    the original `train_batch_size=6` would likely have kept working (at
    the cost of longer epochs once total categories exceed that chunk size).

After editing locally, verify the change actually propagated to the HPC side
of the mount before submitting (network-mount visibility lag is a known
gotcha):

```bash
ssh hpc_muc 'grep -n "category_chunk_size\|experiment_log_dir\|pin_dir_root\|checkpoint_path\|train_full_model" <REPO_ROOT>/sam3/train/configs/full_train_config.yaml'
```

### 3. Submit the training job as a non-blocking batch job

Don't use `-Is` (interactive) for a full training run — it blocks the calling
shell for the entire training duration. Always submit as a batch job with
output/error log files instead, so the ssh call returns immediately.

**Queue strategy — try `gpu_compute` first, fall back to multi-GPU on `gpu`
(not a smaller batch size):**

There are two GPU queues, each with a different resource profile:

- `gpu_compute`: `num=1` (single GPU per job), likely newer/higher-memory
  hardware (e.g. H100-class). Try this **first**.
- `gpu`: supports `num=3` (3 GPUs per job, per the README's `mgpu_` example),
  generally older/lower-memory cards individually, but you get 3 of them.

0. **Pre-flight: check `gpu_compute`'s backlog BEFORE submitting**, in the
   same round-trip as everything else — this is a free, fast signal that
   predicts whether step 2's wait-and-check is even worth doing:

   ```bash
   ssh hpc_muc 'bqueues | grep -E "QUEUE_NAME|^gpu"'
   ```

   **Calibration data (measured 2026-07-16):** with `gpu_compute` `PEND=33`
   at submit time, a job was still `PEND` after 4.5 minutes of waiting (and
   `PEND` count had climbed to 41 by then, not shrunk) — the backlog was
   getting worse, not better. Rule of thumb: if `PEND` on `gpu_compute` is
   already **> ~15-20** at submit time, don't bother with a short
   wait-and-check — either go straight to the `gpu` multi-GPU fallback (step
   3), or submit to `gpu_compute` anyway but tell the user up front to expect
   a multi-minute-to-longer wait rather than polling repeatedly for it.

1. Submit to `gpu_compute` first (includes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, a low-risk fragmentation mitigation to include unconditionally per step 3's OOM fix order below):

   ```bash
   ssh hpc_muc '
   REPO=<REPO_ROOT>
   LOGDIR=$REPO/experiments/<experiment_name>/bsub_logs
   mkdir -p "$LOGDIR"
   bsub -q gpu_compute -gpu "num=1:j_exclusive=yes" -R "osrel=80 && ui==aiml-python" -P ma_ai -n 8 \
     -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
     "bash -c \"source /ifxhome/\$USER/.bashrc && source $REPO/.venv/bin/activate && module load <CUDA_MODULE> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1\""
   '
   ```

2. **Check for some time** whether it actually dispatched — only worth doing
   if step 0's `PEND` count was low. Poll `bjobs <job_id>` after a short wait
   (e.g. ~30-60s, 1-2 checks, combine `sleep N && bjobs <id>` into a single
   ssh round-trip rather than polling repeatedly). `RUN` = good, stay with
   this job. If it's still `PEND` after that check (or step 0 already showed
   a high backlog), move to step 3 — don't keep polling for minutes hoping
   it resolves; that wastes turns without new information once the backlog
   signal is already clear.

3. **Fallback: multi-GPU on the `gpu` queue instead of shrinking batch
   size.** If a job is capacity/memory constrained (OOM) or `gpu_compute`
   won't dispatch in reasonable time, prefer scaling out to 3 GPUs on `gpu`
   over reducing `train_batch_size` — the user has explicitly asked not to
   solve capacity problems by shrinking the batch size:

   ```bash
   # first, bkill the still-PEND gpu_compute job (never bkill a RUN job)
   ssh hpc_muc 'bkill <gpu_compute_job_id>'

   ssh hpc_muc '
   REPO=<REPO_ROOT>
   LOGDIR=$REPO/experiments/<experiment_name>/bsub_logs
   mkdir -p "$LOGDIR"
   bsub -q gpu -gpu "num=3:j_exclusive=yes:gmem=20G" -R "select[ui==aiml_batch_training && osrel=80 && type==X64LIN] rusage[mem=200]" -n 4 \
     -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
     "bash -c \"source /ifxhome/\$USER/.bashrc && source $REPO/.venv/bin/activate && module load <CUDA_MODULE> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_config.yaml --num-gpus 3 --num-nodes 1\""
   '
   ```

   Note `--num-gpus 3` (not `1`) to match the `-gpu "num=3..."` request —
   these must stay in sync, or the job will launch with GPUs it isn't using
   (wasteful) or fewer than the launcher expects (crash).

**Correction from real testing (2026-07-16), important:** "bigger GPU / more
GPUs fixes OOM" is **not reliable** for this repo's config. Two OOM crashes
happened on the same 5-category config (`category_chunk_size=5`,
`resolution=1008`) at *different* card sizes: `train_batch_size=20` OOM'd on
a 22GB card, and even after reducing to `train_batch_size=12` it *still*
OOM'd — this time on a 44GB card. `train_batch_size` in this trainer appears
to be **per-GPU**, so adding more GPUs (`num=3`) does not reduce each GPU's
per-process memory footprint — it likely will not fix an OOM that happens at
`num=1` with the same `train_batch_size`, since each of the 3 GPUs would face
the same per-GPU load.

**Do not change `scratch.resolution`.** The SAM3 ViT uses RoPE positional
embeddings whose frequency tensor (`freqs_cis`) is precomputed for the
configured input size. Changing `resolution` from `1008` to e.g. `896`
triggers an assertion in `sam3/model/vitdet.py`:
```python
assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
```
The model code would need to be modified to recompute `freqs_cis` for a new
resolution. Treat `resolution` as fixed.

**For multi-GPU runs, create a separate config file rather than passing
Hydra-style overrides.** `sam3/train/train.py` uses a fixed `ArgumentParser`
that only knows `--config`, `--num-gpus`, `--num-nodes`, etc. It does **not**
accept Hydra-style CLI overrides such as
`paths.experiment_log_dir=...` or `trainer.data.train.dataset.coco_json_loader.category_chunk_size=8`.
Passing them results in an immediate `unrecognized arguments` error and an
`EXIT` status (e.g. job `240844634`, 2026-07-24). The correct workflow is:
1. Copy `full_train_config.yaml` to a new config (e.g.
   `configs/full_train_<experiment>.yaml`).
2. Edit the copy to set `paths.experiment_log_dir`,
   `paths.pin_dir_root`, both `category_chunk_size` values, and any other
   per-run values.
3. Ensure the launcher block uses `experiment_log_dir: ${paths.experiment_log_dir}`
   so it picks up the path set in `paths`.
4. Submit with `python train.py -c configs/full_train_<experiment>.yaml --num-gpus N --num-nodes 1`.

Given that: if OOM happens at `num=1` regardless of which queue/card you're
on, the actual fix is **`train_batch_size` or `val_batch_size` or
`category_chunk_size`** — not the GPU count. **`gradient_accumulation_steps`
is implemented in this trainer** (`GradientAccumulationWrapper` in
`sam3/train/data/torch_dataset.py` and `_run_step` in `sam3/train/trainer.py`
accumulate gradients across `no_sync()` DDP steps), so setting it to `N` with
`train_batch_size` divided by `N` keeps the effective batch size while lowering
per-step memory. However, it does **not** help validation, because validation
runs with `val_batch_size` and no accumulation. Try a bigger card
(`gpu_compute` vs `gpu`) or more GPUs (`num=3`) only as an initial
troubleshooting step; if that OOMs too, don't keep retrying with more
hardware — tell the user directly and work through the fixes below in order.

**OOM fix order (validated 2026-07-16 / 2026-07-21 on `resolution=1008`
configs):**

1. Add `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the
   `python train.py` call in the bsub command (PyTorch's own suggestion in
   the OOM error message — cheap, no training-behavior impact, do this
   unconditionally).
2. Reduce `train_batch_size`. Validated progression on this config:
   `20` OOM'd on a 22GB card; `12` OOM'd on a 44GB card (bigger card did
   **not** help, confirming this is a real per-batch memory problem, not a
   queue-luck problem); `6` ran stably (job `236913741`, `Mem (GB): 44.00`
   right at the edge of a 44GB card, multiple training steps completed with
   decreasing loss, no crash). Prefer this over resolution changes first,
   since it doesn't affect image quality. If you want to preserve the
   effective batch size, combine this with `gradient_accumulation_steps`
   (e.g. `train_batch_size=2`, `gradient_accumulation_steps=2` gives the
   same effective batch as `train_batch_size=4`).

   **Critical: when using `gradient_accumulation_steps > 1`, the trainer's
   collator asserts `len(batch) == gradient_accumulation_steps` and will
   crash at the end of the first epoch if the dataset size isn't divisible
   by `gradient_accumulation_steps × train_batch_size × num_gpus`.** The
   error looks like:
   ```
   AssertionError: Expected 2 batches, got 1
   ```
   in `sam3/train/trainer.py` `_run_step`. Always check divisibility
   before submitting. Example: a dataset with 10,228 train images is NOT
   divisible by 8 (`2 GPUs × train_batch_size 2 × gradient_accumulation 2`);
   setting `gradient_accumulation_steps: 1` fixes it because
   `10,228 ÷ 4 = 2,557` is exact. For a single-GPU run the divisor is
   `1 × train_batch_size × gradient_accumulation_steps` — the same 10,228
   images is also not divisible by 4 (`1 × 2 × 2`), so it would crash too.
   Quick check:
   ```bash
   python3 -c "n=10228; gpus=2; bs=2; ga=2; print('divisible:', n % (gpus*bs*ga) == 0)"
   ```
   If the remainder is non-zero, either (a) set `gradient_accumulation_steps`
   to 1, (b) adjust `train_batch_size`, or (c) pad/trim the dataset so the
   train split is divisible by the chosen product. Prefer (a) unless you
   specifically need the larger effective batch size.
3. **Also reduce `val_batch_size`.** Validation uses `category_chunk_size`
   categories per sample too, and the default `val_batch_size: 10` can OOM
   independently of training. Drop it to the same value as
   `train_batch_size` (or lower) when raising `category_chunk_size`.
4. **Reduce `category_chunk_size`** back to the last memory-safe value (e.g.
   `8`) if step 2/3 are not enough. This keeps all categories present but
   splits them across multiple chunks per image, trading epoch length for
   memory.
5. **Do not reduce `scratch.resolution`.** See the warning above — it breaks
   the RoPE shape assertion in `vitdet.py` and requires model-code changes.

Confirm dispatch with `bjobs -l <job_id>` (expect `PEND` then `RUN` — `PEND`
waiting for a free GPU is normal, not an error, as long as the chosen queue's
backlog is short).

**Idle-GPU auto-kill policy (learned 2026-07-17):** this HPC cluster
automatically kills a job if its GPU utilization stays below ~5% for more
than ~5 minutes continuously — it shows up as `TERM_OWNER: job killed by
owner.` / exit code `143` in the `.out` job summary, which is easy to
mistake for a manual `bkill`. If a job you expected to run much longer ends
early with that signature, check for a slow setup phase or a stretch of
high-`Data Time` steps (see `hpc-gpu-submit/SKILL.md` for full detail) before
assuming a script bug or someone else killing it.

**Do you actually need `num=3` GPUs, or just a big-enough single GPU?**
`num=3` (multi-GPU/DDP) in step 3's fallback is a **speed** lever (parallel
throughput), not a memory-capacity lever, once you're already pinning
`gmem=40G`. Validated evidence: `train_batch_size=6` on this 5-category
config ran stably on a **single** 44GB card with `Mem (GB): 44.00` (job
`236913741`) — the model + batch fits comfortably on one big-enough GPU.
`train_batch_size` is per-GPU in this trainer, so requesting `num=3` doesn't
reduce each GPU's memory footprint at all; it only lets 3 GPUs each do a
third of the epoch's steps in parallel, cutting wall-clock time roughly 3x
at the cost of reserving 3x the GPU resource (harder to schedule, and 3
GPUs simultaneously idle during any data-loading stall is a bigger hit
against the idle-kill policy above than 1 GPU idling). Default to
`num=1` with `gmem=40G`(+) pinned unless you specifically want the 3x
wall-clock speedup and are fine reserving 3 GPUs — don't reach for `num=3`
reflexively as an OOM fix; it was only ever needed here because earlier
submissions weren't pinning `gmem` high enough to reliably land on a
big-enough single card.

**Multi-GPU scheduling on the `gpu` queue: not all hosts can satisfy
`num=3`.** Hosts in the `gpu` queue have varying GPU counts (e.g.
`muclcgpu01/09/10/11/12/13/14/15/16/17` have 3+ GPUs; many others have only
2). A `num=3:gmem=42G` request can therefore stay `PEND` with reasons like
"Not enough GPUs with the required memory on the hosts" even when the queue
has free slots. If that happens, either lower the request to `num=2` (still
a meaningful speedup) or relax the `gmem` requirement. Validated example
(2026-07-24, 9 categories, `category_chunk_size=8`, `train_batch_size=4`):
`bsub -q gpu -gpu "num=2:j_exclusive=yes:gmem=42G" -n 8 ...` dispatched to
`muclcgpu17` and began training in under a minute (job `240845039`).

**GPU queue host GPU counts (measured 2026-07-24 via `lshosts -gpu`):**
| Host | GPUs |
|---|---|
| muclcgpu01 | 3 |
| muclcgpu02 | 1 |
| muclcgpu03 | 2 |
| muclcgpu04 | 2 |
| muclcgpu05 | 2 |
| muclcgpu06 | 2 |
| muclcgpu07 | 2 |
| muclcgpu08 | 2 |
| muclcgpu09 | 3 |
| muclcgpu10 | 3 |
| muclcgpu11 | 3 |
| muclcgpu12 | 7 |
| muclcgpu13 | 7 |
| muclcgpu14 | 3 |
| muclcgpu15 | 3 (unavail at measurement time) |
| muclcgpu16 | 3 |
| muclcgpu17 | 3 |
Use this to decide `num=` before submitting — don't blindly request 3 GPUs
on a host class that only has 2.

**`gpu_compute` does not appear to support `num>1` per job.** Its queue
limit shows `GPU_REQ: num=1` and `num=3` submissions simply pend with no
matching hosts. Treat `gpu_compute` as single-GPU only. If you want bigger
memory there, request `gmem=80G` (targets H200) and increase
`train_batch_size`/`category_chunk_size` within that single GPU, rather than
requesting more GPUs.

**H200-specific config for `gpu_compute` (optional).** The `gpu_compute`
queue sometimes lands jobs on H200-class GPUs with ~80GB of memory. For the
9-category wire dataset (`9cat_diverse_wires_v1`, `resolution=1008`), this
is enough to keep `category_chunk_size: 9` (all categories processed
together) and raise batch sizes well above the conservative values used for
smaller cards. A validated config is checked in at
`sam3/train/configs/full_train_9cat_diverse_v1_gpucompute_h200.yaml` with:
- `category_chunk_size: 9` (train and val)
- `train_batch_size: 6`
- `val_batch_size: 6`
- `gradient_accumulation_steps: 1`
- `experiment_log_dir: .../experiments/9cat_diverse_wires_v1_gpucompute_h200`

Use it by requesting `gmem=80G` so the scheduler prefers an H200 host:

```bash
ssh hpc_muc '
REPO=<REPO_ROOT>
LOGDIR=$REPO/experiments/9cat_diverse_wires_v1_gpucompute_h200/bsub_logs
mkdir -p "$LOGDIR"
bsub -q gpu_compute -gpu "num=1:j_exclusive=yes:gmem=80G" -R "osrel=80 && ui==aiml-python" -P ma_ai -n 8 \
  -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
  "bash -c \"source /ifxhome/\\\$USER/.bashrc && source $REPO/.venv/bin/activate && module load <CUDA_MODULE> && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_9cat_diverse_v1_gpucompute_h200.yaml --num-gpus 1 --num-nodes 1\""
'
```

**When to use the H200 config vs the standard `gpu_compute` config:**
- Use the H200 config when you are confident the job will land on an
  H200-class host (`gmem=80G` request) and you want the fastest per-step
  throughput and the strongest negative-example pressure from
  `category_chunk_size: 9`.
- Use the standard `gpu_compute` config (`full_train_9cat_diverse_v1_gpucompute.yaml`,
  `train_batch_size: 2`, `val_batch_size: 3`, `category_chunk_size: 9`) when
  you don't care which `gpu_compute` host the job lands on and prefer it to
  fit on the smallest available single-GPU host in that queue.

If an H200 job still OOMs, reduce `train_batch_size` and `val_batch_size`
together (keep them equal) rather than lowering `category_chunk_size`,
since the whole point of the H200 config is to afford the all-categories
chunk.

**If a submitted job stays `PEND` and the user wants to try a different
queue:** only `bkill` it if it's still `PEND` (never a `RUN` job — that would
throw away in-progress training). Killing a `PEND` job is safe since nothing
has started writing checkpoints yet. Resubmit to the other queue using the
same `experiment_log_dir` — don't run both queues' jobs at once against the
same experiment dir, since they'd collide on checkpoint/log writes.

### 4. Validate after training

Run evaluation (`04_single_eval_pipeline.ipynb` / `05_multi_eval_pipeline.ipynb`
or the `.py` equivalents) on **all** categories in the merged dataset, to
confirm any newly-added category learned successfully and there's no
regression on the pre-existing ones.

### 5. Run post-training error analysis

After the training job finishes and the checkpoint is packaged, run the
`post-training-error-analysis` skill to generate:

- A test-set bad-case poster and summary for `thin wire panda`.
- A test-set category overlay poster.
- A training-set label overlay poster.
- Training-set error analysis, once val-on-train predictions are available.

All artifacts are saved under the dataset directory. See
`.github/skills/post-training-error-analysis/SKILL.md` for the exact
command sequence.

## Decision Logic

- **Full fine-tune vs decoder-only:** default/recommend `train_full_model: True`
  (full fine-tune) per the README and this repo's validated evidence. Only use
  `False` (decoder-only) if the user explicitly asks for faster/cheaper
  iteration and accepts likely-lower final quality. Never route decoder-only
  requests to the separate `decoder_train_config.yaml` file — it's stale
  (predates the annotation-merge fix, the bf16/GradScaler fix, and the
  per-category evaluator); use `full_train_config.yaml`'s `train_full_model`
  flag instead so the run inherits every current fix.
- Default queue order: `gpu_compute` (single GPU, try first) → if not
  dispatched within a short check window, `gpu` with `num=3`/`--num-gpus 3`
  (multi-GPU fallback) → only reduce `train_batch_size` as a last resort, and
  say so explicitly before doing it.
- If `checkpoint_path` isn't specified, ask — don't silently assume resume vs.
  from-scratch, since it changes training behavior significantly.
- If `experiment_log_dir` currently points into another user's project space,
  flag it and default to the current user's own space instead, unless they
  explicitly want to reuse the shared location.
- Always submit as a non-interactive batch job (`bsub` with `-o`/`-e` log
  files), never `-Is`, for anything that will run longer than a quick check —
  `-Is` blocks the calling shell for the job's full duration.
- If `num_categories` grew from a previous run and the previous
  `category_chunk_size` already matched the previous total, do NOT
  automatically bump `category_chunk_size` to match the new total — that's
  what caused the 2026-07-17 OOM (see step 2 above). Default to keeping
  `category_chunk_size` at the last validated memory-safe value and accept
  longer epochs, unless the user confirms there's memory headroom to spare.

## Output Format

1. Validated dataset summary (image/annotation counts, category list) from
   step 1.
2. Confirmation of which config fields were changed
   (`pin_dir_root`, `experiment_log_dir`, `checkpoint_path`,
   `category_chunk_size`, `train_full_model`), verified visible on the HPC
   side.
3. Training job command submitted, job ID, and initial `bjobs` status.
4. Post-training evaluation summary across all categories.

## Example prompts

- `Update the config for my merged dataset and start training.`
- `Fine-tune on the dataset I just merged.`
- `Resume training from the last checkpoint on this dataset.`
