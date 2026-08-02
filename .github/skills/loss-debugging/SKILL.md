---
name: loss-debugging
description: 'Debug NaN/Inf training loss crashes (FloatingPointError: Loss is nan) in the SAM3 fine-tuning trainer. Documents which loss terms in sam3/train/loss/loss_fns.py are already NaN-guarded, which are not, exact file/line locations, the instrumentation choke point, and known false leads (benign pymp cleanup errors). Use this FIRST before re-tracing the loss stack from scratch whenever training crashes with a non-finite loss.'
argument-hint: '[--job-log-path <path/to/bsub_logs/train.<jobid>.out>]'
user-invocable: true
disable-model-invocation: false
---
# Loss Debugging (NaN / Inf training loss)

Debug `FloatingPointError: Loss is nan, attempting to stop training` (or Inf)
crashes during SAM3 fine-tuning. This skill exists so a future investigation
does not have to re-read and re-trace the entire loss stack
(`sam3/train/loss/loss_fns.py`, `sam3_loss.py`, `box_ops.py`, `trainer.py`)
from scratch — it documents exactly what's already fixed, what's already
confirmed correct, and where the remaining gaps are.

## When To Use

- A training job dies with `FloatingPointError: Loss is nan` (or inf) in its
  `bsub_logs/train.<jobid>.err` log.
- You want to know whether a given loss term already has NaN protection
  before touching it again.
- You're about to grep/read the loss code from scratch — check the "Already
  checked" list below first; it's usually faster to trust it and move on.

## Step 0 — Rule out the pymp red herring first

`.err` logs for this trainer commonly contain a huge wall of

```
OSError: [Errno 16] Device or resource busy: '.../pymp-XXXXXXXX'
```

tracebacks from `multiprocessing/util.py::_remove_temp_dir`. **This is
benign** — DataLoader worker temp-dir cleanup noise at process teardown /
rank exit, unrelated to any NaN. Do not spend time on it. Instead grep
specifically for `FloatingPointError` or `Loss is nan` — that's the real
signal, always near the very end of the `.err` file.

When grepping a large `bsub_logs/*.out` file (these can be tens of MB with
one line per logged step), use `grep_search` with **narrow, specific
literal strings** (e.g. `"Loss is nan"`, `"Estimated time remaining"`) —
broad regex/literal queries across the whole file can time out.

## Step 1 — Sanity-check it isn't generic training instability

Read the `.out` log's `Losses/train_all_loss` progression from the start of
training. If it's healthy and decreasing right up to the crash (as opposed
to trending upward / erratic before the crash), that points to an isolated
bad sample/batch rather than optimizer divergence — go to Step 2.

Also check `sam3/train/configs/full_train_config.yaml` for:
- `optim.amp.amp_dtype` — if `bfloat16` (not `float16`), fp16-range overflow
  to Inf is ruled out (bf16 has the same exponent range as fp32).
- `optim.gradient_clipping.max_norm` — if set (e.g. `0.1`), exploding
  gradients are already mitigated.

As of 2026-07-17 for `experiments/5cat_matrix_chip_v2` (job `237165894`):
loss was healthy/decreasing (572 → ~120 in epoch 0, ~85 by epoch 1), crashed
~3h20m later. `amp_dtype: bfloat16`, `max_norm: 0.1` — both already in place.
This pointed to an isolated bad sample, not generic instability.

## Step 2 — What's already fixed (don't re-fix, just confirm still present)

Fixed on **2026-07-16** in `sam3/train/loss/loss_fns.py` (search for the
`(Sy) Added` / `(Sy) Changed` comments to confirm):
- `dice_loss()` / `_dice_loss()` — `torch.nan_to_num(loss, nan=0.0,
  posinf=10.0, neginf=0.0)` + `max(num_boxes, 1)` zero-division guard.
- `sigmoid_focal_loss()` — same `nan_to_num` guard, applied to **both** the
  triton fast-path (`reduce=True, loss_on_multimask=False` — the path
  `Masks.get_loss()` actually uses) and the general path, plus the same
  `max(num_boxes, 1)` guard.

Fixed **earlier** (before 2026-07-16) in `sam3/model/box_ops.py` (also
`(Sy)`-tagged): `box_iou()`, `generalized_box_iou()`, `fast_diag_box_iou()`,
`fast_diag_generalized_box_iou()` all add `+ 1e-8` to every division
denominator — GIoU/IoU math cannot produce a 0/0 NaN from a degenerate
(zero-area) box.

## Step 3 — What's already confirmed correct/guarded (verified 2026-07-17, don't re-derive)

All via the guards in Step 2, given this repo's active `weight_dict`
(`sam3/train/configs/full_train_config.yaml`, ~L85-118):
- `Masks.get_loss` → `loss_mask`, `loss_dice` — guarded (uses
  `sigmoid_focal_loss` + `dice_loss`).
- `IABCEMdetr`'s presence-loss branch → `presence_loss` — guarded (uses
  `sigmoid_focal_loss`).
- `SemanticSegCriterion.get_loss` → `loss_semantic_seg`, `loss_semantic_dice`
  — guarded (`focal: True` in this config routes through `sigmoid_focal_loss`
  + `dice_loss`). `loss_semantic_presence` is a dummy `0.0` tensor in this
  config since `presence_loss: False` is set for `loss_fn_semantic_seg`, so
  it isn't actually computed at all right now.

## Step 4 — Known remaining gap (as of 2026-07-17, check if since patched)

Two loss terms in `sam3/train/loss/loss_fns.py` have **no** `nan_to_num`
guard and are NOT covered by anything in Step 2/3:

1. **`IABCEMdetr.get_loss`** (~L361-530) — `loss_ce`, weight `20.0` (tied
   for the single largest weight in the whole `loss_fns_find` list). Two raw
   `F.binary_cross_entropy_with_logits` calls:
   - ~L390-401, "Soft loss on positives" branch.
   - ~L411-413, "Negatives" branch (added to the positives branch).
   No `nan_to_num` guard is applied to the combined `loss_bce` anywhere
   before it's reduced (`.mean()` / `.sum() / (...)`).
2. **`Boxes.get_loss`** (~L547-579) — `loss_bbox` (weight `5.0`, raw
   `F.l1_loss`) and `loss_giou` (weight `2.0`, via
   `fast_diag_generalized_box_iou`, which is itself epsilon-safe per Step 2
   — but the final `loss_giou = 1 - fast_diag_generalized_box_iou(...)`
   result still isn't `nan_to_num`-guarded before `.sum() / num_boxes`).

If investigating a fresh NaN and Step 5's instrumentation points at
`loss_ce` or `loss_bbox`/`loss_giou`, the fix is the same one-line pattern
used in Step 2: wrap the loss tensor in
`torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=0.0)` right before its
final reduction, matching the existing `(Sy) Added`/`(Sy) Changed` style.

## Step 5 — Instrumentation (permanent, added 2026-07-17)

`Sam3LossWrapper.compute_loss()` in `sam3/train/loss/sam3_loss.py`
(~L83-155) is the single choke point where every registered loss_fn's
returned dict has `CORE_LOSS_KEY` popped and summed into `total_core_loss`
(both the main `o2o` call and the `o2m` variant). This function logs a
non-finite check per returned loss key (loss_fn class name + key + `suffix`
+ `is_aux`) BEFORE it reaches `trainer.py`'s top-level
`math.isfinite(loss.item())` crash check — so any future NaN should already
be named in the log instead of showing up as a bare
`FloatingPointError: Loss is nan` with no further detail. Check this log
output first before adding new debugging code.

This instrumentation is intentionally kept permanent (not removed after the
investigation) per explicit user decision — do not strip it out during
cleanup.

## Step 6 — Crash point reference

`sam3/train/trainer.py::_run_step` (~L950-963): `math.isfinite(loss.item())`
raises `FloatingPointError`. `train_epoch` (~L795-880) always calls
`_run_step` with the default `raise_on_error=True`, and its
`except FloatingPointError as e: raise e` just re-raises — there is
deliberately **no** skip-bad-batch fallback (explicit user decision: keep
hard-crash-on-NaN, since the guards above already absorb the "expected"
math edge cases — a crash surviving them likely signals something that
shouldn't be silently skipped).

## Step 7 — Likely data-side root cause to check first

If the instrumentation (Step 5) points at `loss_ce`/`loss_bbox`/`loss_giou`
and the loss curve was healthy before the crash (Step 1), suspect a single
degenerate sample rather than a systemic bug. First check whether the crash
correlates with the most-recently-appended data. As of 2026-07-17 the most
recent addition was `matrix_chip` patches added 2026-07-16 via
`append_images_to_existing_category()` — repo memory
(`/memories/repo/sam3_fine_tune-hpc.md`) already documents a known
image/mask filename-stem mismatch gotcha (`_ch0`/`_mask` suffixes) for this
patch-style dataset's `coco_format.py` conversion path, which could produce
an empty/misaligned mask → degenerate (near-zero-area) box for one sample.

A CPU-only audit script (no GPU/training needed) can check the merged
dataset for: `masks_to_boxes` producing zero-area boxes, fully-empty masks,
mismatched image/mask filename stems, and NaN/Inf pixel values — much
cheaper than re-running a multi-hour GPU job to find the same thing. Use
`.github/scripts/check_dataset_degenerate_samples.py <annotation.json>
[--images-dir <dir>] [--check-pixels]` (added 2026-07-17), submitted via the
`hpc-cpu-submit` skill rather than run inline in the login shell.

## Step 8 — Anomalous-batch dumps and the exactly-zero-loss failure mode (added 2026-07-17)

A *different* failure mode from the NaN crash above: train loss can go to
**exactly `0.00000`** for many consecutive steps (or a whole epoch) without
ever crashing (`0.0` is finite, so `math.isfinite()` never trips). This was
observed on `experiments/5cat_matrix_chip_v2` starting at epoch 9 of job
`237436875`, and silently overwrote the last-good `checkpoint.pt` with a
collapsed/dead model — see repo memory
(`/memories/repo/sam3_fine_tune-hpc.md`) for the full timeline. A repro
attempt from the last known-good checkpoint did **not** reproduce the
collapse through the same epoch, and a full CPU-only data audit (Step 7's
script, `--check-pixels`, both splits, all 5 categories old and new) found
zero issues — pointing away from a deterministic data/code bug and toward a
rare/hardware-driven event. Two pieces of tooling now exist for the next
occurrence:

1. **`<experiment_log_dir>/dumps/anomalous_batches/`** — populated
   automatically by `Trainer._maybe_dump_anomalous_batch()`
   (`sam3/train/trainer.py`), triggered whenever the real (non-`DummyLoss`)
   training loss goes non-finite **or exactly zero**. Rate-limited to 3
   dumps per run (a persistent collapse could otherwise write hundreds of
   large files). Each `<phase>_<key>_epoch<E>_step<S>_<n>.pt` file contains
   **both**:
   - post-augmentation `img_batch` + `find_targets` (boxes/num_boxes) — what
     the model actually saw that step, and
   - pre-augmentation `raw_images` + `raw_boxes` (new field on `Datapoint` in
     `sam3/train/data/sam3_image_dataset.py`, mirroring the pre-existing
     `raw_images` mechanism that was already threaded through the collator
     but never used for debugging) — the exact source data *before* any
     transform (resize/crop/pad/box-noise) touched it.

   Inspect a dump with `.github/scripts/inspect_anomalous_batch.py
   <dump_path>` — it prints a side-by-side pre- vs. post-augmentation
   box/image comparison, which directly answers "did the augmentation
   pipeline introduce a degenerate box, or was it already bad in the
   source?" without re-deriving anything.

2. **`sam3/train/loss/sam3_loss.py`'s `compute_loss()`** logs a
   `"(Ha) Zero total_core_loss diagnostic"` warning (grep for this string in
   the `.err` log) any time `total_core_loss == 0.0`, with `num_boxes`,
   `targets_num_boxes_sum`, and matched-pair counts per stage — lets you
   tell apart an expected all-negative batch (`num_boxes == 0`, benign given
   `BinaryHungarianMatcherV2`'s `remove_samples_with_0_gt=True` default)
   from a genuine 0-matched-pairs-despite-real-targets bug.

Note this instrumentation only takes effect on **new** job submissions —
Python doesn't hot-reload, so a job already running when this code was
added won't have it.

## Step 9 — ROOT CAUSE CONFIRMED (2026-07-18): the zero-loss collapse IS a NaN, just silently absorbed

**This resolves Step 8's "rare/hardware-driven event" framing — it is
neither rare-and-unexplainable nor hardware-driven. It is the exact same
underlying NaN-producing instability from Step 0-7 above, just no longer
visible, because the very `nan_to_num` guards added on 2026-07-16/17 to stop
the `FloatingPointError` crash also strip out the evidence.**

Confirmed on `experiments/8cat_bondwire` (job `237913353`, a recurrence of
the exact scenario Step 8 describes) by adding a `model_output_debug` capture
to `Sam3LossWrapper.compute_loss()` (`sam3_loss.py`) that stashes the main
stage's `pred_boxes`/`pred_logits`/`pred_masks` AT THE MATCHED INDICES
whenever the zero-loss diagnostic fires, then inspecting the resulting dump
(`train_all_epoch5_step9768_0.pt`):

```
pred_boxes[0,6]  = [nan, nan, nan, nan]
pred_logits[0,6] = [nan]
pred_masks[0,6]  min/max/mean = nan, nan, nan
```

**The matched predictions are literally NaN — not zero, and not a
padding/matching-indices bug** (the originally-suspected alternate theory,
now ruled out). The mechanism:

1. The model's forward pass produces NaN at some query position(s) for
   some batch (root numerical-instability cause still unidentified — see
   "Next step" below).
2. Every loss function that consumes that query's prediction
   (`Boxes`/`IABCEMdetr`/`Masks`, both `o2o` and `o2m`) computes a NaN loss
   value from it.
3. The `nan_to_num(nan=0.0, posinf=10.0, neginf=0.0)` guards added in Step 2
   (2026-07-16, specifically to stop `FloatingPointError` from crashing
   training) convert that NaN loss to exactly `0.0`.
4. Since **every** individual loss term across all ~6 stages x 3 loss_fns x
   o2o/o2m is independently `0.0` (confirmed via the per-term breakdown
   added the same day — see `nonzero_contributions=NONE` in the diagnostic
   log), the summed `total_core_loss` is also exactly `0.0` — silently, with
   `math.isfinite()` never tripping, so training continues with a dead
   (zero-gradient) step instead of crashing.

**In short: the 2026-07-16 crash fix didn't fix the underlying bug — it hid
it.** The model still produces NaN under the same conditions; it just no
longer tells you when.

### New tooling for this (2026-07-18)

1. **`Sam3LossWrapper.last_zero_loss_debug`** (`sam3_loss.py`) — captures
   `matched_pairs`, `matched_pred_boxes`, `matched_pred_logits`,
   `matched_pred_masks`, `target_boxes` (sliced to ONLY the matched
   `(batch, query)` positions, moved to CPU immediately at the capture site)
   whenever the zero-loss diagnostic fires. Picked up by
   `Trainer._maybe_dump_anomalous_batch()` into the dump payload as
   `model_output_debug`. **Important perf/safety note:** the *first*
   version of this capture kept the FULL `pred_boxes`/`pred_logits`/
   `pred_masks` tensors (all ~200 queries, not just matched ones)
   GPU-resident via `.detach()` alone (no `.cpu()`), which caused a
   **real OOM crash** (job `237913353`, step 730) on a config already at
   ~44/44.39GB with zero headroom — `pred_masks` in particular is
   `num_queries x H x W`, large. Fixed by slicing to matched indices only
   and moving to CPU immediately, not after returning to `trainer.py`. If
   you ever touch this capture again, do NOT revert to capturing full-size
   tensors without slicing first.
2. **`.github/scripts/inspect_anomalous_batch.py`**'s
   `_describe_model_output_debug()` prints the matched prediction values
   directly — this is what confirmed the NaN.
3. **`_install_nan_debug_hooks()`** (`trainer.py`, gated behind
   `SAM3_NAN_DEBUG_HOOKS=1` env var — off by default, real per-step
   overhead) — registers a forward hook on every named submodule that
   checks its output for non-finite values, logging the FIRST (in forward
   call order) submodule to produce NaN/Inf. This is the tool to use NEXT:
   `model_output_debug` can only see the NaN after it already reached the
   final prediction heads; these hooks bisect the forward pass to find
   where it actually originates (attention, a specific norm layer, the
   image encoder, etc.).

### Next step (DONE — see Step 10 below)

Ran a job with `SAM3_NAN_DEBUG_HOOKS=1` set and waited for the collapse to
recur (non-deterministic — has occurred at steps 630, 670, 730, 900, and
9766 across different resumes from the same epoch-5 checkpoint, and not at
all in between). This bisected the forward/optimizer pass down to the exact
mechanism — see Step 10.

**Also documented alongside this investigation (not a loss-code fix, but
directly relevant when a training job appears to die/end unexpectedly):**
this HPC cluster auto-kills a job if GPU utilization stays below ~5% for
more than ~5 minutes, showing as `TERM_OWNER: job killed by owner.` / exit
code `143` — easy to mistake for a manual `bkill`. See
`.github/skills/hpc-gpu-submit/SKILL.md`'s Decision Logic section for known
triggers and how to tell it apart from an actual crash.

## Step 10 — ROOT CAUSE + FIX CONFIRMED (2026-07-18): GradScaler decays into fp32 subnormal territory for bfloat16

**This is the actual origin of the NaN from Step 9 — not the loss guards, not
the matcher, not the data pipeline.** Added
`_check_patch_embed_weight_nan()` (`trainer.py`), which checks
`backbone.vision_backbone.trunk.patch_embed.proj`'s weight/bias for
non-finite values immediately after every `scaler.step()` + `scaler.update()`,
logging the step number and `scaler.get_scale()` the first time it fires.
Job `237937642` (resumed from the same epoch-5 checkpoint, `SAM3_NAN_DEBUG_HOOKS=1`)
hit it:

```
WEIGHT-NaN: patch_embed.proj.weight went non-finite at step=9766,
scaler.get_scale()=2.938735877055719e-39, n_nan=602112, n_inf=0, param_numel=602112
```

`n_nan == param_numel` — **100% of the Conv2d weight's elements went NaN in
one step**, at the exact moment `scaler.get_scale()` had decayed to
`2.94e-39`, a **subnormal fp32 value** (~143 consecutive `backoff_factor=0.5x`
halvings from the default `init_scale=65536.0`).

**Root cause:** `torch.amp.GradScaler` is designed for `float16`'s narrow
~5-bit exponent range, where dynamic loss scaling is genuinely necessary to
prevent gradient underflow. This trainer uses `bfloat16`
(`optim.amp.amp_dtype: bfloat16`), which has the **same exponent range as
fp32** and never needed loss scaling at all — but `GradScaler` doesn't know
that, so it still dynamically halves the scale on any inf/nan
(`backoff_factor=0.5`). Given enough halvings, the scale value *itself*
spirals down into subnormal fp32 territory, at which point the
`scale`/`unscale_` round-trip (`loss * scale`, later `grad / scale`) becomes
numerically unstable in its own right — this instability, not a genuinely
bad gradient, is what corrupted `patch_embed.proj`'s weights.

### The fix

Added `optim.amp.bypass_dynamic_scale_for_bf16` config flag (default `True`)
in `sam3/train/configs/full_train_config.yaml`, matched by a new field in
the `OptimAMPConf` dataclass in `trainer.py`. When active (and
`amp_dtype` is bf16), `self.scaler = torch.amp.GradScaler(...)` is
constructed with `init_scale=1.0` and near-identity growth/backoff factors
plus a huge `growth_interval`, effectively pinning the scale at ~1.0 forever
while keeping `enabled=True` (the found-inf skip-step safety net in
`unscale_()`/`step()` still applies):

```python
self.scaler = torch.amp.GradScaler(
    self.device,
    enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
    init_scale=1.0,
    growth_factor=1.0 + 1e-7,
    backoff_factor=1.0 - 1e-7,
    growth_interval=2**31 - 1,
)
```

**Gotcha:** `torch.amp.GradScaler`'s constructor hard-asserts
`growth_factor > 1.0` and `backoff_factor < 1.0` — a literal `1.0` raises
`AssertionError` at instantiation (hit this on the first resubmit attempt:
job `237943473` went to instant `Status <EXIT>` after 6s with
`AssertionError('The growth factor must be > 1.0.')` in the `.err` log —
same "near-instant EXIT = config/instantiation error, not a real training
crash" pattern documented elsewhere in this repo's memory). Near-identity
values (`1 ± 1e-7`) with `growth_interval=2**31-1` satisfy the assertions
while still behaviorally pinning the scale.

**Validation attempt #1 FAILED — checkpoint resume silently undid the fix.**
Killed the in-flight debug job (`bkill 237937642`), resubmitted from the
SAME epoch-5 checkpoint with the SAME debug settings as job `237943948`.
The fix's log line appeared at startup, but the collapse **recurred anyway**
with the byte-for-byte IDENTICAL subnormal scale (`2.938735877055719e-39`)
at nearly the same relative step (~1220/3617 into epoch 5). Root cause:
`_load_resuming_checkpoint()` unconditionally calls
`self.scaler.load_state_dict(checkpoint["scaler"])`, and
`torch.amp.GradScaler.state_dict()`/`load_state_dict()` round-trips **all
five** internal fields (`scale`, `growth_factor`, `backoff_factor`,
`growth_interval`, `_growth_tracker`) — since this checkpoint was saved by
an earlier run using the OLD dynamic scaler, loading it silently overwrote
the freshly-constructed pinned-scale scaler right after `__init__`. The
Step 10 fix above was correct in isolation; it just never survived the
checkpoint-resume step.

**Fix #2:** stash the bypass flag as `self._scaler_bypass_dynamic_scale`
during `__init__`, and skip `scaler.load_state_dict(...)` entirely in
`_load_resuming_checkpoint()` when it's `True`:

```python
if (
    self.optim_conf.amp.enabled
    and "scaler" in checkpoint
    and not getattr(self, "_scaler_bypass_dynamic_scale", False)
):
    self.scaler.load_state_dict(checkpoint["scaler"])
elif getattr(self, "_scaler_bypass_dynamic_scale", False):
    logging.info("(Ha) ... skipping restore of checkpoint's saved GradScaler state ...")
```

**Validation attempt #2 (in progress):** killed `237943948`, committed fix
#2, resubmitted as job `237950477` from the same checkpoint/debug settings.
Confirmed BOTH log lines fire at startup (pin + skip-restore), and early
steps show clean losses with zero `WEIGHT-NaN` hits. Full long-run
validation (does it stay clean past step ~1220, where BOTH prior attempts
collapsed?) tracked in `nbs/10_debug_zero_loss_collapse.ipynb`'s Open Items.

**Generalizable lesson:** when a fix touches any object restored from a
checkpoint (optimizer, scaler, scheduler, EMA, etc.), always check whether
`_load_resuming_checkpoint()`/`save_checkpoint()` round-trips that object's
state — a constructor-level fix is silently worthless if a later
`load_state_dict()` call overwrites it. This cost one full validation cycle
(~50 min of run time on job `237943948`) to discover.

**Once validated over a full run:** revert `train_batch_size` back to `4`
and unset `SAM3_NAN_DEBUG_HOOKS` before the next production submission (the
debug settings were only needed to catch this bug, not required by the fix
itself).

## Full active loss weight_dict (2026-07-17 reference)

From `sam3/train/configs/full_train_config.yaml` `loss_fns_find` /
`loss_fn_semantic_seg` (~L85-118) — re-check this file if the config has
since changed:

| loss key | weight | source class | guard status |
|---|---|---|---|
| `loss_bbox` | 5.0 | `Boxes` | unguarded (Step 4) |
| `loss_giou` | 2.0 | `Boxes` | unguarded (Step 4) |
| `loss_ce` | 20.0 | `IABCEMdetr` | unguarded (Step 4) |
| `presence_loss` | 20.0 | `IABCEMdetr` | guarded (Step 3) |
| `loss_mask` | 20.0 | `Masks` | guarded (Step 3) |
| `loss_dice` | 10.0 | `Masks` | guarded (Step 3) |
| `loss_semantic_seg` | 20.0 | `SemanticSegCriterion` | guarded (Step 3) |
| `loss_semantic_presence` | 1.0 | `SemanticSegCriterion` | dummy 0.0, not computed |
| `loss_semantic_dice` | 30.0 | `SemanticSegCriterion` | guarded (Step 3) |
