---
name: training-watchdog
description: 'Start a background watchdog on the HPC that monitors a SAM3 training experiment and resubmits the batch job if it disappears from the scheduler, while leaving successfully completed jobs alone.'
argument-hint: '[--experiment-dir <path>] [--check-interval <minutes>]'
user-invocable: true
disable-model-invocation: false
---
# Training Watchdog

Start a lightweight background process on the Acme HPC login node that
polls LSF every N minutes for a specific SAM3 training experiment.  If the
job is no longer RUN or PEND, the watchdog resubmits it using a stored bsub
command.  If the latest `.out` log reports `Successfully completed.`, the
watchdog exits cleanly instead of restarting.

## When To Use

- You are starting a long-running SAM3 fine-tuning job that may be killed by
the cluster's idle-GPU auto-kill rule (GPU load < ~5% for > ~5 minutes shows
up as `TERM_OWNER`).
- You want the job to auto-restart after transient failures without manual
intervention.
- You do **not** want the watchdog to restart a job that finished normally.

## Prerequisites

- SSH access to the HPC is configured (see `ssh-hpc-connect`).
- The training experiment directory already exists or will be created by the
first bsub submission.
- The SAM3 training config (`full_train_config.yaml`) has already been updated
for the dataset you want to train on (see `finetune-merged-dataset`).

## Inputs

- `experiment_dir`: the value of `paths.experiment_log_dir` from
  `full_train_config.yaml`.  The watchdog stores `watchdog.log`,
  `watchdog.pid`, and `watchdog_bsub_cmd.sh` here and uses this path to
  identify the job in `bjobs -w` output.
- `check_interval_minutes`: how often to poll the scheduler (default: 10).
- `bsub_cmd_file`: optional path to a file containing the exact bsub command
  to run on restart.  Default is `<experiment_dir>/watchdog_bsub_cmd.sh`.

## Procedure

### 1. Create the bsub command file

Write the exact bsub command you want the watchdog to run into
`<experiment_dir>/watchdog_bsub_cmd.sh`.  The file must be executable and
self-contained.  Example for a single-GPU run on the `gpu` queue:

```bash
#!/bin/bash
set -euo pipefail

REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
EXPERIMENT_DIR=$REPO/experiments/9cat_unique_panda
LOGDIR=$EXPERIMENT_DIR/bsub_logs
mkdir -p "$LOGDIR"

bsub -q gpu -gpu "num=1:j_exclusive=yes:gmem=40G" \
  -R "select[ui==aiml_batch_training && osrel==80 && type==X64LIN] rusage[mem=200]" -n 4 \
  -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
  "bash -c \"source /ifxhome/\\\$USER/.bashrc && source $REPO/.venv/bin/activate && module load cuda/12.8 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1\""
```

Make it executable:

```bash
ssh hpc_muc 'chmod +x /path/to/experiment/watchdog_bsub_cmd.sh'
```

### 2. Start the watchdog

```bash
ssh hpc_muc '
  EXPERIMENT_DIR=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_unique_panda
  WATCHDOG=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/.github/scripts/training_watchdog.sh
  nohup "$WATCHDOG" "$EXPERIMENT_DIR" 10 "$EXPERIMENT_DIR/watchdog_bsub_cmd.sh" > /dev/null 2>&1 &
  echo $!
'
```

The watchdog writes its PID to `<experiment_dir>/watchdog.pid` and logs to
`<experiment_dir>/watchdog.log`.

### 3. Verify it is running

```bash
ssh hpc_muc 'cat /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_unique_panda/watchdog.pid && tail -n 5 /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune/experiments/9cat_unique_panda/watchdog.log'
```

You should see a line like:

```text
[2026-07-23 23:00:00] Watchdog started for experiment: .../9cat_unique_panda
[2026-07-23 23:00:00] Training job found for experiment (RUN/PEND)
```

### 4. Stop the watchdog

If you need to stop it manually (for example, after the run completes or you
want to change the bsub command):

```bash
ssh hpc_muc 'kill $(cat /path/to/experiment/watchdog.pid)'
```

A successfully completed training run stops the watchdog automatically on its
next poll.

## Decision Logic

- Always store the bsub command in a file rather than embedding it in the
  watchdog script, so the same watchdog can guard many different experiments.
- Default check interval is 10 minutes.  Use a shorter interval only if you
  expect very short kills and want faster recovery; use a longer interval to
  reduce scheduler load.
- Do not run more than one watchdog per experiment directory.  The script uses
  a PID-file lock and exits if another watchdog is already active.
- If a job disappears but the latest `train.*.out` log contains
  `Successfully completed.`, the watchdog exits.  This prevents infinite
  re-submission of finished jobs.

## Output Format

1. Confirmation that the watchdog process started (PID).
2. Tail of `watchdog.log` showing the first check succeeded.
3. Optional: evidence from a controlled test that the watchdog resubmits when
   no matching job is present.
