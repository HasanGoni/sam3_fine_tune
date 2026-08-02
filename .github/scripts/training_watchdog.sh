#!/bin/bash
# training_watchdog.sh
#
# Monitors a SAM3 training experiment and resubmits the batch job if no
# matching job is RUN or PEND on the HPC cluster.  Intended to survive the
# cluster's idle-GPU auto-kill rule (TERM_OWNER) and transient scheduler
# failures, while NOT restarting a job that finished successfully.
#
# Usage:
#   training_watchdog.sh <experiment_log_dir> [check_interval_minutes] [bsub_cmd_file]
#
# Arguments:
#   experiment_log_dir   Directory passed to train.py as experiment_log_dir.
#                        The watchdog uses this path to identify the job in
#                        bjobs output and to store its own log/pid files.
#   check_interval_minutes
#                        Sleep between checks (default: 10).
#   bsub_cmd_file        File containing the bsub command to run when a
#                        restart is needed (default: <experiment_log_dir>/watchdog_bsub_cmd.sh).
#
# The bsub command file must be executable and should contain a single
# self-contained bsub invocation, for example:
#   #!/bin/bash
#   REPO=/home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune
#   LOGDIR=$REPO/experiments/9cat_unique_panda/bsub_logs
#   mkdir -p "$LOGDIR"
#   bsub -q gpu -gpu "num=1:j_exclusive=yes:gmem=40G" \
#     -R "select[ui==aiml_batch_training && osrel==80 && type==X64LIN] rusage[mem=200]" -n 4 \
#     -o "$LOGDIR/train.%J.out" -e "$LOGDIR/train.%J.err" \
#     "bash -c \"source /ifxhome/\\\$USER/.bashrc && source $REPO/.venv/bin/activate && module load cuda/12.8 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO/sam3/train && python train.py -c configs/full_train_config.yaml --num-gpus 1 --num-nodes 1\""
#
# Start in the background with:
#   nohup /path/to/training_watchdog.sh /path/to/experiment > /dev/null 2>&1 &
#
# Stop with:
#   kill $(cat /path/to/experiment/watchdog.pid)

set -euo pipefail

EXPERIMENT_DIR="${1:?Usage: $0 <experiment_log_dir> [check_interval_minutes] [bsub_cmd_file]}"
CHECK_INTERVAL_MINUTES="${2:-10}"
BSUB_CMD_FILE="${3:-$EXPERIMENT_DIR/watchdog_bsub_cmd.sh}"

LOG_FILE="$EXPERIMENT_DIR/watchdog.log"
PID_FILE="$EXPERIMENT_DIR/watchdog.pid"
BSUB_LOG_DIR="$EXPERIMENT_DIR/bsub_logs"

mkdir -p "$BSUB_LOG_DIR"

# ---------------------------------------------------------------------------
# PID-file lock: prevent multiple watchdogs for the same experiment.
# ---------------------------------------------------------------------------
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog already running (PID $OLD_PID) for $EXPERIMENT_DIR" >&2
        exit 1
    fi
fi

echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

# ---------------------------------------------------------------------------
# Validate the bsub command file exists.
# ---------------------------------------------------------------------------
if [[ ! -f "$BSUB_CMD_FILE" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: bsub command file not found: $BSUB_CMD_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

job_exists_for_experiment() {
    # Check for any job (RUN, PEND, etc.) whose output-file path references the
    # experiment directory.  Using the OUTPUT_FILE column is robust against bsub
    # commands that reference the experiment dir via shell variables (e.g.
    # $EXPERIMENT_DIR) -- bjobs shows those variables unexpanded in the COMMAND
    # column but expands them in OUTPUT_FILE.
    bjobs -o "jobid stat output_file command" 2>/dev/null | grep -q "$EXPERIMENT_DIR" || return 1
}

latest_out_log() {
    # Find the most recent train.*.out log for this experiment.
    ls -t "$BSUB_LOG_DIR"/train.*.out 2>/dev/null | head -n 1
}

training_completed_successfully() {
    local latest_out
    latest_out=$(latest_out_log)
    [[ -n "$latest_out" ]] || return 1
    grep -q "Successfully completed\." "$latest_out" 2>/dev/null
}

resubmit_training() {
    log "Resubmitting training job via $BSUB_CMD_FILE"
    # Redirect stdin away from the subshell so any command inside the bsub
    # file that reads stdin does not steal loop input.
    bash "$BSUB_CMD_FILE" < /dev/null >> "$LOG_FILE" 2>&1
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
log "Watchdog started for experiment: $EXPERIMENT_DIR"
log "PID: $$"
log "Check interval: ${CHECK_INTERVAL_MINUTES} minutes"
log "Bsub command file: $BSUB_CMD_FILE"

while true; do
    if job_exists_for_experiment; then
        log "Training job found for experiment (RUN/PEND)"
    else
        if training_completed_successfully; then
            log "Latest training log reports 'Successfully completed.' -- watchdog exiting."
            exit 0
        fi

        log "No training job found and latest log does not report successful completion."
        resubmit_training

        # Give the scheduler a short window to register the new job.
        sleep 60
        if job_exists_for_experiment; then
            log "Resubmitted job is now visible to scheduler."
        else
            log "WARNING: Resubmitted job is not yet visible to scheduler; will retry next cycle."
        fi
    fi

    sleep "${CHECK_INTERVAL_MINUTES}m"
done
