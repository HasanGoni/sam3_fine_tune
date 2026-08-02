---
name: hpc-gpu-submit
description: 'Run a project bash script on the Acme HPC GPU queue (gpu_new_) after auto-detecting the correct CUDA module for the active venv. Use for training jobs, finetuning scripts, or any GPU workload that needs bashrc + venv + CUDA module + bsub submission.'
argument-hint: '[script_path] [script_args...]'
user-invocable: true
disable-model-invocation: false
---
# HPC GPU Submit

Run a bash script on the Acme HPC GPU queue (`gpu_new_`) with the project
venv activated and the CUDA module matched to the PyTorch build in that venv.

## When To Use

- Kicking off model training / finetuning from a bash script in the project.
- Running any GPU Python workload that requires `module load cuda/...` before
  execution.
- Need the HPC job to source `~/.bashrc` so `module`, conda, aliases, etc. are
  available inside the bsub environment.
- Prefer this skill over the single-shot `.github/prompts/hpc-submit.prompt.md`
  when you want the agent to auto-detect CUDA and validate the launch.

## Prerequisites

Run the `configure-repo` skill first to resolve `REPO_ROOT` and the detected
`CUDA_MODULE` in one round-trip (it also verifies SSH connectivity as part of
reaching HPC). Use its output directly instead of repeating repo-root
discovery or torch/cuda detection below.

## Inputs

- `script_path`: absolute or repo-relative path to the bash script to run
  (required; no default — provide the repo-relative or absolute path to the script).
- `script_args...`: any arguments to forward to the script.
  - Example training arguments (depends on the actual script):
    - `--epochs <int>`
    - `--batch <int>`
    - `--imgsz <int>`
- Optional named context:
  - `cuda_module`: exact module name, e.g. `cuda/12.8`. If omitted, the skill
    detects it from `torch.version.cuda` in the project venv.
  - `queue`: `gpu_new_` (default) or `gpu_`.
  - `venv`: path to activate script (default: project root `.venv/bin/activate`).

## Procedure

1. **Resolve repo root + CUDA module.**

   - Run the `configure-repo` skill (`ssh hpc_muc 'bash -s' < .github/scripts/hpc_repo_context.sh`)
     to get `REPO_ROOT` and `CUDA_MODULE` in one call, unless the caller already
     provided `cuda_module` explicitly.
   - Convert any Windows path the user provides to the HPC mount path.
   - Verify the detected `CUDA_MODULE` actually exists via `module avail cuda`
     before submitting (see `configure-repo`'s Decision Logic for the fallback
     if it isn't listed).
2. **Submit the job.**

   - Use `gpu_new_` by default.
   - **Preferred working pattern (avoids source-group rejection):** open an
     interactive SSH session, activate the venv and load the CUDA module on the
     login node, then submit a simple command string that just runs the script:

     ```bash
     ssh hpc_muc -t 'source /ifxhome/$USER/.bashrc && source ${REPO_ROOT}/.venv/bin/activate && module load <cuda_module> && cd ${REPO_ROOT} && bsub -Is -q gpu_new_ -R "rusage[mem=<memory>]" "bash <script_path> <script_args>"'
     ```

   - Avoid embedding `source ... && source ... && module load ... && bash ...`
     inside the bsub command string; that has triggered HPC source-group
     rejections (`You are not in the source group`) on some execution hosts.
   - Wait for dispatch and capture the first lines of stdout to confirm the
     GPU is visible (e.g. `nvidia-smi` or a quick torch CUDA check).
3. **Validate launch.**

   - Confirm `torch.cuda.is_available()` is `True` inside the job.
   - Print job ID and assigned GPU node.

## Decision Logic

- If `torch.version.cuda` is `None`, fall back to `cuda/12.8` and warn.
- If the detected module is not available on HPC, list the closest `cuda/...`
  modules and ask which to use (or pick the same major version).
- If the user supplies `cuda_module`, use it exactly and skip detection.
- **Source-group rejection on HPC batch queues (learned 2026-07-17):**
  submitting from a non-interactive one-liner with a long
  `source ~/.bashrc && source venv && module load cuda/... && bash ...`
  command string can fail with:

  ```
  You are not in the source group. Skipping processing.
  Please ask DSX Team to add you!
  ```

  This was reproduced for CPU submissions on `static_hi`; the same pattern
  likely affects GPU queues. The fix is to activate the environment on the
  login node first and pass only the actual workload command (`bash
  <script_path>` or `python <script_path>`) to `bsub`.
- **Idle-GPU auto-kill policy (learned 2026-07-17):** this HPC cluster
  automatically kills a job if its GPU utilization stays below ~5% for more
  than ~5 minutes continuously. This shows up as `TERM_OWNER: job killed by
  owner.` / exit code `143` in the `.out` job summary — easy to mistake for a
  manual `bkill` or an admin action, but it's an automated idle-GPU policy,
  not a person killing the job. Known triggers:
  - A long dataset/model setup phase before the training loop starts (in this
    repo's trainer, the gap between `Setting up components...` and the first
    `Train Epoch: [N][ 0/...]` log line can be 30s-several minutes on a fresh
    experiment dir — mostly CPU/IO work with near-zero GPU use).
  - Any step with unusually high `Data Time` relative to `Batch Time` in this
    repo's `train_utils.py` logging (e.g. `Batch Time: 17.12 | Data Time:
    13.65` — 13.65s of that step was data loading, not GPU compute) — if
    several such slow, IO-bound steps stack up, cumulative GPU idle time can
    cross the 5-minute threshold even though the job is actually still making
    progress.
  - Resuming into a brand-new/never-before-used `experiment_log_dir` (as
    opposed to one that's already been trained in) tends to have a slower
    first-epoch warm-up (directory creation, prediction-dump dir setup, first
    dataset indexing) than resuming into an already-warm one.
  - If a job you expected to run for a long time (e.g. a multi-epoch repro or
    a full training run) shows `Status <EXIT>` / exit code `143` /
    `TERM_OWNER` in `bjobs -l <id>` or the `.out` summary sooner than
    expected, check this policy **before** assuming a person killed it or
    that it's a script bug — cross-check the last few `Train Epoch:` log
    lines for a `Data Time` spike right before the kill.
  - Mitigation if this repeatedly interrupts a real job: avoid submitting
    into a brand-new empty `experiment_log_dir` when a warm one is available,
    and if unavoidable, expect the first few minutes of a fresh run to be at
    higher risk and don't be surprised by an early kill purely from setup
    overhead.

## Output Format

1. Detected CUDA module and torch version.
2. Command that was submitted to `gpu_new_`.
3. Job ID and assigned node.
4. Launch validation output (CUDA available / GPU name).
