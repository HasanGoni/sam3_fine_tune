---
name: hpc-cpu-submit
description: 'Submit a Python script to the Acme HPC CPU queue using bsub with the project venv activated. Use for preprocessing, data conversion, or any CPU workload that needs bashrc + venv + bsub submission.'
argument-hint: '[script_path] [script_args...]'
user-invocable: true
disable-model-invocation: false
---
# HPC CPU Submit

Submit a Python script on the Acme HPC CPU queue with the project venv
activated.

## When To Use

- Running preprocessing scripts (e.g., COCO format conversion, dataset merging).
- Running any CPU-bound Python workload that should run on the HPC batch/interactive queue.
- Need the HPC job to source `~/.bashrc` so `module`, conda, aliases, etc. are available inside the bsub environment.

## Prerequisites

Run the `configure-repo` skill first to resolve `REPO_ROOT` in one round-trip
(it also verifies SSH connectivity as part of reaching HPC). Use its output
directly instead of repeating repo-root discovery below.

## Inputs

- `script_path`: absolute or repo-relative path to the Python script to run (required).
- `script_args...`: any arguments to forward to the script.
- Optional named context:
  - `queue`: bsub queue or submit helper (default: interactive CPU submit as used by `cpu_c`).
  - `venv`: path to activate script (default: discovered repo root `.venv/bin/activate`).
  - `memory`: memory request (default: `50G`).
  - `cores`: number of cores (default: `30`).

## Default bsub command

Use the HPC `cpu_c` helper alias (preferred) or the equivalent full `bsub`
command:

```bash
bsub -Is -R "type=X64LIN && osrel>=70 && ui=pytest rusage[mem=<memory>]" -n <cores> <command>
```

## Procedure

1. **Resolve the repo root on HPC.**

   Run the `configure-repo` skill (`ssh hpc_muc 'bash -s' < .github/scripts/hpc_repo_context.sh`)
   to get `REPO_ROOT` in one call instead of running `discover_repo_root.sh`
   separately.

2. **Resolve paths.**

   - Convert any Windows path the user provides to the HPC mount path.
   - Make data/output paths absolute or clearly repo-relative.

3. **Submit the job.**

   **Preferred working pattern (avoids source-group rejection):**
   open an interactive SSH session, activate the venv on the login node, then
   submit a simple `python /abs/path/to/script.py` command via `cpu_c`:

   ```bash
   ssh hpc_muc -t 'source /ifxhome/$USER/.bashrc && source ${REPO_ROOT}/.venv/bin/activate && cpu_c "python /abs/path/to/script.py <args>"'
   ```

   If you must use a single non-interactive SSH one-liner, the simple
   `cpu_c "python ..."` form above is still more reliable than embedding
   `source && source && python` inside the bsub command string.

4. **Validate launch.**

   - Confirm the job is dispatched.
   - Capture the first lines of stdout to confirm the script starts (e.g., Python version, first log line).

## Decision Logic

- Prefer the user's `cpu_c` alias when it is available; it has been observed to
  avoid HPC source-group restrictions that a manually expanded `bsub` command
  with embedded `source ... && python ...` can trigger.
- If the user supplies custom `memory` or `cores` and you need a manual `bsub`,
  substitute those values into the resource string, but keep the command string
  simple (just `python <script_path> <script_args>`) and rely on the venv being
  pre-activated on the login node.
- If the script path is repo-relative, use absolute paths or prepend
  `cd ${REPO_ROOT} &&` so imports and relative paths resolve correctly.

## Known Issues

- **Source-group rejection on `static_hi`:** Submitting a bsub job from a
  non-interactive one-liner with a long `source ~/.bashrc && source venv &&
  python ...` command string can fail with:

  ```
  You are not in the source group. Skipping processing.
  Please ask DSX Team to add you!
  ```

  This was reproduced when running `ssh hpc_muc 'bsub -Is -R ... "source ... &&
  python ..."'`. The same user and same queue worked when the venv was
  activated on the login node first and only `python /abs/path/script.py` was
  passed to `cpu_c` / `bsub`.
- **Different execution hosts enforce checks differently:** Jobs landing on
  `muclc0849` triggered the source-group error, while `muclc0840` accepted the
  simple command form. Keep the submitted command simple to minimize host-
  specific prologue/esub issues.

## Output Format

1. Discovered repo root.
2. Full bsub command that was submitted.
3. Job ID and assigned host.
4. Launch validation output (first log lines / script output).

## Example prompts

- `Submit coco_format.py to HPC CPU.`
- `Run preprocessing/coco_format.sh on HPC CPU.`
- `Submit scripts/merge_datasets.py --input-dir /path/to/data --output-dir /path/to/out.`
