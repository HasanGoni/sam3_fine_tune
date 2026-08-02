---
name: configure-repo
description: 'Resolve the repo root, activate the project venv, and detect the python/torch/cuda versions on the Acme HPC in a single ssh round-trip. Run this FIRST as a prerequisite for any other HPC skill (hpc-cpu-submit, hpc-gpu-submit, coco-format-conversion, merge-category-dataset, finetune-merged-dataset) so they do not each re-derive the same context separately.'
argument-hint: '[project_name]'
user-invocable: true
disable-model-invocation: false
---
# Configure Repo

Resolve everything other HPC skills need up front, in **one** ssh call, so
they don't each spend a separate round-trip rediscovering the same facts.

## When To Use

- As the first step of any other HPC skill in this repo (`hpc-cpu-submit`,
  `hpc-gpu-submit`, `coco-format-conversion`, `merge-category-dataset`,
  `finetune-merged-dataset`, ...).
- Whenever you need the HPC repo root, confirmation the venv activates
  cleanly, or the torch/cuda version to pick the right `cuda/X.Y` module.

## Prerequisites

Verify HPC connectivity first — use the `ssh-hpc-connect` skill or run
`ssh hpc_muc echo connected` and confirm no password prompt.

## Inputs

- `project_name`: repo directory name to search for (default: `sam3_fine_tune`).

## Procedure

Run the single consolidated script:

```bash
ssh hpc_muc 'bash -s -- [project_name]' < .github/scripts/hpc_repo_context.sh
```

This prints:

```text
REPO_ROOT=/home/hpc_storage/data/projects/<user>/.../sam3_fine_tune
PYTHON_BIN=<repo>/.venv/bin/python
PYTHON_VERSION=3.12.11
TORCH_VERSION=2.10.0+cu128
TORCH_CUDA=12.8
CUDA_MODULE=cuda/12.8
```

Reuse these values directly in whatever command you build next — don't
re-run `discover_repo_root.sh`, re-activate the venv, or re-check
`torch.version.cuda` separately later in the same task.

**`CUDA_MODULE` is a best guess** derived from `torch.version.cuda` (e.g.
`12.8` → `cuda/12.8`). Before relying on it for a GPU job submission, verify it
actually exists:

```bash
ssh hpc_muc 'module avail cuda 2>&1 | grep <CUDA_MODULE base version>'
```

If it's not listed, fall back to the closest available `cuda/X.Y` with the
same major version and note the substitution to the user.

## Decision Logic

- If `REPO_ROOT` discovery fails (project not found under
  `/home/hpc_storage/data/projects/$USER`), ask the user for the exact HPC
  path instead of guessing.
- If `TORCH_CUDA` comes back `unknown`/`None`, torch isn't importable in the
  venv — stop and flag this before submitting any GPU job (submitting anyway
  will fail on the compute node).
- Set `FORCE_REFRESH=1` before the ssh command only if the repo may have moved
  or the cached path looks stale; otherwise the cached `REPO_ROOT` (from
  `~/.cache/hpc_repo_root/<project_name>`) is reused automatically.

## Output Format

1. `REPO_ROOT`, `PYTHON_BIN`, `PYTHON_VERSION`, `TORCH_VERSION`, `TORCH_CUDA`,
   `CUDA_MODULE` — pass these straight into the next skill/command.

## Repo conventions

- `scratch/` is the directory for ad-hoc / temporary Python scripts. Put
  exploration, debugging, and one-off data-audit scripts there, not at the
  repo root.

## Example prompts

- `Set up the repo context before we run training.`
- `What's the repo root and cuda module on HPC right now?`
