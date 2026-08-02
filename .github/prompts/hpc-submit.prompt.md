---
title: Submit a script to HPC GPU
---

# HPC GPU Submit (single-shot)

Submit a project bash script to the Acme HPC GPU queue with the correct
CUDA module loaded automatically.

## Parameters

- `script_path` (string, required)
  - Absolute or repo-relative path to the bash script to run. No default is set because this repo does not include a `finetuning.sh` script.
- `script_args` (string, optional)
  - Arguments to forward to the script, e.g. `--epochs 50 --batch 8`.
- `cuda_module` (string, optional)
  - Exact CUDA module, e.g. `cuda/12.8`. If omitted, detect from
    `torch.version.cuda` in the project venv.
- `queue` (string, default: `gpu_new_`)
  - HPC submit helper: `gpu_new_` or `gpu_`.

## Steps

1. Discover the repo root on HPC with the bundled helper:
   ```bash
   ssh hpc_muc 'bash -s' < .github/scripts/discover_repo_root.sh [PROJECT_NAME]
   ```
   Use the printed path as `<repo_root>` below. The helper searches
   `/home/hpc_storage/data/projects/$USER` for a directory matching the project
   name (default `sam3_fine_tune`) and falls back to the first directory
   containing `pyproject.toml`.
2. Resolve `script_path` to the HPC mount path if a Windows path is given.
3. Activate the project venv and read `torch.version.cuda`.
4. Map the PyTorch CUDA version to a module (e.g. `12.8` -> `cuda/12.8`).
5. Submit the job:

   ```bash
   ssh hpc_muc '<queue> "bash -c '\''source /ifxhome/$USER/.bashrc && module load <cuda_module> && source <repo_root>/.venv/bin/activate && cd <repo_root> && bash <script_path> <script_args>'\''"'
   ```

6. Report job ID, assigned GPU node, and confirm `torch.cuda.is_available()`
   inside the job.

## Example prompts

- `Run <repo-relative-script.sh> on HPC GPU.`
- `Submit scripts/my_training.sh --epochs 50 --batch 8.`
