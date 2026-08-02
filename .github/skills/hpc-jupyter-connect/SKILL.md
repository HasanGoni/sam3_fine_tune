---
name: hpc-jupyter-connect
description: 'Start a Jupyter Lab server on the Acme HPC via dsx_jl inside the dsx_p312_only_uv environment, extract the tunneled URL, and connect the active VS Code notebook to that remote kernel automatically.'
argument-hint: '[--queue batch|gpu] [--root-dir <path>] [--timeout <seconds>]'
user-invocable: true
disable-model-invocation: false
---
# HPC Jupyter Connect

Start a Jupyter Lab server on the Acme HPC using the DSX `dsx_p312_only_uv`
environment + project `.venv`, capture the tunneled URL, and automatically
select that server as the kernel for the current VS Code notebook.

## When To Use

- You want to run a notebook in VS Code against the HPC Python environment.
- You normally run `dsx_p312_only_uv`, then `dsx_jl`, copy the URL, and paste
  it into VS Code's Jupyter kernel picker.
- You want this connection automated from a single request.

## Prerequisites

Before using this skill, verify SSH HPC connectivity. Use the `ssh-hpc-connect`
skill or run `ssh hpc_muc echo connected` locally and confirm it works without a
password. If it fails, set up key-based login first.

## How It Works

1. SSH to `hpc_muc` (login node).
2. Enter the DSX virtualenv with the alias:
   `dsx_p312_only_uv`
   - This changes `$HOME` to `/home/ai_dsx/jupyter-homes/userB` and loads the
     `ai_dsx/virtualenvs/only_uv_p312` module.
3. Discover the repo root and activate the project venv:
   ```bash
   REPO_ROOT=$(bash .github/scripts/discover_repo_root.sh sam3_fine_tune)
   source "${REPO_ROOT}/.venv/bin/activate"
   ```
4. Run `dsx_jl` to start Jupyter Lab via `bsub` and create an SSH tunnel.
5. Wait for the output line:
   `Access Jupyter at: Hostname: http://<hostname>:<port>/lab?token=<token>`
6. Parse the hostname, port, and token.
7. In VS Code, select the current notebook's kernel → **Existing Jupyter Server**,
   and paste the URL.

## Inputs

Defaults used by the manual launch:

- Queue: `batch`.
- Root directory: discovered dynamically from the current workspace using `.github/scripts/discover_repo_root.sh`.
- Startup timeout: 180 seconds.

> **Note:** If the wrapper scripts `scripts/hpc_get_or_start_jupyter.sh` and `scripts/hpc_start_jupyter.exp` exist in the repo, you can use them for a one-command launch. If they are missing, use the manual procedure below.

## Discovering the repo root

To avoid hard-coding paths, use the bundled helper from the project root (run it after `ssh hpc_muc`):

```bash
bash .github/scripts/discover_repo_root.sh [PROJECT_NAME]
```

It searches `/home/hpc_storage/data/projects/$USER` for a directory matching `PROJECT_NAME` (default `sam3_fine_tune`) and falls back to the first directory containing a `pyproject.toml`. Use the printed path as `$REPO_ROOT` in the commands below.

## Procedure

### Start Jupyter Lab on HPC

1. `ssh -t hpc_muc`
2. `source /ifxhome/$USER/.bash_aliases` (replace `$USER` with your HPC username if needed)
3. Verify the DSX alias is available:
   ```bash
   type dsx_p312_only_uv
   ```
   If the alias is missing, **stop**. Tell the user to add the DSX environment alias to `~/.bash_aliases` (or activate the equivalent `only_uv_p312` environment manually), then retry.
4. `dsx_p312_only_uv`
5. `REPO_ROOT=$(bash .github/scripts/discover_repo_root.sh sam3_fine_tune)`
6. `source "${REPO_ROOT}/.venv/bin/activate"`
7. Verify `dsx_jl` is available:
   ```bash
   type dsx_jl
   ```
   If `dsx_jl` is missing, **stop**. Tell the user that `dsx_jl` must be added to the path or aliased (it normally appears after entering `dsx_p312_only_uv`), then retry.
8. `dsx_jl --root-dir "${REPO_ROOT}"`
9. Copy the **Hostname** URL from the `Access Jupyter at:` block.

If the repo is cloned elsewhere or the project name differs, pass the correct
name to `discover_repo_root.sh` or set `REPO_ROOT` manually.

### Connect VS Code to the server

1. In the active notebook, open the kernel selector (top-right).
2. Choose **Select Another Kernel** → **Existing Jupyter Server**.
3. Paste the full `http://<hostname>:<port>/lab?token=...` URL.
4. Select the kernel named like the project venv.

### Validation

Run a cell with:

```python
import sys, torch
print(sys.executable)
print(torch.cuda.is_available())
```

Confirm the path is inside the project venv and CUDA is available if on GPU.

## Where the link is stored

`dsx_jl` may write the last captured URL to:
`/home/ai_dsx/jupyter-homes/$USER/.dsx_jl_last.url`

If that file exists, retrieve the stored URL manually with:

```bash
ssh hpc_muc 'cat /home/ai_dsx/jupyter-homes/$USER/.dsx_jl_last.url'
```

## Limitations

- This skill **cannot** directly control VS Code's kernel picker through tools.
  It prints/retrieves the URL; the user pastes it into **Existing Jupyter Server**.
- `dsx_jl` keeps running on the HPC login node and maintains the tunnel, so the
  SSH session can close after the URL is shown.
- `dsx_jl` is only available after entering `dsx_p312_only_uv`; before that it is
  a symlink to `__dsx_unavailable`.

## Example Prompts

- `Connect this notebook to HPC Jupyter.`
- `Start dsx_jl on HPC and give me the kernel URL.`
- `Launch HPC Jupyter on GPU queue.`
