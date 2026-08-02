---
name: ssh-hpc-connect
description: 'Check the local SSH setup for connecting to the Acme HPC and guide the user to configure passwordless login. Required before using any HPC skills.'
argument-hint: '[--site muc|vih]'
user-invocable: true
disable-model-invocation: false
---
# SSH HPC Connect

Verify the local SSH configuration for connecting to the Acme HPC and, if
needed, guide the user through setting up passwordless key-based login.

## Why this matters

All HPC skills in this repository assume `ssh hpc_muc` (or an equivalent alias)
connects directly to the Acme HPC without asking for a password. If the SSH
keys or `~/.ssh/config` entry are missing, every HPC command will hang at a
password prompt.

## Checklist

Before running any HPC skill, verify the following on the user's local machine:

1. `~/.ssh` directory exists.
2. A valid SSH private/public key pair exists, e.g.:
   - `~/.ssh/id_rsa` + `~/.ssh/id_rsa.pub`
   - `~/.ssh/id_ed25519` + `~/.ssh/id_ed25519.pub`
3. The public key has been uploaded to the HPC account.
4. `~/.ssh/config` contains a host alias for the HPC login node.
5. `ssh hpc_muc` connects without a password.

## How to check

Run on the local machine (not on HPC):

```bash
ls -la ~/.ssh
```

Expected files:

```text
id_rsa
id_rsa.pub
config
known_hosts
```

Then verify the alias works:

```bash
ssh hpc_muc echo "connected"
```

If the output is `connected` without prompting for a password, the setup is
done.

## Default HPC host aliases

| Site | Direct SSH command                  | Suggested alias |
|------|-------------------------------------|-----------------|
| MUC  | `ssh <unix-account>@login.muc.example.com` | `hpc_muc`       |
| VIH  | `ssh <unix-account>@login.vih.example.com` | `hpc_vih`       |

> Replace `<unix-account>` with the user's actual Unix account name.

## If the setup is incomplete

**Stop and ask the user.** Do not attempt to connect further. Provide the
following steps clearly.

### Step 1: Create an SSH key pair (if none exists)

```bash
ssh-keygen -t ed25519 -C "<your-email>@example.com"
# Press Enter to accept the default location (~/.ssh/id_ed25519)
# Do not set a passphrase if you want fully passwordless login
```

### Step 2: Copy the public key to the HPC login server

```bash
ssh-copy-id <unix-account>@login.muc.example.com
# Or for VIH:
# ssh-copy-id <unix-account>@login.vih.example.com
```

If `ssh-copy-id` is not available, copy `~/.ssh/id_ed25519.pub` manually and
append it to `~/.ssh/authorized_keys` on the HPC login server.

### Step 3: Add an alias to `~/.ssh/config`

Create or edit `~/.ssh/config`:

```text
Host hpc_muc
    HostName login.muc.example.com
    User <unix-account>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host hpc_vih
    HostName login.vih.example.com
    User <unix-account>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

### Step 4: Test the connection

```bash
ssh hpc_muc echo "connected"
```

If it prints `connected` without asking for a password, the setup is complete.

## Procedure for the agent

1. Ask the user: *"Have you already configured passwordless SSH to the HPC
   (e.g., `ssh hpc_muc` works without a password)?"*
2. If yes, proceed with the requested HPC skill.
3. If no or unknown, run the local checks above (`ls -la ~/.ssh`, `ssh hpc_muc
   echo connected`).
4. If the check fails, **stop** and present the setup steps above. Do not
   continue with any HPC command until the user confirms `ssh hpc_muc` works.

## Example prompts

- `Check my SSH HPC setup.`
- `Connect to HPC.`
- `Run my script on HPC GPU.` (agent should perform this SSH check first)

## Notes

- The agent cannot create SSH keys for the user because doing so requires
  knowing the user's Unix account name and HPC password.
- The agent should never ask the user to type their HPC password into the chat.
  If a password prompt appears in the terminal, tell the user to type it
  directly in the terminal.
