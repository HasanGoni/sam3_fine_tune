---
name: git-feature-workflow
description: 'Decide whether a change needs a feature branch + merge request or can be committed directly to main, and if a branch is needed, derive its name/prefix and drive it through commit -> push -> MR -> self-review -> merge without the user having to specify branch creation or naming each time. Use at the START of any non-trivial code/config change in this repo.'
argument-hint: '[short description of the change you are about to make]'
user-invocable: true
disable-model-invocation: false
---
# Git Feature Workflow

Decide branch-vs-direct-to-main up front, and if a branch is needed, run the
whole branch → commit → push → MR lifecycle without asking the user to spell
out "create a new branch" / "what should the branch name be" each time. This
repo has no CI pipeline, so this workflow (branch isolation + self-review
before merge) is the primary safety checkpoint before anything lands on
`main` — see the reasoning in the 2026-07-17 session that established this.

## When To Use

Run this FIRST, before starting the actual change, whenever the user asks
for a bug fix, new feature, refactor, or any change that:

- Touches training/loss/model code, config files, or anything that needs an
  HPC job to validate before you'd trust it on `main`.
- Spans multiple files or multiple logical commits.
- Is an experiment that might be abandoned or need iteration.

Do **NOT** branch for trivial changes: a single typo fix, a one-line doc
wording tweak, a single config value change with no validation needed. Those
go straight to `main` with a normal commit.

## Decision Logic (branch vs. direct-to-main)

| Change type | Action |
|---|---|
| Typo, one-line doc/comment fix, single trivial config value | Commit directly to `main` |
| Bug fix, new feature, refactor, anything touching `sam3/` code | Feature branch → MR |
| Anything needing an HPC job (training run, data audit) to validate | Always a branch — validate on the branch, merge only after |
| Multi-file / multi-commit change, or an experiment that might be abandoned | Branch — keeps the change revertable as one unit |

If genuinely unsure which bucket a change falls into, default to branching —
the overhead is low and it's the safer default in a repo with no CI.

## Procedure

1. **Check git status first, over the standard HPC ssh route** (per repo
   memory — git ops for this repo run via
   `ssh hpc_muc "cd /home/hpc_storage/data/projects/userB/test_sam3/sam3_fine_tune && git ..."`,
   not local Windows git-bash):
   ```bash
   ssh hpc_muc "cd <repo_root> && git status --short && git branch --show-current"
   ```
   If there's already unrelated uncommitted work in the tree, leave it alone —
   don't fold it into this branch/commit (per the standing delete/scope-safety
   rule — never touch files outside the current task's scope without asking).

2. **Derive the branch name yourself** — don't ask the user to supply one
   unless the change is ambiguous. Use `<type>/<short-kebab-summary>`:
   - Prefix from conventional commit types already used in this repo's
     history: `fix/`, `feat/`, `docs/`, `chore/`, `refactor/`, `perf/`.
   - Summary: 2-5 words, kebab-case, describing the concrete change (e.g.
     `fix/loss-nan-ce-bbox-guards`, `feat/append-images-existing-category`).
   - Create it from the current branch tip:
     ```bash
     ssh hpc_muc "cd <repo_root> && git checkout -b <type>/<short-kebab-summary>"
     ```

3. **Do the work on the branch.** Commit in focused, logical units (per user's
   commit-style preference — one commit per logical change, not one commit
   per file). Update `CHANGELOG.md` per the `git-last-change-summary` skill
   as you go (or once at the end) — reverse-chronological, today's heading at
   the top.

4. **Validate on the branch before merging**, if the change needs it — e.g.
   run the relevant HPC job (`finetune-merged-dataset`, `hpc-gpu-submit`,
   `hpc-cpu-submit`) and confirm the result before considering it done. Never
   merge an unvalidated change that was supposed to be validated.

5. **Push the branch** (ask before pushing UNLESS the user has already given
   standing push authorization for this repo — check user memory first; as of
   2026-07-17 no standing authorization is recorded for `sam3_fine_tune`, so
   ask/confirm on first push of a session, then it's fine to keep pushing
   follow-up commits to the same already-pushed branch without re-asking):
   ```bash
   ssh hpc_muc "cd <repo_root> && git push -u origin <branch-name>"
   ```
   GitLab prints a direct "create a merge request" URL in the push output —
   surface that link to the user.

6. **Fill the MR using `.gitlab/merge_request_templates/Default.md`** (create
   it if missing — see that file for the expected Summary/Changes/
   Validation/Checklist structure). Always fill in the Validation section
   with concrete evidence (job ID, result, what was checked), not just "it
   should work."

7. **Self-review before merging**, even for solo-authored, empirically
   validated changes — a passing HPC job validates *behavior*, not diff
   hygiene (stray files, stale comments, out-of-scope changes). A quick pass
   over the MR's "Changes" tab is enough; no external reviewer is required
   for solo work in this repo.

8. **Merge options**: "Delete source branch when merge request is accepted"
   ✓ (keeps branch list clean). Leave "Squash commits" unchecked if the
   commits are already focused/logical — squashing loses useful history for
   no benefit in that case.

## Merge Conflict Resolution

If the user reports a merge conflict (or `git merge-tree` / a merge attempt
shows conflicts), do not just repeatedly run `git merge` and hope. Use a
clean rebase workflow:

1. **Abort any half-finished merge first**:
   ```bash
   ssh hpc_muc "cd <repo_root> && git merge --abort 2>/dev/null || true"
   ```

2. **Identify the real base of the feature branch.** If the feature branch
   was accidentally based on another feature branch (e.g. DINOv3 work), use
   `git rebase --onto` to transplant only the feature commits onto `main`:
   ```bash
   ssh hpc_muc "cd <repo_root> && git rebase --onto origin/main <old-base> <feature-branch>"
   ```
   Example from the watchdog resolution:
   ```bash
   git rebase --onto origin/main 01e3b4a feat/training-watchdog
   ```

3. **Resolve conflicts one file at a time.** For `CHANGELOG.md` conflicts,
   prefer reconstructing the file from `origin/main` plus the feature's
   entries, because conflict markers may span hundreds of lines:
   ```bash
   git show origin/main:CHANGELOG.md > /tmp/main_changelog.md
   # ...extract the feature's changelog section(s)...
   cat <feature-sections> /tmp/main_changelog.md > CHANGELOG.md
   git add CHANGELOG.md
   git rebase --continue
   ```

4. **Never leave old watchdogs or daemon processes running.** A stale
   instance that survives a rebase/push can submit duplicate HPC jobs and
   waste GPU hours. After any branch cleanup that affects a running
   watchdog, kill the old PID, `bkill` any duplicate jobs it spawned, remove
   the stale `watchdog.pid`, and restart a single instance.

5. **Force-push carefully** only after the rebase succeeds and you have
   verified the branch history:
   ```bash
   ssh hpc_muc "cd <repo_root> && git push --force-with-lease origin <feature-branch>"
   ```

## Scratch / temporary files

Ad-hoc exploration, debugging, and one-off audit scripts should be created
under `scratch/` rather than at the repo root. Keep a short `scratch/README.md`
explaining that the scripts are reference-only and not part of the package.
This keeps the root directory clean and makes it easy to find/reuse prior
helper scripts.

## Completion Criteria

- Branch created with a `<type>/<summary>` name the agent derived itself
  (user wasn't asked to supply one, unless genuinely ambiguous).
- All commits are focused, logical units with CHANGELOG entries.
- Any HPC validation needed was actually run and its result is recorded in
  the MR's Validation section before merge.
- MR uses the repo's default template; self-review done before merge.
