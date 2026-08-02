---
name: git-last-change-summary
description: 'Summarize latest git changes, update CHANGELOG.md, and draft a commit message. Use for release notes, handoff summaries, and pre-commit documentation.'
argument-hint: '[optional] scope or commit SHA; default analyzes working tree, fallback to latest commit'
user-invocable: true
disable-model-invocation: false
---
# Git Last Change Summary

Create a concise, traceable change summary from git, write it into CHANGELOG.md, and produce a high-quality commit message.

## When To Use

- You want a quick summary of what changed in the repository.
- You need to append a changelog entry before committing.
- You need a commit message that matches the actual diff.
- You are preparing a status update or handoff.

## Inputs

- Optional scope argument.
- If scope is omitted:

1. Analyze uncommitted changes first.
2. If working tree is clean, analyze the latest commit (`HEAD`).

## HPC note

If running on the Acme HPC, first verify SSH HPC connectivity using the
`ssh-hpc-connect` skill or by running `ssh hpc_muc echo connected`. Once that
works, discover the project directory dynamically and then run git commands
inside it:

```bash
ssh hpc_muc
REPO_ROOT=$(bash .github/scripts/discover_repo_root.sh sam3_fine_tune)
cd "${REPO_ROOT}"
# Optional: source .venv/bin/activate
```

The helper searches `/home/hpc_storage/data/projects/$USER` for a directory matching the project name (default `sam3_fine_tune`) and falls back to the first directory containing `pyproject.toml`.

## Procedure

1. **Determine source of truth.**
   - Run `git status --short`.
   - If there are modified files, collect staged and unstaged diffs.
   - If no local changes, inspect latest commit with `git show --name-status --stat --oneline -1`.

2. **Build structured summary.**
   - Group changes by file and intent:
     1. Added
     2. Updated
     3. Removed
     4. Refactor/cleanup
   - Extract key behavioral impact:
     1. Feature behavior changes
     2. Pipeline/inference changes
     3. Config/data path changes
     4. Docs/tests updates

3. **Update CHANGELOG.md.**
   - Create the file if it does not exist.
   - **Newest entry always goes at the top.** Insert the new dated heading
     (`## YYYY-MM-DD`) immediately after the file's title/intro, directly
     *before* the first existing `## YYYY-MM-DD` heading — never append at
     the end of the file. The file must read reverse-chronological (most
     recent date first). If today's date section already exists (same-day
     follow-up run), append bullets to that existing section instead of
     creating a duplicate heading.
   - Add a short context line, then flat bullet points describing concrete changes.
   - Keep wording factual and tied to files/diff.

4. **Draft commit message.**
   - Subject line format: `<type>(<scope>): <summary>`.
   - Candidate types: `feat`, `fix`, `refactor`, `docs`, `chore`.
   - Add body bullets for major points when change set spans multiple files.
   - Include one explicit line for user-facing or pipeline-impacting behavior.
   - Always provide a terminal-ready copy/paste command using repeated `-m` flags, for example:
     `git commit -m "<type>(<scope>): <summary>" -m "<bullet 1>" -m "<bullet 2>" -m "Pipeline impact: <impact>."`

5. **Final validation.**
   - Ensure summary matches actual changed files.
   - Ensure CHANGELOG.md was appended, not rewritten.
   - Run `head -5 CHANGELOG.md` (or equivalent) and confirm the topmost dated
     heading is today's date — if an older date appears above it, the entry
     was inserted in the wrong place; fix before committing.
   - Ensure commit message maps to strongest change intent.
   - Ensure no sensitive data appears in changelog or commit text.

## Decision Logic

- If code + docs changed, prioritize code intent in commit type and list docs in body.
- If only docs changed, use `docs` type.
- If mostly renaming/restructure without behavior changes, use `refactor` or `chore`.
- If thresholds/configs alter runtime outputs, treat as behavior-impacting and call it out.

## Completion Criteria

- Changelog entry exists for current run date with concrete bullets.
- Changelog reads reverse-chronological: today's dated heading is the topmost
  `## YYYY-MM-DD` section in the file.
- Summary captures all primary changed files.
- Commit message subject is one line and under 72 chars.
- Commit body (if present) is concise and consistent with diff.

## Output Format

1. Change summary (3-8 bullets)
2. Changelog text added
3. Suggested commit message
4. Copy-paste terminal commit command (`git commit -m ... -m ...`)
