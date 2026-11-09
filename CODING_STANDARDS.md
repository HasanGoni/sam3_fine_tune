# Coding standards

Read during review, not implementation. Mechanical rules are enforced by checks, not prose:
`tests/test_repo_standards.py` (CI + pre-commit) and `ufmt` (black + usort, line length 88). This file holds only the
judgement calls. Use the terms in `GLOSSARY.md`.

## Match the sibling

- A new CLI mirrors `sam3_fine_tune/dataset/` (see "Adding a Hydra CLI" in `CLAUDE.md`): Hydra config with `???` for
  required keys, `hydra.run.dir: .` so no output folders appear, a thin `cli.py` over a plain-function module.
- A new test folder mirrors an existing one in `tests/` and builds its data in fixtures; no test needs real weights, a GPU
  or the network.

## Seams

- Anything that calls a model takes the call as an injectable `predict` function (or a small `Predictor` of `encode`/`segment` callables when the image should be encoded once and reused, as in `sam3-infer`), so output logic is tested with a
  fake. Loading weights lives in one `make_*predictor` function per module.
- Validate user input before loading a model. Errors name the offending key and what to change; the CLI turns them into
  one `SystemExit` line, not a traceback.

## Don't change training behaviour by accident

- `sam3/` is vendored. Prefer wrappers in `sam3_fine_tune/`; when `sam3/` must change, keep the diff small and update the
  reference copy under `sam3_fine_tune/modified_script/` if the file has one.
- `scratch.resolution` stays 1008; `category_chunk_size` is a memory lever (see `CLAUDE.md`); do not edit
  `decoder_train_config.yaml`.
- Output files overwrite by name; never delete or rewrite user data in an output folder implicitly.

## Docs and claims

- A feature is documented in `README.md` in the same change, next to the training and evaluation shortcuts.
- A PR or issue comment says what was verified and what was not (for example: no real `sam3.pt` run, no GPU run).
- Record the reason for a surprising choice in the code or an ADR (`docs/adr/`), not in the commit message alone.
