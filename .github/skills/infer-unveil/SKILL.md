---
name: infer-unveil
description: 'Run UNVEIL on images with text prompts using `sam3-infer-unveil`: same image, prompt, output and device arguments as `sam3-infer`, same output layout, plus the presence gate and UNVEIL''s frozen SAM3 path. Use when you need masks or overlays from an UNVEIL checkpoint, or the concept map without the presence check.'
argument-hint: '[images-or-coco-split]'
user-invocable: true
disable-model-invocation: false
---
# Run UNVEIL (`sam3-infer-unveil`)

Mirror of `sam3-infer` (`.github/skills/sam3-infer/SKILL.md`): swap the command, keep the arguments. It writes `masks/`,
`overlays/` and `summary.json` in the same layout, so downstream scripts work for both.

## Where things live

- Options, defaults and examples: `sam3_fine_tune/infer_unveil/conf/infer_unveil.yaml`. `--help` only prints the Hydra
  group list; read the YAML.
- Code: `sam3_fine_tune/infer_unveil/`.
- `sam3_checkpoint` and `bpe_path` are required (UNVEIL builds on base SAM3); local copies are in CLAUDE.md "Local
  weights and data". `checkpoint` is UNVEIL's own `.safetensors` (from `sam3-train-unveil`); omit it for UNVEIL's default.

## Rules

1. Needs the `unveil` package and a CUDA GPU. It fails on CPU.
2. **The gate changes the numbers a lot.** `gate=true` (default) empties a mask when the presence probability is below
   the prompt's threshold; `gate=false` is the raw concept map. Always say which one a result used; never mix them.
   `path=sam3` is UNVEIL's frozen SAM3 semantic path (base-model baseline) and has no gate.
3. The prompt is the category name used in training, exactly as typed.
4. UNVEIL gives a concept map, not scored instances: per-prompt `threshold` works, a threshold sweep does not.
5. For the final number use `sam3-compare` on `test` (`.github/skills/compare/SKILL.md`), not this CLI's own scoring.

## Not tested

Only a stub model has run in the tests. The real UNVEIL adapter has never run through this CLI.
