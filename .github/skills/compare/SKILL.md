---
name: compare
description: 'Score base SAM3, fine-tuned SAM3 and UNVEIL models on the test split of a prepared root with `sam3-compare` and print one table plus a JSON file. Use when you need the final SAM3 versus UNVEIL numbers, a quick seeded subset check, or to read a table with n/a, not trained and suspect cells.'
argument-hint: '[root]'
user-invocable: true
disable-model-invocation: false
---
# Compare models (`sam3-compare`)

One metric for every model: pooled union-mask IoU per category at the threshold, over test images annotated for that
category. Instance AP is not computed here (that stays in `sam3-finetune`'s evaluation).

## Where things live

- Options, the model-entry kinds and an example: `sam3_fine_tune/compare/conf/compare.yaml` (Hydra). `--help` only prints
  the Hydra group list; read the YAML.
- Code: `sam3_fine_tune/compare/` (`ops.py` scoring and table). Input: `ROOT/prepared/sam3/test` from `sam3-prepare`.

## Rules

1. An image flagged `not_annotated_category_ids` for a category is skipped for it, never a negative or a positive. This is
   why `sam3-prepare` must have the recipe's `families:` filled in first.
2. Read the table, not just the means: the image count per category (a 4-image row is noise), `n/a` (no test images, not
   zero), `not trained` (listed outside the entry's `trained_categories`, not a weakness) and `suspect` (below
   `suspect_below`: check prompt spelling and threshold before trusting it; `thin_wire_panda` was 0.000 in #18 and is
   not explained yet).
3. A fine-tuned SAM3 prompt uses the trained spelling ("thin wire panda" for `thin_wire_panda`); set `trained_categories`
   from the run's own metadata.
4. **Checkpoint trust.** The trainers' `checkpoint.pt` holds optimizer state, so loading it needs `trust_checkpoint=true`,
   which unpickles and can run arbitrary code. Use it only for checkpoints you trained yourself. UNVEIL's `.safetensors`
   holds no pickled objects and needs no flag.
5. **Gate.** Each UNVEIL column carries `gate: true|false`; the table prints the setting per column. Put a gated and an
   ungated column of the same checkpoint side by side rather than choosing one.
6. UNVEIL kinds need a CUDA GPU and the `unveil` package. `sam3_checkpoint` and `bpe_path` are needed by the SAM3 kinds.
7. `subset=N seed=S` for a quick reproducible check; the final number is the full split. `test` is touched only here.

## Not tested

The `unveil` and `unveil-sam3-path` columns have only run against stubs; no real UNVEIL column has been scored.
