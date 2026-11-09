# Glossary

Domain vocabulary for this repo. Use these terms in code, issues and reviews.

**Category**: one kind of object to segment (for example "bond wire", "pin", "crack", "chip"). Its `category_name` in the
COCO annotation is the **prompt** and must match exactly at train and inference time.

**Prompt**: the text string given to SAM3 to say what to segment. In this repo it is a category name. Box prompts exist
in SAM3 but are not used yet (issue #17).

**Instance**: one detected object for a prompt. A prediction for an image is zero or more instances, each with a mask and
a score. The saved **combined mask** is the union of all instances.

**Score / threshold**: SAM3's confidence per instance, and the cut-off below which an instance is dropped (an instance is kept when its score is `>` the threshold).

**Sweep**: a prompt with a list of thresholds. The model runs once at the lowest; higher ones only re-filter by score. Each threshold gets a `thr_<t>/` output folder.

**Base model**: SAM3 with the released `sam3.pt` weights, no fine-tuning. **Fine-tuned model**: a **checkpoint** produced
by this repo's training. Both need the tokenizer vocabulary (`bpe_simple_vocab_16e6.txt.gz`).

**Checkpoint**: a `.pt` file. A trainer checkpoint is a dict with `"model"` (the weights) plus optimizer and counters.
The **best checkpoint** is the one with the highest validation segmentation AP.

**Full fine-tuning / decoder-only**: which parts of SAM3 are trained. Full fine-tuning (the default,
`trainer.optim.train_full_model`) also trains the vision backbone, text encoder and cross-attention encoder.

**Dataset root**: `<root>/{train,test}/{images,annotation.json}`, COCO polygons. The **merged annotation**
(`annotation_merged.json`, a copy of `annotation.json` when none exists) is what the training configs read.

**Extend vs new**: adding images of a category to a dataset where it already exists, versus creating the category.

**Test split**: the held-out images used by `sam3-finetune` evaluation; a share of each added category goes there.

**Category chunk size**: how many categories' queries are grouped per training step; a GPU-memory lever.

**Foreground IoU**: intersection over union of predicted and true foreground pixels, computed per category and overall.
Precision, recall and F1 are computed from the same pixel counts.

**Experiment dir**: `experiments/<run>/` holding checkpoints, tensorboard logs, `config_used.yaml`, `eval/`, and on a crash
`dumps/anomalous_batches/`.

**Anomalous batch**: a batch dumped when the loss goes NaN or exactly zero, for inspection.

**SAMBA-UNet**: the baseline segmentation model in `sam3_fine_tune/sambaunet/`, evaluated with the same metrics so
results compare key-by-key with SAM3.

**HPC**: the Acme cluster reached with `ssh hpc_muc`, where training and evaluation run and the git remote lives; this
workspace is a staging copy. **Runbook skills**: the `.github/skills/*/SKILL.md` files describing HPC procedures.

**Console scripts**: `sam3-dataset` (build or extend a dataset), `sam3-finetune` (train then evaluate), `sam3-infer`
(run base or fine-tuned SAM3 with one or several text prompts, each with one threshold or a sweep of thresholds, over a folder).
