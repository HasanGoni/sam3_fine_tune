#!/usr/bin/env python3
"""Append more images into an EXISTING category of a merged COCO dataset.

Standalone script version of the `append_images_to_existing_category()` cell
in nbs/02_merge_dataset.ipynb, so it can be submitted as a normal Python job
to the HPC CPU queue via bsub (see run_append_images_to_existing_category.sh)
instead of running image copies synchronously in the login-node ssh session.

Usage:
    python append_images_to_existing_category.py <nb_path> <new_dataset_path> <category_name> <prefix> <existing_output>
"""

import json
import sys


def load_append_images_to_existing_category(nb_path: str):
    """Load only the `#| export`-tagged cells of nbs/02_merge_dataset.ipynb and
    exec them in-memory to get `append_images_to_existing_category` without
    running the real nbdev-export CLI (this repo's nbdev config is still
    settings.ini, not migrated to pyproject.toml)."""
    with open(nb_path) as f:
        nb = json.load(f)

    src = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        text = "".join(lines) if isinstance(lines, list) else lines
        if text.lstrip().startswith("#| export") or text.lstrip().startswith("#|export"):
            src.append(text)

    ns = {}
    exec(compile("\n".join(src), nb_path, "exec"), ns)
    return ns["append_images_to_existing_category"]


def main():
    nb_path, new_path, category_name, prefix, existing_output = sys.argv[1:6]

    append_images_to_existing_category = load_append_images_to_existing_category(nb_path)

    append_images_to_existing_category(
        {"path": new_path, "category_name": category_name, "prefix": prefix},
        existing_output,
    )

    print("")
    print("=== append_images_to_existing_category completed ===")
    for split in ("train", "test"):
        ann_path = f"{existing_output}/{split}/annotation.json"
        try:
            with open(ann_path) as f:
                d = json.load(f)
            print(
                f"{split}: {len(d['images'])} images, {len(d['annotations'])} annotations, "
                f"categories={[c['name'] for c in d['categories']]}"
            )
        except FileNotFoundError:
            print(f"{split}: annotation.json not found (skipped)")


if __name__ == "__main__":
    main()
