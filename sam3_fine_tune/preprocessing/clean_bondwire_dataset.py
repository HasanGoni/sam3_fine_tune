"""Build a COCO dataset from clean separated bondwire labels.

Input convention (from Bent-Wire-ALU-Train/20250512_parted_thin_flat):
- *_original.png  -> image
- *_lblthin.png   -> thin wire mask
- *_lblthick.png  -> thick wire mask
- *_gatepads.png  -> gate pad mask
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from sam3_fine_tune.preprocessing.coco_format import form_coco_dataset, save_ann_file

CATEGORY_CONFIG = {
    "thin_wire": {"suffix": "_lblthin", "category_id": 1},
    "thick_wire": {"suffix": "_lblthick", "category_id": 1},
    "gate_pad": {"suffix": "_gatepads", "category_id": 1},
}

IMAGE_SUFFIX = "_original"


def _merge_datasets_multi(datasets: list, output_path: str):
    """Merge multiple COCO datasets into one multi-class dataset."""
    for split in ["train", "test"]:
        out_split = Path(output_path) / split
        out_split.mkdir(parents=True, exist_ok=True)
        (out_split / "images").mkdir(exist_ok=True)

        categories = [
            {"id": i + 1, "name": ds["category_name"]} for i, ds in enumerate(datasets)
        ]
        merged = {
            "info": {"description": "Clean bondwire dataset", "version": "1.0", "year": 2025},
            "images": [],
            "categories": categories,
            "annotations": [],
        }

        img_id_counter = 1
        ann_id_counter = 1

        for ds_idx, ds_config in enumerate(datasets):
            ds_path = Path(ds_config["path"])
            prefix = ds_config["prefix"]
            category_id = ds_idx + 1

            ann_path = ds_path / split / "annotation.json"
            if not ann_path.exists():
                print(f"  Skipping {split} - {ds_config['category_name']}: no annotation.json")
                continue

            with open(ann_path) as f:
                data = json.load(f)

            old_to_new_img = {}
            for img in data["images"]:
                old_id = img["id"]
                new_img = {
                    "id": img_id_counter,
                    "file_name": f"{prefix}_{img['file_name']}",
                    "width": img["width"],
                    "height": img["height"],
                }
                merged["images"].append(new_img)
                old_to_new_img[old_id] = img_id_counter
                img_id_counter += 1

                src = ds_path / split / "images" / img["file_name"]
                dst = out_split / "images" / f"{prefix}_{img['file_name']}"
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    print(f"  Warning: Missing {src}")

            for ann in data["annotations"]:
                merged["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": old_to_new_img[ann["image_id"]],
                    "category_id": category_id,
                    "bbox": ann["bbox"],
                    "segmentation": ann.get("segmentation", []),
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                })
                ann_id_counter += 1

            print(f"  {split} - {ds_config['category_name']}: {len(data['images'])} images")

        out_ann = out_split / "annotation.json"
        with open(out_ann, "w") as f:
            json.dump(merged, f, indent=2)

        print(
            f"  {split} TOTAL: {len(merged['images'])} images, "
            f"{len(merged['annotations'])} annotations"
        )
        print(f"  Categories: {[c['name'] for c in categories]}")


def discover_samples(src_dir: Path | str) -> dict[str, dict[str, Path]]:
    """Return {sample_stem: {'image': path, 'thin_wire': path, ...}}."""
    src_dir = Path(src_dir)
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    for p in sorted(src_dir.glob("*.png")):
        name = p.name
        if name.endswith(f"{IMAGE_SUFFIX}.png"):
            stem = name[: -len(f"{IMAGE_SUFFIX}.png")]
            groups[stem]["image"] = p
        else:
            for cat, cfg in CATEGORY_CONFIG.items():
                suffix = cfg["suffix"]
                if name.endswith(f"{suffix}.png"):
                    stem = name[: -len(f"{suffix}.png")]
                    groups[stem][cat] = p
                    break

    complete = {}
    required = {"image"} | set(CATEGORY_CONFIG)
    for stem, files in groups.items():
        if required.issubset(files):
            complete[stem] = dict(files)
        else:
            missing = required - set(files)
            print(f"Skipping incomplete sample {stem}, missing {missing}")

    return complete


def stage_clean_dataset(
    src_dir: Path | str,
    output_root: Path | str,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Stage the clean separated bondwire data into per-category train/test dirs.
    Returns the list of train stems and test stems.
    """
    src_dir = Path(src_dir)
    output_root = Path(output_root)
    samples = discover_samples(src_dir)
    stems = sorted(samples.keys())

    train_stems, test_stems = train_test_split(
        stems, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"Total complete samples: {len(stems)}")
    print(f"Train: {len(train_stems)}, Test: {len(test_stems)}")

    for split, split_stems in [("train", train_stems), ("test", test_stems)]:
        for stem in tqdm(split_stems, desc=f"Staging {split}"):
            files = samples[stem]
            for cat in CATEGORY_CONFIG:
                img_dir = output_root / cat / split / "images"
                msk_dir = output_root / cat / split / "masks"
                img_dir.mkdir(parents=True, exist_ok=True)
                msk_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(files["image"], img_dir / f"{stem}.png")
                shutil.copy2(files[cat], msk_dir / f"{stem}.png")

    return train_stems, test_stems


def convert_to_coco(
    staging_root: Path | str,
    categories: list[str] | None = None,
    splits: list[str] = ["train", "test"],
) -> dict[str, dict[str, Path]]:
    """
    Run coco_format.py form_coco_dataset for every category + split.
    Returns {category: {split: annotation_path}}.
    """
    staging_root = Path(staging_root)
    categories = categories or list(CATEGORY_CONFIG.keys())
    paths: dict[str, dict[str, Path]] = {}

    for cat in categories:
        paths[cat] = {}
        for split in splits:
            img_dir = staging_root / cat / split / "images"
            msk_dir = staging_root / cat / split / "masks"
            out_dir = staging_root / cat / split

            coco = form_coco_dataset(
                mask_dir=msk_dir,
                img_dir=img_dir,
                categories=[{"id": 1, "name": cat}],
            )
            save_ann_file(out_dir, "annotation", coco)
            paths[cat][split] = out_dir / "annotation.json"
            print(f"{cat}/{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    return paths


def merge_clean_bondwire(
    staging_root: Path | str,
    output_path: Path | str,
    prefixes: dict[str, str] | None = None,
):
    """Merge the three clean bondwire categories into one multi-class COCO dataset."""
    prefixes = prefixes or {"thick_wire": "g", "thin_wire": "g", "gate_pad": "g"}
    datasets = [
        {"path": str(Path(staging_root) / cat), "category_name": cat, "prefix": prefixes[cat]}
        for cat in ["thick_wire", "thin_wire", "gate_pad"]
    ]
    _merge_datasets_multi(datasets, str(output_path))


def patch_into_existing(
    existing_dataset: Path | str,
    clean_staging: Path | str,
    output_dataset: Path | str,
):
    """Replace bondwire annotations in an existing merged dataset with clean ones."""
    import subprocess

    cmd = [
        "python",
        "-m",
        "sam3_fine_tune.preprocessing.patch_bondwire_annotations",
        "--src_dataset", str(existing_dataset),
        "--dst_dataset", str(output_dataset),
        "--bridged_staging", str(clean_staging),
        "--bondwire_categories", "thick_wire,thin_wire,gate_pad",
        "--bondwire_cat_ids", "6,7,8",
        "--src_prefix", "g_",
        "--bridged_prefixes", "g_,g_,g_",
    ]
    subprocess.run(cmd, check=True)
