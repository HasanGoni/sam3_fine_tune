"""Build a category-rebalanced COCO dataset using DINOv3-embedding diversity selection.

For one or more "fixed" categories, images are copied through unchanged
(existing train/test split preserved). For one or more "diverse" categories,
a target number of images is selected from a larger source pool using
farthest-point sampling over DINOv3 embeddings (maximizing visual
diversity), then split into train/test.

Output category names can be overridden per source category name via the
optional `category_display_names` config map (e.g. to rename an
underscore-joined source name like `thick_wire_panda` to a natural-language
prompt string like `thick wire panda` in the final COCO categories list,
without affecting any internal source-category matching).

To avoid the "split-photo contradictory negatives" bug documented in this
repo's HPC memory (an image selected for one category silently asserting a
false negative for another target category it also contains), every selected
image keeps ALL of its annotations that belong to ANY of the target
categories, not just the category that caused it to be selected.

Usage:
    python build_diverse_wire_dataset.py --config configs/diverse_wire_dataset_example.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from pycocotools import mask as maskUtils
from skimage import measure
from tqdm.auto import tqdm

from coco_format import save_metadata_file
from diversity.dinov3_embedder import DEFAULT_CHECKPOINT, DEFAULT_MODEL_NAME, DINOv3Embedder
from diversity.embedding_cache import EmbeddingCache
from diversity.select_diverse import farthest_point_sample

DEFAULT_EMBEDDING_CACHE = (
    "/home/hpc_storage/data/projects/userB/sam3_data/_diverse_embedding_cache/"
    "{model_name}_cache.npz"
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split_annotations(root: Path, split: str, target_category_names: set):
    """Load one split's COCO json and return (image_id -> record) filtered to
    only images that have >=1 annotation in target_category_names, each
    record carrying only the annotations that belong to target_category_names.
    """
    ann_path = root / split / "annotation_merged.json"
    if not ann_path.exists():
        ann_path = root / split / "annotation.json"
    coco = json.loads(ann_path.read_text())

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        cname = cat_id_to_name.get(ann["category_id"])
        if cname not in target_category_names:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append((cname, ann))

    records = {}
    for img_id, anns in anns_by_img.items():
        img = img_by_id[img_id]
        records[img_id] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "image_dir": root / split / "images",
            "anns": anns,  # list of (category_name, original_ann_dict)
        }
    return records


def collect_candidates(root: Path, splits: list, target_category_names: set):
    """Merge records for all splits of one dataset root into one dict keyed by
    a synthetic unique key (split, image_id)."""
    merged = {}
    for split in splits:
        recs = load_split_annotations(root, split, target_category_names)
        for img_id, rec in recs.items():
            merged[(split, img_id)] = rec
    return merged


def category_has(rec, name):
    return any(cname == name for cname, _ in rec["anns"])


def compute_category_mask_stats(rec, name):
    """Return (total_area, coverage, max_component_area) for category `name`."""
    h, w = rec["height"], rec["width"]
    total = np.zeros((h, w), dtype=np.uint8)
    for cname, ann in rec["anns"]:
        if cname != name:
            continue
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            m = maskUtils.decode(seg)
        else:
            m = maskUtils.decode(maskUtils.frPyObjects(seg, h, w))
        if m.ndim == 3:
            m = m.max(axis=2)
        total = np.maximum(total, m)
    total_area = int(total.sum())
    img_area = h * w
    coverage = total_area / img_area if img_area > 0 else 0.0
    labels = measure.label(total, connectivity=2)
    regions = measure.regionprops(labels)
    max_component_area = max((int(r.area) for r in regions), default=0)
    return total_area, coverage, max_component_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # `fixed_categories` (list) is the current schema; also accept the older
    # singular `fixed_category` (dict) for backward compatibility.
    if "fixed_categories" in cfg:
        fixed_cfgs = cfg["fixed_categories"]
    else:
        fixed_cfgs = [cfg["fixed_category"]]
    diverse_cfgs = cfg["diverse_categories"]
    category_order = cfg["category_order"]
    display_names = cfg.get("category_display_names", {})
    train_ratio = float(cfg["split"].get("train_ratio", 0.8))
    seed = int(cfg["split"].get("seed", 42))
    output_root = Path(cfg["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_root / "diverse_wire_dataset_config.yaml")

    all_target_names = {fc["name"] for fc in fixed_cfgs} | {d["name"] for d in diverse_cfgs}
    rng = np.random.default_rng(seed)

    used_keys: set = set()  # (split_source_tag, root, split, image_id) already consumed

    # ------------------------------------------------------------------
    # 1) Fixed categories: keep exactly as-is (existing train/test split).
    #    Grouped by source root so a shared root's annotation file is only
    #    loaded/parsed once even if it backs several fixed categories.
    # ------------------------------------------------------------------
    output_assignment: dict[str, list] = {"train": [], "test": []}

    root_to_names: dict[tuple, list] = {}
    for fc in fixed_cfgs:
        root_key = (str(fc["source_root"]), tuple(fc.get("splits", ["train", "test"])))
        root_to_names.setdefault(root_key, []).append(fc["name"])

    for (root_str, fsplits), names_in_root in root_to_names.items():
        froot = Path(root_str)
        fixed_candidates = collect_candidates(froot, list(fsplits), all_target_names)
        counts = {n: {"train": 0, "test": 0} for n in names_in_root}
        for (split, img_id), rec in fixed_candidates.items():
            full_key = ("fixed", root_str, split, img_id)
            if full_key in used_keys:
                continue
            matched = [n for n in names_in_root if category_has(rec, n)]
            if not matched:
                continue
            rec = dict(rec)
            rec["_source_tag"] = "fixed"
            output_assignment[split].append(rec)
            used_keys.add(full_key)
            for n in matched:
                counts[n][split] += 1

        for n in names_in_root:
            print(
                f"[fixed:{n}] train={counts[n]['train']}, "
                f"test={counts[n]['test']} (unchanged)"
            )

    # ------------------------------------------------------------------
    # 2) Diverse categories: embed candidate pool, farthest-point sample,
    #    then split train/test.
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", DEFAULT_MODEL_NAME)
    embedder = None
    cache_path = model_cfg.get(
        "embedding_cache", DEFAULT_EMBEDDING_CACHE.format(model_name=model_name)
    )
    embedding_cache = EmbeddingCache(cache_path)
    print(f"[cache] {cache_path}: {len(embedding_cache)} embeddings loaded")
    manifest = {
        "fixed_categories": [fc["name"] for fc in fixed_cfgs],
        "diverse_categories": [],
    }

    source_cfg = cfg["source"]
    source_root = Path(source_cfg["root"])
    source_splits = source_cfg.get("splits", ["train", "test"])
    source_candidates = collect_candidates(source_root, source_splits, all_target_names)

    used_file_names: set = set()
    for rec in output_assignment["train"] + output_assignment["test"]:
        used_file_names.add(rec["file_name"])

    for d_cfg in diverse_cfgs:
        name = d_cfg["name"]
        target_count = int(d_cfg["target_count"])

        min_total_area = d_cfg.get("min_total_area")
        min_coverage = d_cfg.get("min_coverage")
        min_max_component_area = d_cfg.get("min_max_component_area")

        def passes_mask_filter(rec, name):
            if min_total_area is None and min_coverage is None and min_max_component_area is None:
                return True
            total_area, coverage, max_comp = compute_category_mask_stats(rec, name)
            if min_total_area is not None and total_area < min_total_area:
                return False
            if min_coverage is not None and coverage < min_coverage:
                return False
            if min_max_component_area is not None and max_comp < min_max_component_area:
                return False
            return True

        raw_pool_items = [
            (key, rec)
            for key, rec in source_candidates.items()
            if category_has(rec, name) and rec["file_name"] not in used_file_names
        ]
        pool_items = [
            (key, rec) for key, rec in raw_pool_items if passes_mask_filter(rec, name)
        ]
        pool_size = len(pool_items)
        print(
            f"[diverse:{name}] raw pool={len(raw_pool_items)}, "
            f"after mask filter={pool_size}, target={target_count}"
        )

        if pool_size <= target_count:
            selected = pool_items
        else:
            image_paths = [str((rec["image_dir"] / rec["file_name"]).resolve()) for _, rec in pool_items]
            cached = embedding_cache.get_many(image_paths)
            missing_paths = [p for p in image_paths if p not in cached]
            print(
                f"[diverse:{name}] embeddings cached={len(cached)}, "
                f"to_compute={len(missing_paths)}"
            )

            if missing_paths:
                if embedder is None:
                    embedder = DINOv3Embedder(
                        model_name=model_name,
                        checkpoint_path=model_cfg.get("checkpoint", DEFAULT_CHECKPOINT),
                    )
                    print(f"Loaded embedder on device={embedder.device}, dim={embedder.embed_dim}")

                batch_size = int(model_cfg.get("batch_size", 64))
                save_every = int(model_cfg.get("cache_save_every_batches", 20))
                new_embeds = {}
                for i, start in enumerate(
                    tqdm(range(0, len(missing_paths), batch_size), desc=f"Embedding {name}"),
                    start=1,
                ):
                    chunk = missing_paths[start : start + batch_size]
                    chunk_embeds = embedder.embed_batch(chunk, batch_size=batch_size)
                    for p, e in zip(chunk, chunk_embeds):
                        new_embeds[p] = e
                    if i % save_every == 0:
                        embedding_cache.update(new_embeds)
                        embedding_cache.save()
                        new_embeds = {}
                embedding_cache.update(new_embeds)
                embedding_cache.save()

            cached_all = embedding_cache.get_many(image_paths)
            embeds = np.stack([cached_all[p] for p in image_paths])

            picked_idx = farthest_point_sample(embeds, target_count, seed=seed)
            selected = [pool_items[i] for i in picked_idx]

        for key, rec in selected:
            used_file_names.add(rec["file_name"])

        # Shuffle then split train/test. If `test_count` is set for this
        # category, use that exact absolute test count (remainder to train)
        # instead of the global train_ratio -- useful for topping up just the
        # training set of one category without touching its test count.
        order = rng.permutation(len(selected))
        fixed_test_count = d_cfg.get("test_count")
        if fixed_test_count is not None:
            n_test = min(int(fixed_test_count), len(selected))
            test_idx = set(order[:n_test].tolist())
        else:
            n_train = int(round(len(selected) * train_ratio))
            train_idx = set(order[:n_train].tolist())

        n_train_actual = 0
        n_test_actual = 0
        for i, (key, rec) in enumerate(selected):
            rec = dict(rec)
            rec["_source_tag"] = "diverse"
            if fixed_test_count is not None:
                split_name = "test" if i in test_idx else "train"
            else:
                split_name = "train" if i in train_idx else "test"
            output_assignment[split_name].append(rec)
            if split_name == "train":
                n_train_actual += 1
            else:
                n_test_actual += 1

        print(f"[diverse:{name}] selected={len(selected)} -> train={n_train_actual}, test={n_test_actual}")
        manifest["diverse_categories"].append(
            {
                "name": name,
                "pool_size": pool_size,
                "target_count": target_count,
                "selected_count": len(selected),
                "train_count": n_train_actual,
                "test_count": n_test_actual,
                "used_farthest_point_sampling": pool_size > target_count,
            }
        )

    if embedder is not None:
        manifest["embedder"] = {
            "model_name": model_cfg.get("name", DEFAULT_MODEL_NAME),
            "checkpoint": str(model_cfg.get("checkpoint", DEFAULT_CHECKPOINT)),
        }

    # ------------------------------------------------------------------
    # 3) Assemble final COCO dataset per split with the requested category
    #    order and copy image files.
    # ------------------------------------------------------------------
    categories = [
        {"id": i + 1, "name": display_names.get(n, n)} for i, n in enumerate(category_order)
    ]
    name_to_new_id = {n: c["id"] for n, c in zip(category_order, categories)}

    for split in ["train", "test"]:
        dst_img_dir = output_root / split / "images"
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        new_images = []
        new_annotations = []
        next_ann_id = 1

        for new_img_id, rec in enumerate(
            tqdm(output_assignment[split], desc=f"Assembling {split}"), start=1
        ):
            src_path = rec["image_dir"] / rec["file_name"]
            ext = Path(rec["file_name"]).suffix
            out_name = f"{new_img_id:06d}_{Path(rec['file_name']).stem}{ext}"
            shutil.copy2(src_path, dst_img_dir / out_name)

            new_images.append(
                {
                    "id": new_img_id,
                    "file_name": out_name,
                    "width": rec["width"],
                    "height": rec["height"],
                }
            )

            for cname, ann in rec["anns"]:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = new_img_id
                new_ann["category_id"] = name_to_new_id[cname]
                new_annotations.append(new_ann)
                next_ann_id += 1

        new_coco = {
            "info": {"description": "Diversity-rebalanced wire dataset", "version": "1.0"},
            "images": new_images,
            "annotations": new_annotations,
            "categories": categories,
        }

        out_ann = output_root / split / "annotation_merged.json"
        with open(out_ann, "w") as f:
            json.dump(new_coco, f, indent=2)
        save_metadata_file(output_root / split, "annotation_merged", new_coco)

        print(
            f"[{split}] images: {len(new_images)}, annotations: {len(new_annotations)}, "
            f"categories: {[c['name'] for c in categories]}"
        )

    with open(output_root / "selection_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDiverse wire dataset ready at: {output_root}")


if __name__ == "__main__":
    main()
