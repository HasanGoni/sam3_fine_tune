"""Generate a grid poster of the converted thin wire panda training images.

Reads the cleaning record, builds a temporary COCO JSON containing only the
images that received converted predictions, and renders a grid poster with the
new annotations overlaid.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-annotation", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--n-cols", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()

    cleaned = json.loads(Path(args.cleaned_annotation).read_text())
    record = json.loads(Path(args.record).read_text())

    convert_ids = set(record["no_gt_converted_image_ids"])
    if not convert_ids:
        raise ValueError("No converted images found in record")

    # Keep only converted images and their converted annotations
    images = [im for im in cleaned["images"] if im["id"] in convert_ids]
    annotations = [
        ann for ann in cleaned["annotations"]
        if ann["image_id"] in convert_ids
        and ann.get("source") == "converted_from_model_prediction_train"
    ]

    subset = {
        "info": cleaned.get("info", {}),
        "licenses": cleaned.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": cleaned["categories"],
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(subset, f)
        subset_path = f.name

    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "save_coco_grid_poster.py"),
            "--coco-json", subset_path,
            "--image-dir", args.images,
            "--categories", "thin wire panda",
            "--output", args.out,
            "--panel-size", str(args.panel_size),
            "--n-cols", str(args.n_cols),
            "--alpha", str(args.alpha),
            "--dpi", str(args.dpi),
            "--jpeg-quality", str(args.jpeg_quality),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        Path(subset_path).unlink(missing_ok=True)

    print(f"Saved converted training poster to {args.out}")


if __name__ == "__main__":
    main()
