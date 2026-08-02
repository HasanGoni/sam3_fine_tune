from pathlib import Path

base = Path("/home/hpc_storage/data/projects/userB/data/bent-wire/Bent-Wire-ALU-Test")
mask_suffixes = ("lblthick", "lblthin", "gatepads")

for d in sorted(base.iterdir()):
    if not d.is_dir() or d.name in ("Thin_multicat", "Thin_multicat_test"):
        continue
    for img in sorted(d.iterdir()):
        if not img.is_file() or "original" not in img.stem.lower():
            continue
        stem = img.stem.replace("original", "")
        missing = []
        for s in mask_suffixes:
            with_underscore = img.parent / f"{stem}_{s}{img.suffix}"
            without_underscore = img.parent / f"{stem}{s}{img.suffix}"
            if not (with_underscore.exists() or without_underscore.exists()):
                missing.append(s)
        if missing:
            print(f"MISSING {img}: {missing}")
