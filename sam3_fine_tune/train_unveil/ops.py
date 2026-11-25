"""Validation of the UNVEIL recipe against the prepared UNVEIL view (rules: docs/unveil-recipe.md).

Every problem is collected, and each message names the fix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

import yaml

KINDS = ("region", "balanced")
PAPER_IMAGES_PER_STEP = 24  # recipe rule 2


class RecipeError(ValueError):
    """The recipe or the prepared view breaks a rule; ``args[0]`` lists every problem."""


def epochs_problem(value: object) -> str | None:
    """Why ``value`` is not a usable number of epochs (a finite number above 0), or ``None`` when it is."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        return f"epochs is {value!r}; use a number above 0 (`epochs=E` or `epochs:` in recipe.yaml), or remove it"
    return None


def unsafe_path(value: object) -> bool:
    """True for anything that is not a relative path inside its folder (absolute, ``..``, empty or not a string)."""
    if not isinstance(value, str) or not value:
        return True
    p = PurePosixPath(value)
    return p.is_absolute() or ".." in p.parts or value.startswith("\\")


def safe_join(root: str | Path, value: object) -> Path:
    """``root / value`` for a path read from a data file; ``ValueError`` when it would leave ``root``."""
    if unsafe_path(value):
        raise ValueError(
            f"path {value!r} in an index file is absolute or contains `..`; "
            "run `sam3-prepare` again to rebuild the view"
        )
    return Path(root) / str(value)


def record_problems(where: str, records: list) -> list[str]:
    """Malformed index records (a missing key, a path leaving the root), each naming the file and the entry."""
    problems = []
    for n, r in enumerate(records, 1):
        if not isinstance(r, dict):
            problems.append(
                f"{where} entry {n} is not an object; run sam3-prepare again"
            )
            continue
        missing = [k for k in ("id", "image", "gt", "categories") if k not in r]
        if missing:
            problems.append(
                f"{where} entry {n} lacks key(s) {missing}; run sam3-prepare again to rebuild the view"
            )
        for k in ("image", "gt"):
            if k in r and unsafe_path(r[k]):
                problems.append(
                    f"{where} entry {n} has {k} {r[k]!r}, which is absolute or contains `..`; "
                    "run sam3-prepare again to rebuild the view"
                )
    return problems


def verified_absent(
    family_of: dict, negatives: dict, name: str, records: dict[str, list[dict]]
) -> list[dict]:
    """Distinct images of the category's negatives that carry no label of the category's own family (rule 4)."""
    family = family_of[name]
    seen: dict[object, dict] = {}
    for neg in negatives[name]:
        for r in records[neg]:
            if all(family_of.get(c) != family for c in r["categories"]):
                seen.setdefault(r["id"], r)
    return list(seen.values())


def unveil_dir(root: str | Path) -> Path:
    return Path(root) / "prepared" / "unveil"


def _load_recipe(path: Path) -> dict:
    try:
        recipe = yaml.safe_load(path.read_text())
    except OSError:
        raise RecipeError(
            f"recipe not found: {path}; run sam3-prepare on the root (it writes a skeleton) or set recipe=<file>"
        )
    except yaml.YAMLError as e:
        raise RecipeError(f"{path} is not valid YAML ({e}); fix the file and run again")
    if not isinstance(recipe, dict) or not isinstance(recipe.get("categories"), dict):
        raise RecipeError(
            f"{path} needs a `categories:` mapping of name -> {{kind: region|balanced}}; "
            "see docs/unveil-recipe.md"
        )
    return recipe


def _families(recipe: dict, problems: list[str]) -> dict[str, str]:
    """category -> family name, from the recipe's ``families:``."""
    family_of: dict[str, str] = {}
    declared = recipe.get("families") or {}
    if not isinstance(declared, dict):
        problems.append(
            "families is not a mapping of family -> [categories]; fix `families:` in recipe.yaml"
        )
        return family_of
    for fam, members in declared.items():
        if not isinstance(members, list) or not members:
            problems.append(
                f"family {fam!r} has no category list ({members!r}); give `{fam}: [category, ...]` "
                "under `families:` in recipe.yaml or remove it"
            )
            continue
        for name in members:
            if name in family_of:
                problems.append(
                    f"category {name!r} is in two families ({family_of[name]!r} and {fam!r}); "
                    "keep it in one under `families:` in recipe.yaml"
                )
            family_of[name] = fam
    return family_of


def validate_recipe(root: str | Path, recipe_path: str | Path | None = None) -> dict:
    """Check the recipe and the prepared view of ``root``; return the plan or raise ``RecipeError``."""
    view = unveil_dir(root)
    prompts_file = view / "prompts.json"
    if not prompts_file.is_file():
        raise RecipeError(
            f"no UNVEIL view at {view}; run `sam3-prepare root={root}` first"
        )
    prompts = json.loads(prompts_file.read_text())
    prepared, slugs = prompts["task"], prompts["slugs"]
    recipe = _load_recipe(Path(recipe_path) if recipe_path else view / "recipe.yaml")
    cats: dict = recipe["categories"]
    problems: list[str] = []
    if not cats:
        problems.append(
            "categories is empty; list at least one category under `categories:` in recipe.yaml"
        )

    for name in cats:
        if name not in prepared:
            problems.append(
                f"category {name!r} is not in the prepared data (prompts.json has {prepared}); "
                "remove it from recipe.yaml or add its images and run sam3-prepare again"
            )
    for name, spec in cats.items():
        kind = spec.get("kind") if isinstance(spec, dict) else None
        if kind not in KINDS:
            problems.append(
                f"category {name!r} has no valid kind (got {kind!r}); set `kind: region` or `kind: balanced` "
                "for it in recipe.yaml, there is no default"
            )

    family_of = _families(recipe, problems)
    for name in cats:
        if name not in family_of:
            problems.append(
                f"category {name!r} is in no family; add it to `families:` in recipe.yaml "
                "(alone in its own family if it never shares an image with another category)"
            )
    for name in family_of:
        if name not in cats:
            problems.append(
                f"family member {name!r} is not in `categories:`; add it there or remove it from `families:`"
            )

    negatives: dict[str, list[str]] = {}
    declared = recipe.get("negatives") or {}
    if not isinstance(declared, dict):
        problems.append(
            "negatives is not a mapping of category -> [categories]; fix `negatives:` in recipe.yaml"
        )
        declared = {}
    for name in cats:
        default = [c for c in cats if family_of.get(c) != family_of.get(name)]
        listed = declared.get(name, default)
        if not isinstance(listed, list):
            problems.append(
                f"negatives of {name!r} is {listed!r}, not a list; write `{name}: [category, ...]` "
                "(or `[]` for none) under `negatives:` in recipe.yaml, or remove the entry for the default"
            )
            listed = []
        negatives[name] = list(listed)
        for neg in negatives[name]:
            if neg not in cats:
                problems.append(
                    f"negative {neg!r} of {name!r} is not in `categories:`; fix `negatives:` in recipe.yaml"
                )
            elif family_of.get(neg) == family_of.get(name):
                problems.append(
                    f"negative {neg!r} of {name!r} is from the same family ({family_of[name]!r}); same-family "
                    "images are never verified absent, remove it from `negatives:` in recipe.yaml"
                )

    if recipe.get("exclude_unannotated", True) is not True:
        problems.append(
            "exclude_unannotated is false, so unannotated images would reach the training index; "
            "set `exclude_unannotated: true` in recipe.yaml (an empty label does not mean absent)"
        )

    train_images: dict[str, int] = {}
    for name in cats:
        index = view / "index" / "train" / f"{slugs.get(name, name)}.jsonl"
        if not index.is_file():
            train_images[name] = 0
            problems.append(
                f"no train index for category {name!r} at {index}; run `sam3-prepare root={root}` "
                "to build the view, or remove the category from recipe.yaml"
            )
            continue
        records = [json.loads(x) for x in index.read_text().splitlines() if x.strip()]
        train_images[name] = len(records)
        bad = record_problems(f"train index {index.name}", records)
        if bad:
            problems += bad
            continue
        empty = [r["id"] for r in records if not r.get("categories")]
        if empty:
            problems.append(
                f"{index.name} lists image id(s) {empty[:5]} that are not annotated for any category; "
                f"they must not reach the training index: run `sam3-prepare root={root}` again "
                "to rebuild the view"
            )
        if not records and name in prepared:
            problems.append(
                f"category {name!r} has no train images in {index}; add images or remove it from recipe.yaml"
            )
    joint = recipe.get("joint")
    if joint is not None and not (
        isinstance(joint, list)
        and len(joint) == 2
        and joint[0] != joint[1]
        and all(j in cats for j in joint)
    ):
        problems.append(
            f"joint is {joint!r}; give exactly two different categories of `categories:` "
            "(`joint: [a, b]`) or remove `joint:` from recipe.yaml to skip the joint module"
        )
        joint = None
    if "calibration" in recipe:
        problems.append(
            "`calibration:` is not used (thresholds are fitted on the val split, so no train image is held out); "
            "remove it from recipe.yaml"
        )
    v = recipe.get("epochs")
    if v is not None and (msg := epochs_problem(v)):
        problems.append(msg)
    v = recipe.get("accumulate")
    if v is not None and (
        isinstance(v, bool) or not isinstance(v, (int, float)) or v < 1 or v != int(v)
    ):
        problems.append(
            f"accumulate is {v!r}; use a whole number of 1 or more under `accumulate:` in recipe.yaml, or remove it"
        )
    if problems:
        raise RecipeError(
            f"{len(problems)} problem(s) in the recipe:\n- " + "\n- ".join(problems)
        )
    return {
        "categories": list(cats),
        "kinds": {n: s["kind"] for n, s in cats.items()},
        # the one prompt-id order of the run: the recipe's category order (text encoder rows, checkpoint ``prompts``)
        "prompts": {n: i for i, n in enumerate(cats)},
        "per_gpu": max(1, PAPER_IMAGES_PER_STEP // max(1, len(cats))),
        "train_images": train_images,
        "negatives": negatives,
        "joint": joint,
        "family_of": family_of,
        "epochs": recipe.get("epochs"),
        "accumulate": int(recipe.get("accumulate") or 1),
    }
