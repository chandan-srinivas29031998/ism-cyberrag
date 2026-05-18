"""
Build the deterministic 30-question stratified subset used by the HPO trial run.

Run once. Re-running produces byte-identical output. The output JSON lives at
`evaluations/sprint-3/hpo/hpo_subset.json` and is read by `hpo_base_task.py`.

Usage:
    python scripts/hpo_subset.py            # build the subset (errors if file exists)
    python scripts/hpo_subset.py --force    # overwrite an existing subset file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "evaluations" / "eval_questions.json"
OUT_PATH = PROJECT_ROOT / "evaluations" / "sprint-3" / "hpo" / "hpo_subset.json"

SEED = 42
TARGET_PER_CATEGORY = {
    "easy": 8,
    "medium": 8,
    "hard": 6,
    "very_hard": 3,
    "out_of_scope": 5,
}
EXPECTED_TOTAL = sum(TARGET_PER_CATEGORY.values())


def _load_eval_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Eval set must be a JSON array. Got {type(data).__name__}.")
    return data


def _stratified_sample(eval_set: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[tuple[int, dict]]] = {}
    for idx, q in enumerate(eval_set):
        cat = q.get("category", "unknown")
        by_cat.setdefault(cat, []).append((idx, q))

    missing = [c for c in TARGET_PER_CATEGORY if c not in by_cat]
    if missing:
        raise ValueError(
            f"Eval set is missing categories required by the HPO subset spec: {missing}. "
            f"Found categories: {sorted(by_cat.keys())}"
        )

    picked: list[dict] = []
    for cat, want in TARGET_PER_CATEGORY.items():
        pool = sorted(by_cat[cat], key=lambda t: t[0])  # deterministic input order
        if len(pool) < want:
            raise ValueError(
                f"Not enough questions in category '{cat}': have {len(pool)}, want {want}."
            )
        chosen = rng.sample(pool, want)
        chosen.sort(key=lambda t: t[0])
        for original_idx, q in chosen:
            picked.append({
                "subset_index": len(picked),
                "source_index": original_idx,
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "category": q.get("category", "unknown"),
            })

    if len(picked) != EXPECTED_TOTAL:
        raise AssertionError(
            f"Subset size mismatch: produced {len(picked)}, expected {EXPECTED_TOTAL}."
        )
    return picked


def _validate(subset: list[dict]) -> None:
    counts = Counter(item["category"] for item in subset)
    for cat, want in TARGET_PER_CATEGORY.items():
        got = counts.get(cat, 0)
        if got != want:
            raise AssertionError(f"Category '{cat}': expected {want}, got {got}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the subset file if it already exists.",
    )
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(
            f"ERROR: subset file already exists at {OUT_PATH}.\n"
            f"       Re-running would be a no-op (deterministic output), but refusing\n"
            f"       to overwrite by default. Use --force to rebuild.",
            file=sys.stderr,
        )
        return 1

    try:
        eval_set = _load_eval_set(EVAL_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        subset = _stratified_sample(eval_set, seed=SEED)
        _validate(subset)
    except (ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "source_path": str(EVAL_PATH.relative_to(PROJECT_ROOT)),
        "target_per_category": TARGET_PER_CATEGORY,
        "questions": subset,
    }
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Round-trip read for integrity.
    with OUT_PATH.open("r", encoding="utf-8") as f:
        reloaded = json.load(f)
    if len(reloaded["questions"]) != EXPECTED_TOTAL:
        print("ERROR: round-trip read produced a different size.", file=sys.stderr)
        return 1

    counts = Counter(item["category"] for item in reloaded["questions"])
    print(f"Wrote {EXPECTED_TOTAL} questions to {OUT_PATH.relative_to(PROJECT_ROOT)}")
    for cat in TARGET_PER_CATEGORY:
        print(f"  {cat:14s}: {counts.get(cat, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
