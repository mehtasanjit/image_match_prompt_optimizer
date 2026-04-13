"""
Precision-Recall tradeoff analysis for GEPA grid run results.

Reads all grid run JSON files from a folder, groups cells by
validation precision brackets, and within each bracket picks
the cell with the best recall. Outputs a tradeoff table.

Usage:
    python scripts/analyze_grid_precision_recall_tradeoff.py \
        --input_dir grid_runs/sari_3
"""

import argparse
import glob
import json
import math
import os
import sys

PRECISION_BRACKETS = [
    (0.95, 1.00),
    (0.90, 0.95),
    (0.85, 0.90),
    (0.80, 0.85),
    (0.75, 0.80),
    (0.70, 0.75),
    (0.65, 0.70),
    (0.60, 0.65),
    (0.50, 0.60),
    (0.00, 0.50),
]


# Known CF name prefixes to strip for short display names.
# Add your own prefixes here if your CF names follow a different convention.
_CF_PREFIXES = [
    "img_match_metric_match_only_",
    "img_match_metric_mismatch_only_",
    "img_mismatch_metric_match_only_",
    "img_mismatch_metric_mismatch_only_",
    "img_match_weighted_",
    "img_mismatch_weighted_",
]


def _extract_cf_short(cf_name: str) -> str:
    """Extract a short CF label by stripping known prefixes.

    'img_match_metric_match_only_highly_aggressive'    -> 'highly_aggressive'
    'img_match_weighted_guarded'                       -> 'guarded'
    Falls back to last segment if no prefix matches.
    """
    for prefix in _CF_PREFIXES:
        if cf_name.startswith(prefix):
            return cf_name[len(prefix):]
    return cf_name.rsplit("_", 1)[-1]


def _get_class_metrics(eval_block):
    """Extract metrics_match or metrics_mismatch from an eval block (auto-detect)."""
    if not eval_block or "error" in eval_block:
        return None
    agg = eval_block.get("aggregated_metrics", {})
    # Try match first, then mismatch
    m = agg.get("metrics_match")
    if m:
        return m
    m = agg.get("metrics_mismatch")
    if m:
        return m
    return {}


def load_all_cells(input_dir):
    """Load all cells from all grid JSON files in the directory."""
    cells = []
    for filepath in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}", file=sys.stderr)
            continue

        for r in data.get("results", []):
            if "error" in r:
                continue

            gc = r.get("grid_cell", {})
            gepa = r.get("gepa", {})

            # Support both old ("eval") and new ("eval_validation") key names
            val_block = r.get("eval_validation", r.get("eval", {}))
            train_block = r.get("eval_train", {})
            test_block = r.get("eval_test", {})
            full_block = r.get("eval_full", {})

            val_m = _get_class_metrics(val_block)
            train_m = _get_class_metrics(train_block)
            test_m = _get_class_metrics(test_block)
            full_m = _get_class_metrics(full_block)

            if not val_m:
                continue

            cells.append({
                "cf_name": gc.get("eval_cf_name", "?"),
                "cf_short": _extract_cf_short(gc.get("eval_cf_name", "?")),
                "num_iterations": gc.get("num_iterations", "?"),
                "score_key": gc.get("eval_score_key", "?"),
                "gepa_initial": gepa.get("initial_eval_score"),
                "gepa_final": gepa.get("final_eval_score"),
                "val_precision": val_m.get("precision", 0),
                "val_recall": val_m.get("recall", 0),
                "val_f1": val_m.get("f1", 0),
                "train_precision": train_m.get("precision", 0) if train_m else None,
                "train_recall": train_m.get("recall", 0) if train_m else None,
                "train_f1": train_m.get("f1", 0) if train_m else None,
                "test_precision": test_m.get("precision", 0) if test_m else None,
                "test_recall": test_m.get("recall", 0) if test_m else None,
                "test_f1": test_m.get("f1", 0) if test_m else None,
                "full_precision": full_m.get("precision", 0) if full_m else None,
                "full_recall": full_m.get("recall", 0) if full_m else None,
                "full_f1": full_m.get("f1", 0) if full_m else None,
                "source_file": os.path.basename(filepath),
            })

    return cells


def build_tradeoff_table(cells, brackets=None):
    """Group cells into precision brackets, pick best recall in each."""
    if brackets is None:
        brackets = PRECISION_BRACKETS

    table = []

    for lo, hi in brackets:
        # Filter cells whose validation precision falls in [lo, hi)
        # Exception: top bracket includes hi (i.e. [0.95, 1.00])
        in_bracket = [
            c for c in cells
            if lo <= c["val_precision"] < hi or (hi == 1.00 and c["val_precision"] == 1.00)
        ]

        if not in_bracket:
            table.append({
                "bracket": f"{lo:.2f}-{hi:.2f}",
                "count": 0,
                "best": None,
            })
            continue

        # Pick cell with best validation recall within this precision bracket
        best = max(in_bracket, key=lambda c: c["val_recall"])

        table.append({
            "bracket": f"{lo:.2f}-{hi:.2f}",
            "count": len(in_bracket),
            "best": best,
        })

    return table


def print_table(table):
    """Pretty-print the tradeoff table."""
    # Header
    header = (
        f"{'Precision':>12s} | {'#':>3s} | {'CF':>12s} | {'Iters':>5s} | "
        f"{'Val P':>6s} {'Val R':>6s} {'Val F1':>6s} | "
        f"{'Trn P':>6s} {'Trn R':>6s} {'Trn F1':>6s} | "
        f"{'Tst P':>6s} {'Tst R':>6s} {'Tst F1':>6s} | "
        f"{'P Δ':>5s} | "
        f"{'GEPA':>7s}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for row in table:
        bracket = row["bracket"]
        count = row["count"]
        b = row["best"]

        if b is None:
            print(f"{bracket:>12s} | {count:>3d} | {'—':>12s} |")
            continue

        def fmt(v):
            return f"{v:.3f}" if v is not None else "  n/a"

        p_spread = _precision_spread(b)
        p_spread_s = f"{p_spread:.2f}" if p_spread != float("inf") else " n/a"

        print(
            f"{bracket:>12s} | {count:>3d} | {b['cf_short']:>12s} | {b['num_iterations']:>5} | "
            f"{fmt(b['val_precision']):>6s} {fmt(b['val_recall']):>6s} {fmt(b['val_f1']):>6s} | "
            f"{fmt(b['train_precision']):>6s} {fmt(b['train_recall']):>6s} {fmt(b['train_f1']):>6s} | "
            f"{fmt(b['test_precision']):>6s} {fmt(b['test_recall']):>6s} {fmt(b['test_f1']):>6s} | "
            f"{p_spread_s:>5s} | "
            f"{fmt(b['gepa_final']):>7s}"
        )

    print(sep)


def _get_bracket_label(val_precision):
    """Return the precision bracket label for a given value."""
    for lo, hi in PRECISION_BRACKETS:
        if lo <= val_precision < hi or (hi == 1.00 and val_precision == 1.00):
            return f"{lo:.2f}-{hi:.2f}"
    return "?"


def print_all_cells(cells):
    """Print every cell sorted by validation precision descending."""
    sorted_cells = sorted(cells, key=lambda c: -c["val_precision"])

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  n/a"

    header = (
        f"{'#':>3s} | {'Bracket':>11s} | {'CF':>25s} | {'Iters':>5s} | "
        f"{'Val P':>6s} {'Val R':>6s} {'Val F1':>6s} | "
        f"{'Trn P':>6s} {'Trn R':>6s} {'Trn F1':>6s} | "
        f"{'Tst P':>6s} {'Tst R':>6s} {'Tst F1':>6s} | "
        f"{'Ful P':>6s} {'Ful R':>6s} {'Ful F1':>6s} | "
        f"{'P Δ':>5s} | {'Source':>20s}"
    )
    sep = "-" * len(header)
    print(sep)
    print("ALL CELLS (sorted by validation precision)")
    print(sep)
    print(header)
    print(sep)

    for i, c in enumerate(sorted_cells):
        p_spread = _precision_spread(c)
        p_spread_s = f"{p_spread:.2f}" if p_spread != float("inf") else " n/a"
        bracket = _get_bracket_label(c["val_precision"])
        print(
            f"{i+1:>3d} | {bracket:>11s} | {c['cf_short']:>25s} | {c['num_iterations']:>5} | "
            f"{fmt(c['val_precision']):>6s} {fmt(c['val_recall']):>6s} {fmt(c['val_f1']):>6s} | "
            f"{fmt(c['train_precision']):>6s} {fmt(c['train_recall']):>6s} {fmt(c['train_f1']):>6s} | "
            f"{fmt(c['test_precision']):>6s} {fmt(c['test_recall']):>6s} {fmt(c['test_f1']):>6s} | "
            f"{fmt(c['full_precision']):>6s} {fmt(c['full_recall']):>6s} {fmt(c['full_f1']):>6s} | "
            f"{p_spread_s:>5s} | {c['source_file']:>20s}"
        )
    print(sep)
    print()


def print_cf_summary(cells):
    """Print aggregated precision/recall stats grouped by CF name."""
    from collections import defaultdict

    cf_groups = defaultdict(list)
    for c in cells:
        cf_groups[c["cf_short"]].append(c)

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  n/a"

    def _avg(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    def _min_val(values):
        valid = [v for v in values if v is not None]
        return min(valid) if valid else None

    def _max_val(values):
        valid = [v for v in values if v is not None]
        return max(valid) if valid else None

    header = (
        f"{'CF':>25s} | {'#':>3s} | "
        f"{'Trn P':>7s} {'Trn R':>7s} {'Trn F1':>7s} | "
        f"{'Val P':>7s} {'Val R':>7s} {'Val F1':>7s} | "
        f"{'Tst P':>7s} {'Tst R':>7s} {'Tst F1':>7s} | "
        f"{'Ful P':>7s} {'Ful R':>7s} {'Ful F1':>7s}"
    )
    sep = "-" * len(header)

    print(sep)
    print("CF-LEVEL SUMMARY (avg across all iterations per CF)")
    print(sep)
    print(header)
    print(sep)

    for cf_short in sorted(cf_groups.keys()):
        group = cf_groups[cf_short]
        n = len(group)

        trn_p = _avg([c["train_precision"] for c in group])
        trn_r = _avg([c["train_recall"] for c in group])
        trn_f1 = _avg([c["train_f1"] for c in group])
        val_p = _avg([c["val_precision"] for c in group])
        val_r = _avg([c["val_recall"] for c in group])
        val_f1 = _avg([c["val_f1"] for c in group])
        tst_p = _avg([c["test_precision"] for c in group])
        tst_r = _avg([c["test_recall"] for c in group])
        tst_f1 = _avg([c["test_f1"] for c in group])
        ful_p = _avg([c["full_precision"] for c in group])
        ful_r = _avg([c["full_recall"] for c in group])
        ful_f1 = _avg([c["full_f1"] for c in group])

        print(
            f"{cf_short:>25s} | {n:>3d} | "
            f"{fmt(trn_p):>7s} {fmt(trn_r):>7s} {fmt(trn_f1):>7s} | "
            f"{fmt(val_p):>7s} {fmt(val_r):>7s} {fmt(val_f1):>7s} | "
            f"{fmt(tst_p):>7s} {fmt(tst_r):>7s} {fmt(tst_f1):>7s} | "
            f"{fmt(ful_p):>7s} {fmt(ful_r):>7s} {fmt(ful_f1):>7s}"
        )

        # Print min-max range per split
        trn_p_min, trn_p_max = _min_val([c["train_precision"] for c in group]), _max_val([c["train_precision"] for c in group])
        val_p_min, val_p_max = _min_val([c["val_precision"] for c in group]), _max_val([c["val_precision"] for c in group])
        tst_p_min, tst_p_max = _min_val([c["test_precision"] for c in group]), _max_val([c["test_precision"] for c in group])
        ful_p_min, ful_p_max = _min_val([c["full_precision"] for c in group]), _max_val([c["full_precision"] for c in group])

        def rng(lo, hi):
            if lo is None or hi is None:
                return "    n/a"
            return f"{lo:.3f}-{hi:.3f}"

        print(
            f"{'(P range)':>25s} |     | "
            f"{rng(trn_p_min, trn_p_max):>22s} | "
            f"{rng(val_p_min, val_p_max):>22s} | "
            f"{rng(tst_p_min, tst_p_max):>22s} | "
            f"{rng(ful_p_min, ful_p_max):>22s}"
        )

    print(sep)
    print()


def _precision_spread(c):
    """Compute max-min spread of precision across available splits."""
    vals = [v for v in [c.get("train_precision"), c.get("val_precision"), c.get("test_precision")] if v is not None]
    if len(vals) < 2:
        return float("inf")
    return max(vals) - min(vals)


def recommend_best(cells):
    """Pick the cell with lowest F1 variance across train/val/test,
    breaking ties by highest mean F1. Requires val_f1 > 0."""
    eligible = [c for c in cells if c["val_f1"] > 0]
    if not eligible:
        return None

    for c in eligible:
        c["_p_spread"] = _precision_spread(c)
        ps = [v for v in [c.get("train_precision"), c.get("val_precision"), c.get("test_precision")] if v is not None]
        c["_p_mean"] = sum(ps) / len(ps) if ps else 0
        f1s = [v for v in [c.get("train_f1"), c.get("val_f1"), c.get("test_f1")] if v is not None]
        c["_f1_mean"] = sum(f1s) / len(f1s) if f1s else 0

    # Sort by precision spread ascending, then by mean F1 descending
    eligible.sort(key=lambda c: (c["_p_spread"], -c["_f1_mean"]))
    return eligible[0]


def print_recommendation(rec):
    """Print the recommended cell."""
    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print()
    print("=" * 80)
    print("  RECOMMENDATION (smallest precision spread across train/val/test)")
    print("=" * 80)
    print(f"  CF:             {rec['cf_name']}")
    print(f"  Num iterations: {rec['num_iterations']}")
    print(f"  Score key:      {rec['score_key']}")
    print(f"  Source file:    {rec['source_file']}")
    print()
    print(f"  Train:      P={fmt(rec['train_precision'])}  R={fmt(rec['train_recall'])}  F1={fmt(rec['train_f1'])}")
    print(f"  Validation: P={fmt(rec['val_precision'])}  R={fmt(rec['val_recall'])}  F1={fmt(rec['val_f1'])}")
    print(f"  Test:       P={fmt(rec['test_precision'])}  R={fmt(rec['test_recall'])}  F1={fmt(rec['test_f1'])}")
    print(f"  Full:       P={fmt(rec['full_precision'])}  R={fmt(rec['full_recall'])}  F1={fmt(rec['full_f1'])}")
    print(f"  Precision spread (max-min): {fmt(rec['_p_spread'])}")
    print(f"  F1 mean: {fmt(rec['_f1_mean'])}")
    print(f"  GEPA:       {fmt(rec.get('gepa_initial'))} -> {fmt(rec.get('gepa_final'))}")
    print("=" * 80)


def save_json(table, output_path, recommendation=None):
    """Save the tradeoff table as JSON."""
    out = []
    for row in table:
        entry = {"precision_bracket": row["bracket"], "candidates_in_bracket": row["count"]}
        b = row["best"]
        if b:
            entry["best_cell"] = {
                "eval_cf_name": b["cf_name"],
                "num_iterations": b["num_iterations"],
                "eval_score_key": b["score_key"],
                "validation": {"precision": b["val_precision"], "recall": b["val_recall"], "f1": b["val_f1"]},
                "train": {"precision": b["train_precision"], "recall": b["train_recall"], "f1": b["train_f1"]},
                "test": {"precision": b["test_precision"], "recall": b["test_recall"], "f1": b["test_f1"]},
                "full": {"precision": b["full_precision"], "recall": b["full_recall"], "f1": b["full_f1"]},
                "gepa_final_eval_score": b["gepa_final"],
                "source_file": b["source_file"],
            }
        out.append(entry)

    result = {"tradeoff_table": out}
    if recommendation:
        result["recommendation"] = {
            "eval_cf_name": recommendation["cf_name"],
            "num_iterations": recommendation["num_iterations"],
            "eval_score_key": recommendation["score_key"],
            "train": {"precision": recommendation["train_precision"], "recall": recommendation["train_recall"], "f1": recommendation["train_f1"]},
            "validation": {"precision": recommendation["val_precision"], "recall": recommendation["val_recall"], "f1": recommendation["val_f1"]},
            "test": {"precision": recommendation["test_precision"], "recall": recommendation["test_recall"], "f1": recommendation["test_f1"]},
            "full": {"precision": recommendation["full_precision"], "recall": recommendation["full_recall"], "f1": recommendation["full_f1"]},
            "precision_spread": recommendation["_p_spread"],
            "precision_mean": recommendation["_p_mean"],
            "f1_mean": recommendation["_f1_mean"],
            "gepa_final_eval_score": recommendation.get("gepa_final"),
            "source_file": recommendation["source_file"],
            "rationale": "Smallest precision spread (max-min) across train/validation/test splits, tie-broken by highest mean F1",
        }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precision-Recall tradeoff table from GEPA grid run results."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing grid run JSON files")
    parser.add_argument("--output_json", default=None, help="Optional: save tradeoff table as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show all cells, not just the best per bracket")
    args = parser.parse_args()

    cells = load_all_cells(args.input_dir)
    if not cells:
        print("No valid cells found.", file=sys.stderr)
        return

    print(f"Loaded {len(cells)} cells from {args.input_dir}\n")

    if args.verbose:
        print_all_cells(cells)
        print_cf_summary(cells)

    table = build_tradeoff_table(cells)
    print_table(table)

    # Recommendation: best cell by minimal F1 variance across splits
    rec = recommend_best(cells)
    if rec:
        print_recommendation(rec)

    if args.output_json:
        save_json(table, args.output_json, rec)


if __name__ == "__main__":
    main()