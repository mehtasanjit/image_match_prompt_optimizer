"""Extract optimized prompt(s) from grid run results.

Scans all JSON files in a grid run folder, finds cells matching
the given CF name key and num_iterations, and saves the final
prompt text to file(s).

Usage:
    .venv/bin/python scripts/extract_prompt_from_grid_run.py \
        --grid_run_folder grid_runs/match/power_bank_3 \
        --cf_key moderate \
        --num_iterations 18 \
        --output_file prompts/power_bank_moderate_18.txt
"""

import argparse
import json
import logging
import os
import sys
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known CF name prefixes to strip for suffix extraction.
# Add your own prefixes here if your CF names follow a different convention.
_CF_PREFIXES = [
    "img_match_metric_match_only_",
    "img_match_metric_mismatch_only_",
    "img_mismatch_metric_match_only_",
    "img_mismatch_metric_mismatch_only_",
    "img_match_weighted_",
    "img_mismatch_weighted_",
]


def _extract_cf_suffix(cf_name: str) -> str:
    """Strip known prefix to get the variant suffix.

    'img_match_weighted_guarded'      -> 'guarded'
    'img_match_metric_match_only_hi_agg_guarded' -> 'hi_agg_guarded'
    """
    lower = cf_name.lower()
    for prefix in _CF_PREFIXES:
        if lower.startswith(prefix):
            return lower[len(prefix):]
    return lower


def find_matching_prompts(grid_run_folder, cf_key, num_iterations):
    """Scan all JSON files in folder and return matching (file, result) pairs."""
    matches = []
    json_files = sorted(glob.glob(os.path.join(grid_run_folder, "*.json")))

    if not json_files:
        logger.error("No JSON files found in %s", grid_run_folder)
        return matches

    logger.info("Scanning %d JSON files in %s", len(json_files), grid_run_folder)

    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Skipping %s: %s", os.path.basename(json_path), e)
            continue

        results = data.get("results", [])
        for result in results:
            if "error" in result:
                continue

            cell = result.get("grid_cell", {})
            cf_name = cell.get("eval_cf_name", "")
            iters = cell.get("num_iterations")

            # Match CF key: strip known prefix and compare the suffix exactly
            # e.g. "moderate" matches "img_match_metric_match_only_moderate"
            # but "guarded" does NOT match "..._hi_agg_guarded"
            cf_suffix = _extract_cf_suffix(cf_name)
            if cf_suffix != cf_key.lower():
                continue

            if num_iterations is not None and iters != num_iterations:
                continue

            prompt = result.get("gepa", {}).get("final_prompt")
            if not prompt:
                logger.warning("Matched cell in %s but final_prompt is empty", os.path.basename(json_path))
                continue

            matches.append({
                "file": os.path.basename(json_path),
                "cf_name": cf_name,
                "num_iterations": iters,
                "prompt_name": result.get("gepa", {}).get("prompt_name", ""),
                "prompt": prompt,
            })

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Extract optimized prompt from grid run results by CF key and iteration count."
    )
    parser.add_argument("--grid_run_folder", required=True,
                        help="Path to folder containing grid run JSON files.")
    parser.add_argument("--cf_key", required=True,
                        help="CF name key to match (e.g. 'moderate', 'highly_aggressive_guarded'). "
                             "Matches if this string appears in eval_cf_name.")
    parser.add_argument("--num_iterations", type=int, default=None,
                        help="Number of iterations to match. If omitted, matches all iteration counts.")
    parser.add_argument("--output_file", required=True,
                        help="Output file path for the extracted prompt. "
                             "If multiple matches, creates _1.txt, _2.txt, etc.")

    args = parser.parse_args()

    matches = find_matching_prompts(args.grid_run_folder, args.cf_key, args.num_iterations)

    if not matches:
        logger.error("No matching cells found for cf_key='%s', num_iterations=%s",
                      args.cf_key, args.num_iterations)
        sys.exit(1)

    if len(matches) == 1:
        # Single match — write directly to output_file
        m = matches[0]
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, 'w') as f:
            f.write(m["prompt"])
        logger.info("Extracted prompt from %s (cf=%s, iters=%d) -> %s",
                     m["file"], m["cf_name"], m["num_iterations"], args.output_file)
    else:
        # Multiple matches — create numbered files and warn
        base, ext = os.path.splitext(args.output_file)
        if not ext:
            ext = ".txt"

        logger.warning("WARNING: Found %d matching cells! Creating numbered output files.", len(matches))

        for i, m in enumerate(matches, 1):
            out_path = f"{base}_{i}{ext}"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(m["prompt"])
            logger.info("  [%d/%d] %s (cf=%s, iters=%d) -> %s",
                         i, len(matches), m["file"], m["cf_name"], m["num_iterations"], out_path)

        print(f"\nWARNING: {len(matches)} matching cells found (expected 1). "
              f"Created {len(matches)} output files.", file=sys.stderr)


if __name__ == "__main__":
    main()