"""Sample data by ground truth label (Match, Mismatch, Inconclusive).

Reads a mapping CSV, groups rows by ground_truth label, and samples
configurable fractions per label per split (train/validation/test).
Copies images and writes per-split mapping CSVs and JSONs.

Config format:
{
  "train":      {"fraction_match": 0.8, "fraction_mismatch": 0.8, "fraction_inconclusive": 0.8},
  "validation": {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1},
  "test":       {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1}
}

Usage:
    .venv/bin/python scripts/sampling/sample_data_by_label.py \\
        --config_file config/label/sampling_config_label_1.json \\
        --mapping_csv data/downloaded/images/smartwatch/smartwatch_mapping.csv \\
        --image_dir data/downloaded/images/smartwatch \\
        --output_dir data/sampled/smartwatch_1 \\
        --output_name smartwatch_1
"""

import argparse
import json
import logging
import os
import random
import shutil

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SPLITS = ["train", "validation", "test"]
LABELS = ["match", "mismatch", "inconclusive"]

# Column names in the mapping CSV
COL_ID = "id"
COL_GT = "ground_truth"
COL_REF = "reference_image_filename"
COL_IMG = "image_filename"


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def build_label_pools(df):
    """Group row indices by ground_truth label."""
    pools = {label: [] for label in LABELS}
    for idx, row in df.iterrows():
        gt = str(row[COL_GT]).strip().lower()
        if gt in pools:
            pools[gt].append(idx)
    return pools


def sample_splits(df, config):
    """Return {split: list_of_row_indices} with no overlap across splits."""
    pools = build_label_pools(df)
    for label in LABELS:
        logging.info(f"  Pool: {len(pools[label])} {label}")
        random.shuffle(pools[label])

    offsets = {label: 0 for label in LABELS}
    split_indices = {s: [] for s in SPLITS}

    for split in SPLITS:
        cfg = config.get(split, {})
        for label in LABELS:
            frac = cfg.get(f"fraction_{label}", 0.0)
            pool = pools[label]
            n = int(len(pool) * frac)
            drawn = pool[offsets[label] : offsets[label] + n]
            offsets[label] += n
            split_indices[split].extend(drawn)
            logging.info(f"  {split}: {len(drawn)} {label} ({frac*100:.0f}%)")
        logging.info(f"  {split}: total {len(split_indices[split])}")

    return split_indices


def copy_images(split_df, image_dir, output_dir):
    """Copy reference and candidate images for the split."""
    ref_out = os.path.join(output_dir, "reference_images")
    img_out = os.path.join(output_dir, "images")
    os.makedirs(ref_out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    copied, missing = 0, 0
    for _, row in split_df.iterrows():
        for fn_col, subdir, dest in [
            (COL_REF, "reference_images", ref_out),
            (COL_IMG, "images", img_out),
        ]:
            fn = str(row.get(fn_col, "")).strip()
            if not fn:
                continue
            src = os.path.join(image_dir, subdir, fn)
            dst = os.path.join(dest, fn)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                logging.warning(f"  Missing: {src}")
                missing += 1

    logging.info(f"  Copied {copied} files, {missing} missing")


def df_to_eval_json(split_df, output_name):
    """Convert split DataFrame to evaluation JSON list."""
    items = []
    for _, row in split_df.iterrows():
        ref_files = [f.strip() for f in str(row.get(COL_REF, "")).split(",") if f.strip()]
        img_files = [f.strip() for f in str(row.get(COL_IMG, "")).split(",") if f.strip()]

        item = {
            "id": row[COL_ID],
            "ground_truth": str(row[COL_GT]).strip().capitalize(),
            "reference_image_filenames_list": [f"reference_images/{f}" for f in ref_files],
            "image_filenames_list": [f"images/{f}" for f in img_files],
        }

        # Include any extra columns
        skip = {COL_ID, COL_GT, COL_REF, COL_IMG}
        for col in split_df.columns:
            if col not in skip:
                val = row[col]
                item[col] = val if pd.notna(val) else None

        items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Sample data by ground truth label with per-split fractions."
    )
    parser.add_argument("--config_file", required=True,
                        help="Sampling config JSON (fractions per label per split).")
    parser.add_argument("--mapping_csv", required=True,
                        help="Input mapping CSV with id, ground_truth, reference_image_filename, image_filename.")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing reference_images/ and images/ subdirectories.")
    parser.add_argument("--output_dir", required=True,
                        help="Output root. Will contain train/, validation/, test/ subdirs.")
    parser.add_argument("--output_name", required=True,
                        help="Name prefix for output CSV/JSON files.")

    args = parser.parse_args()

    config = load_json(args.config_file)
    df = pd.read_csv(args.mapping_csv)

    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    if "id" not in df.columns and "ID" in df.columns:
        df.rename(columns={"ID": "id"}, inplace=True)

    for req in [COL_ID, COL_GT, COL_REF, COL_IMG]:
        if req not in df.columns:
            logging.error(f"Missing required column '{req}' in {args.mapping_csv}. Columns: {list(df.columns)}")
            return

    logging.info(f"Loaded {len(df)} rows from {args.mapping_csv}")

    split_indices = sample_splits(df, config)

    total = sum(len(v) for v in split_indices.values())
    logging.info(f"Total sampled: {total}")
    if total == 0:
        logging.warning("No rows sampled. Check config fractions.")
        return

    csv_fn = f"{args.output_name}_mapping.csv"
    json_fn = f"{args.output_name}_mapping.json"

    for split in SPLITS:
        indices = split_indices[split]
        if not indices:
            logging.info(f"--- {split.upper()}: 0 rows, skipping ---")
            continue

        logging.info(f"--- {split.upper()}: {len(indices)} rows ---")
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        split_df = df.loc[indices].copy()

        # Write CSV
        csv_path = os.path.join(split_dir, csv_fn)
        split_df.to_csv(csv_path, index=False)
        logging.info(f"  Wrote {csv_path}")

        # Copy images
        copy_images(split_df, args.image_dir, split_dir)

        # Write JSON
        json_path = os.path.join(split_dir, json_fn)
        eval_data = df_to_eval_json(split_df, args.output_name)
        with open(json_path, "w") as f:
            json.dump(eval_data, f, indent=4)
        logging.info(f"  Wrote {json_path}")

    logging.info("Done.")


if __name__ == "__main__":
    main()