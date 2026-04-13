"""Confusion-matrix-based sampling for binary classification.

Extends accuracy-based sampling to support per-quadrant fractions:
  fraction_tp, fraction_fp, fraction_fn, fraction_tn

Supports configurable positive class (default: Match).
  For binary Match vs Not_Match:
    TP = model=Match, GT=Match
    FP = model=Match, GT≠Match
    FN = model≠Match, GT=Match
    TN = model≠Match, GT≠Match

  For binary Mismatch vs Not_Mismatch (--positive_class mismatch):
    TP = model=Mismatch, GT=Mismatch
    FP = model=Mismatch, GT≠Mismatch
    FN = model≠Mismatch, GT=Mismatch
    TN = model≠Mismatch, GT≠Mismatch
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SPLITS = ["train", "validation", "test"]
QUADRANTS = ["tp", "fp", "fn", "tn"]

# Module-level positive class (set from CLI arg or caller)
_POSITIVE_CLASS = "match"


def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return None


def _extract_prediction(model_output_str):
    """Extract product_match label from model output string.
    
    Returns the product_match value if parseable, otherwise empty string.
    Does NOT fall back to raw text (which may be an error message).
    """
    cleaned = str(model_output_str).strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            label = parsed.get("product_match", "").strip()
            if label:
                return label
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _is_positive_prediction(prediction):
    """Check if prediction matches the positive class (case-insensitive)."""
    return prediction.lower().strip() == _POSITIVE_CLASS


def _is_positive_gt(ground_truth):
    """Check if ground truth matches the positive class (case-insensitive)."""
    return ground_truth.lower().strip() == _POSITIVE_CLASS


def _build_confusion_pools(results_filepath):
    """Load results JSON and return {tp, fp, fn, tn} pools of IDs.
    
    Uses _POSITIVE_CLASS to determine what counts as a positive prediction/GT.
    """
    summary_data = load_json(results_filepath)
    if not summary_data or "individual_results" not in summary_data:
        return None

    pools = {q: [] for q in QUADRANTS}
    prediction_counts = {}
    gt_counts = {}

    for result in summary_data["individual_results"]:
        item_id = result.get('id')
        if not item_id:
            continue

        gt = result.get("ground_truth", "")
        prediction = _extract_prediction(result.get("model_output", ""))

        # Count prediction and GT distributions
        pred_label = prediction.strip().capitalize() if prediction.strip() else "Unknown"
        gt_label = gt.strip().capitalize() if gt.strip() else "Unknown"
        prediction_counts[pred_label] = prediction_counts.get(pred_label, 0) + 1
        gt_counts[gt_label] = gt_counts.get(gt_label, 0) + 1

        pred_positive = _is_positive_prediction(prediction)
        gt_positive = _is_positive_gt(gt)

        if pred_positive and gt_positive:
            pools["tp"].append(item_id)
        elif pred_positive and not gt_positive:
            pools["fp"].append(item_id)
        elif not pred_positive and gt_positive:
            pools["fn"].append(item_id)
        else:
            pools["tn"].append(item_id)

    # Log prediction and ground truth distributions
    total = sum(prediction_counts.values())
    logging.info(f"  Results file: {os.path.basename(results_filepath)} ({total} total)")
    logging.info(f"  Prediction distribution: {dict(sorted(prediction_counts.items()))}")
    logging.info(f"  Ground truth distribution: {dict(sorted(gt_counts.items()))}")

    return pools


def _build_pools_legacy(results_filepath):
    """Legacy: return (accurate_pool, inaccurate_pool) for backward compat."""
    summary_data = load_json(results_filepath)
    if not summary_data or "individual_results" not in summary_data:
        return None, None

    accurate_pool = []
    inaccurate_pool = []
    for result in summary_data["individual_results"]:
        item_id = result.get('id')
        if not item_id:
            continue
        if result.get("score", 0.0) == 1.0:
            accurate_pool.append(item_id)
        else:
            inaccurate_pool.append(item_id)
    return accurate_pool, inaccurate_pool


def _detect_config_mode(split_config):
    """Detect if config uses confusion matrix keys or legacy accuracy keys."""
    for q in QUADRANTS:
        if f"fraction_{q}" in split_config:
            return "confusion_matrix"
    if "fraction_accurate_pred" in split_config or "fraction_inaccurate_pred" in split_config:
        return "legacy"
    return "unknown"


def _is_flat_config(config_json):
    """Check if config is flat (splits at top level) vs wrapped (category → splits)."""
    return any(s in config_json for s in SPLITS)


def extract_sampled_ids_by_split(config_json):
    """Sample IDs per split using confusion matrix or legacy mode.

    Supports two config formats:

    Flat (splits at top level):
      { "train": {output_results_filepath, fraction_tp, ...}, "validation": {...}, "test": {...} }

    Wrapped (category → splits):
      { "category_name": {"train": {...}, "validation": {...}, "test": {...}} }
    """
    split_ids = {s: set() for s in SPLITS}

    # Normalize flat config to wrapped format
    if _is_flat_config(config_json):
        logging.info("Detected flat config format (splits at top level)")
        split_configs = config_json
        _process_one_group("default", split_configs, split_ids)
    else:
        for category, split_configs in config_json.items():
            logging.info(f"Processing category: {category}")
            _process_one_group(category, split_configs, split_ids)

    return split_ids


def _process_one_group(label, split_configs, split_ids):
    """Process one group of split configs (either a named category or 'default')."""
    # Detect mode from first split that has config
    first_cfg = next((split_configs.get(s, {}) for s in SPLITS if split_configs.get(s)), {})
    mode = _detect_config_mode(first_cfg)
    logging.info(f"  Config mode: {mode}")

    results_per_split = {}
    for split in SPLITS:
        cfg = split_configs.get(split, {})
        results_per_split[split] = cfg.get("output_results_filepath")

    unique_files = set(f for f in results_per_split.values() if f)
    if not unique_files:
        logging.warning(f"No results files configured for {label}, skipping.")
        return

    if mode == "confusion_matrix":
        _sample_confusion_matrix(label, split_configs, results_per_split, unique_files, split_ids)
    else:
        _sample_legacy(label, split_configs, results_per_split, unique_files, split_ids)


def _sample_confusion_matrix(category, split_configs, results_per_split, unique_files, split_ids):
    """Sample using TP/FP/FN/TN fractions."""
    if len(unique_files) == 1:
        # Shared file: sequential draw
        results_filepath = next(iter(unique_files))
        if not os.path.exists(results_filepath):
            logging.warning(f"Results file missing for {category}: {results_filepath}")
            return

        pools = _build_confusion_pools(results_filepath)
        if pools is None:
            logging.warning(f"Invalid format in {results_filepath}")
            return

        for q in QUADRANTS:
            logging.info(f"  {category}: {len(pools[q])} {q.upper()} samples")
            random.shuffle(pools[q])

        offsets = {q: 0 for q in QUADRANTS}

        for split in SPLITS:
            cfg = split_configs.get(split, {})
            sampled_this_split = []

            for q in QUADRANTS:
                frac = cfg.get(f"fraction_{q}", 0.0)
                n = int(len(pools[q]) * frac)
                sampled = pools[q][offsets[q]:offsets[q] + n]
                offsets[q] += n
                sampled_this_split.extend(sampled)
                logging.info(f"  {category}/{split}: {len(sampled)} {q.upper()} ({frac*100:.0f}%)")

            split_ids[split].update(sampled_this_split)
            logging.info(f"  {category}/{split}: total {len(sampled_this_split)} samples")

    else:
        # Per-split files
        used_ids = set()
        for split in SPLITS:
            cfg = split_configs.get(split, {})
            results_filepath = results_per_split.get(split)
            if not results_filepath or not os.path.exists(results_filepath):
                continue

            pools = _build_confusion_pools(results_filepath)
            if pools is None:
                continue

            # Exclude used IDs
            for q in QUADRANTS:
                pools[q] = [i for i in pools[q] if i not in used_ids]
                random.shuffle(pools[q])

            sampled_this_split = []
            for q in QUADRANTS:
                frac = cfg.get(f"fraction_{q}", 0.0)
                n = int(len(pools[q]) * frac)
                sampled = pools[q][:n]
                sampled_this_split.extend(sampled)
                logging.info(f"  {category}/{split}: {len(sampled)} {q.upper()} ({frac*100:.0f}%)")

            split_ids[split].update(sampled_this_split)
            used_ids.update(sampled_this_split)


def _sample_legacy(category, split_configs, results_per_split, unique_files, split_ids):
    """Legacy sampling: accurate/inaccurate pools."""
    if len(unique_files) == 1:
        results_filepath = next(iter(unique_files))
        if not os.path.exists(results_filepath):
            return

        accurate_pool, inaccurate_pool = _build_pools_legacy(results_filepath)
        if accurate_pool is None:
            return

        logging.info(f"  {category}: {len(accurate_pool)} accurate, {len(inaccurate_pool)} inaccurate (legacy)")
        random.shuffle(accurate_pool)
        random.shuffle(inaccurate_pool)

        acc_offset = 0
        inacc_offset = 0

        for split in SPLITS:
            cfg = split_configs.get(split, {})
            frac_acc = cfg.get("fraction_accurate_pred", 0.0)
            frac_inacc = cfg.get("fraction_inaccurate_pred", 0.0)

            n_acc = int(len(accurate_pool) * frac_acc)
            n_inacc = int(len(inaccurate_pool) * frac_inacc)

            sampled_acc = accurate_pool[acc_offset:acc_offset + n_acc]
            sampled_inacc = inaccurate_pool[inacc_offset:inacc_offset + n_inacc]

            acc_offset += n_acc
            inacc_offset += n_inacc

            split_ids[split].update(sampled_acc)
            split_ids[split].update(sampled_inacc)

            logging.info(f"  {category}/{split}: {len(sampled_acc)} accurate, {len(sampled_inacc)} inaccurate")


def create_split_mapping_csv(master_csv_path, output_csv_path, category_keys, target_ids):
    """Write rows from master CSV whose IDs are in target_ids and category matches."""
    rows = []
    headers = []

    try:
        with open(master_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
                if "ID" in headers:
                    headers[headers.index("ID")] = "id"
            except StopIteration:
                logging.error("Master CSV is empty.")
                return None

            req_cols = ["id", "category", "reference_image_filename", "image_filename"]
            for c in req_cols:
                if c not in headers:
                    logging.error(f"Master CSV lacks required column '{c}'. Headers: {headers}")
                    return None

            id_idx = headers.index("id")
            cat_idx = headers.index("category")

            for row in reader:
                if len(row) <= max(id_idx, cat_idx):
                    continue
                row_id = row[id_idx]
                if row_id not in target_ids:
                    continue
                # Skip category filter when category_keys is None (flat config)
                if category_keys is not None:
                    row_cat = row[cat_idx].strip()
                    if row_cat not in category_keys:
                        continue
                rows.append(row)

        if rows:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            logging.info(f"  Wrote {len(rows)} rows -> {output_csv_path}")
            return output_csv_path
        else:
            logging.warning(f"  No rows matched for {output_csv_path}")
            return None

    except Exception as e:
        logging.error(f"Error creating split CSV: {e}")
        return None


def consolidate_images(mapping_csv_path, base_data_dir, output_dir):
    logging.info(f"  Consolidating images -> {output_dir}")

    ref_out_dir = os.path.join(output_dir, "reference_images")
    img_out_dir = os.path.join(output_dir, "images")
    os.makedirs(ref_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)

    copied = 0
    missing = 0

    try:
        with open(mapping_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.lower().strip(): v for k, v in row.items()}

                ref_fn = row.get("reference_image_filename", "").strip()
                img_fn = row.get("image_filename", "").strip()

                for fn, subdir, out_dir in [
                    (ref_fn, "reference_images", ref_out_dir),
                    (img_fn, "images", img_out_dir)
                ]:
                    src = os.path.join(base_data_dir, subdir, fn)
                    dst = os.path.join(out_dir, fn)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        copied += 1
                    else:
                        logging.warning(f"  Missing: {src}")
                        missing += 1

    except Exception as e:
        logging.error(f"Error during image consolidation: {e}")

    logging.info(f"  Copied {copied} files, {missing} missing")


def convert_mapping_to_vapo_json(csv_path, output_json_path, category_name):
    """Convert mapping CSV to evaluation JSON format."""
    if not os.path.exists(csv_path):
        logging.error(f"Missing input CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required_cols = ['id', 'reference_image_filename', 'image_filename', 'ground_truth']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Missing required column '{col}' in CSV for JSON conversion.")
            return

    vapo_eval_data = []

    for _, row in df.iterrows():
        ref_files = [f.strip() for f in str(row.get('reference_image_filename', '')).split(',')] if pd.notna(row.get('reference_image_filename')) else []
        img_files = [f.strip() for f in str(row.get('image_filename', '')).split(',')] if pd.notna(row.get('image_filename')) else []

        item = {
            "id": row['id'],
            "category": category_name,
            "ground_truth": str(row['ground_truth']).capitalize(),
            "reference_image_filenames_list": [f"reference_images/{f}" for f in ref_files if f],
            "image_filenames_list": [f"images/{f}" for f in img_files if f],
        }

        for col in df.columns:
            if col not in required_cols:
                item[col] = row[col]

        vapo_eval_data.append(item)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vapo_eval_data, f, indent=4)

    logging.info(f"  CSV->JSON: {len(vapo_eval_data)} rows -> {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Confusion-matrix sampling: split data by TP/FP/FN/TN fractions. Also supports legacy accuracy-based mode."
    )

    parser.add_argument("--config_file", required=True,
                        help="Path to sampling config JSON. Supports fraction_tp/fp/fn/tn or fraction_accurate/inaccurate.")
    parser.add_argument("--base_data_dir", required=True,
                        help="Base directory with source images in <category>/<reference_images|images>/ layout.")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory. Will contain train/, validation/, test/ subdirectories.")
    parser.add_argument("--user_defined_category", required=True,
                        help="Category label used for naming output CSV/JSON files.")
    parser.add_argument("--master_csv",
                        default="data/master_mapping.csv",
                        help="Path to master truth CSV.")
    parser.add_argument("--positive_class", default="match",
                        help="Positive class label for confusion matrix (default: match). "
                             "Use 'mismatch' for binary mismatch classification.")

    args = parser.parse_args()

    # Set module-level positive class
    global _POSITIVE_CLASS
    _POSITIVE_CLASS = args.positive_class.lower().strip()
    logging.info(f"Positive class: '{_POSITIVE_CLASS}'")

    config_json = load_json(args.config_file)
    if not config_json:
        return

    split_ids = extract_sampled_ids_by_split(config_json)

    total = sum(len(ids) for ids in split_ids.values())
    logging.info(f"Total sampled IDs: {total} (train={len(split_ids['train'])}, "
                 f"validation={len(split_ids['validation'])}, test={len(split_ids['test'])})")

    if total == 0:
        logging.warning("No IDs sampled. Check config fractions and results files.")
        return

    # For flat config, don't filter by category (accept all rows matching by ID)
    category_keys = None if _is_flat_config(config_json) else list(config_json.keys())
    csv_filename = f"{args.user_defined_category}_mapping.csv"
    json_filename = f"{args.user_defined_category}_mapping.json"

    for split in SPLITS:
        logging.info(f"--- Processing {split.upper()} split ---")
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        ids = split_ids[split]
        if not ids:
            logging.warning(f"  No IDs for {split} split, skipping.")
            continue

        split_csv = os.path.join(split_dir, csv_filename)
        csv_path = create_split_mapping_csv(
            master_csv_path=args.master_csv,
            output_csv_path=split_csv,
            category_keys=category_keys,
            target_ids=ids
        )

        if not csv_path:
            logging.warning(f"  Skipping {split}: no CSV created.")
            continue

        # Print ground truth label distribution for this split
        try:
            split_df = pd.read_csv(csv_path)
            if 'ground_truth' in split_df.columns:
                gt_counts = split_df['ground_truth'].str.lower().value_counts()
                logging.info(f"  {split.upper()} ground truth distribution:")
                for label, count in gt_counts.items():
                    logging.info(f"    {label}: {count}")
                logging.info(f"    TOTAL: {len(split_df)}")
        except Exception as e:
            logging.warning(f"  Could not read label distribution: {e}")

        consolidate_images(
            mapping_csv_path=csv_path,
            base_data_dir=args.base_data_dir,
            output_dir=split_dir
        )

        convert_mapping_to_vapo_json(
            csv_path=csv_path,
            output_json_path=os.path.join(split_dir, json_filename),
            category_name=args.user_defined_category
        )

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()