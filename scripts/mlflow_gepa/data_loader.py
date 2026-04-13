"""
Data loader for MLflow GEPA prompt optimization.

Loads the mapping CSV for a given category, resolves image file paths,
and returns a GEPA-compatible pandas DataFrame.
"""

import csv
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_image_path(data_dir: str, filename: str, subdirectory: str) -> str:
    """
    Resolve the full path for an image file.

    Tries subdirectory path first (data_dir/subdirectory/filename),
    then falls back to direct path (data_dir/filename).

    Args:
        data_dir: Base directory containing images.
        filename: Image filename.
        subdirectory: Primary subdirectory name (e.g. "reference_images", "images").

    Returns:
        Resolved absolute path. Logs a warning if neither location exists.
    """
    subdir_path = os.path.join(data_dir, subdirectory, filename)
    if os.path.exists(subdir_path):
        return subdir_path

    direct_path = os.path.join(data_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    logger.warning(
        "Image not found at '%s' or '%s'. Using subdirectory path as placeholder.",
        subdir_path,
        direct_path,
    )
    return subdir_path


def load_eval_data(
    category: str,
    data_dir: str,
    mapping_csv: str = "",
    limit: int = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load evaluation data for GEPA prompt optimization.

    Reads a mapping CSV (either explicitly provided or auto-resolved as
    ``{data_dir}/{category}_mapping.csv``) and builds a DataFrame with
    the two columns GEPA expects:

    * **inputs** - dict with keys ``reference_image_path``, ``image_path``
    * **outputs** - dict with key ``ground_truth``

    The CSV must contain columns: ``id``, ``reference_image_filename``,
    ``image_filename``, ``ground_truth``. An optional ``category`` column
    is also supported.

    Args:
        category: Product category name (e.g. "smartwatch", "sandal").
        data_dir: Directory that contains images (with reference_images/
                  and images/ subdirectories).
        mapping_csv: Explicit path to the mapping CSV. If empty, defaults
                     to ``{data_dir}/{category}_mapping.csv``.
        limit: Max number of samples to return. 0 means return all.
        random_seed: Seed used when sub-sampling with *limit*.

    Returns:
        pandas DataFrame ready to pass as ``train_data`` to
        ``mlflow.genai.optimize_prompts``.

    Raises:
        FileNotFoundError: If the mapping CSV does not exist.
        ValueError: If the mapping CSV is empty or yields zero usable rows.
    """
    if mapping_csv:
        csv_path = mapping_csv
    elif category:
        csv_path = os.path.join(data_dir, f"{category}_mapping.csv")
    else:
        raise ValueError(
            "Either mapping_csv or category must be provided to locate the mapping CSV."
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Mapping CSV not found: {csv_path}\n"
            f"  Tried: mapping_csv='{mapping_csv}', category='{category}', data_dir='{data_dir}'\n"
            f"  Fix: pass --mapping_csv explicitly, or ensure --category matches the CSV filename "
            f"(expected: {data_dir}/<category>_mapping.csv)"
        )

    # Read CSV
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        raw_data = list(reader)

    if not raw_data:
        raise ValueError(f"Mapping CSV is empty: {csv_path}")

    logger.info("Loaded %d rows from %s", len(raw_data), csv_path)

    records = []
    skipped = 0

    for idx, row in enumerate(raw_data):
        ref_file = row.get("reference_image_filename", "").strip()
        img_file = row.get("image_filename", "").strip()

        if not ref_file or not img_file:
            logger.warning("Row %d missing reference or candidate filename. Skipping.", idx)
            skipped += 1
            continue

        reference_image_path = _resolve_image_path(data_dir, ref_file, "reference_images")
        image_path = _resolve_image_path(data_dir, img_file, "images")

        ground_truth = row.get("ground_truth", "Inconclusive").strip()

        records.append(
            {
                "inputs": {
                    "reference_image_path": reference_image_path,
                    "image_path": image_path,
                },
                "outputs": {"ground_truth": str(ground_truth)},
            }
        )

    if skipped:
        logger.info("Skipped %d rows with missing filenames.", skipped)

    if not records:
        raise ValueError(
            f"No usable rows found in {csv_path}. "
            "Check that rows have reference_image_filename and image_filename columns."
        )

    df = pd.DataFrame(records)
    logger.info("Built DataFrame with %d usable examples.", len(df))

    if limit > 0 and limit < len(df):
        df = df.sample(n=limit, random_state=random_seed).reset_index(drop=True)
        logger.info("Sub-sampled to %d examples (seed=%d).", limit, random_seed)

    return df


# ── CLI smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Smoke-test the GEPA data loader."
    )
    parser.add_argument(
        "--category", required=True, help="Product category name (e.g. smartwatch)"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory with mapping CSV and images",
    )
    parser.add_argument(
        "--mapping_csv",
        default="",
        help="Explicit path to mapping CSV (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max rows to display (0 = all)",
    )
    args = parser.parse_args()

    df = load_eval_data(
        category=args.category,
        data_dir=args.data_dir,
        mapping_csv=args.mapping_csv,
        limit=args.limit,
    )

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.head().to_string(index=False))