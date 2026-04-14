# Data Organization

## Overview

The objective is to compare a **user image** against a **reference image** and determine whether the product shown is a **Match**, **Mismatch**, or **Inconclusive**. Data is organized to support sampling, training, and evaluation for prompt optimization.

---

## Directory Structure

```
data/
├── images/                          # Full labeled dataset
│   ├── <category_a>/                # One folder per product category
│   │   ├── <category_a>_mapping.csv # Category-specific mapping
│   │   ├── reference_images/        # Reference images
│   │   │   ├── uuid1.jpg
│   │   │   └── uuid2.jpg
│   │   └── images/                  # User-submitted images
│   │       ├── uuid1.jpg
│   │       └── uuid2.jpg
│   ├── <category_b>/
│   │   ├── <category_b>_mapping.csv
│   │   ├── reference_images/
│   │   └── images/
│   └── ...
│
└── sampled/                         # Sampled subsets for training/eval
    └── <sample_name>/
        ├── train/
        │   ├── <sample_name>_mapping.csv
        │   ├── <sample_name>_mapping.json
        │   ├── reference_images/
        │   └── images/
        ├── validation/
        │   ├── ...
        └── test/
            ├── ...
```

---

## Mapping File Format

Each mapping CSV contains one row per image pair:

| Column | Description |
|---|---|
| `id` | Unique identifier for the image pair |
| `category` | Product category |
| `reference_image_filename` | Filename of the reference image |
| `image_filename` | Filename of the user image |
| `ground_truth` | Labeled class: `Match`, `Mismatch`, or `Inconclusive` |

### Example

```csv
id,category,reference_image_filename,image_filename,ground_truth
abc123,footwear,abc123.jpg,abc123.jpg,Match
def456,footwear,def456.jpg,def456.jpg,Mismatch
ghi789,footwear,ghi789.jpg,ghi789.jpg,Inconclusive
```

---

## Image Naming Convention

Images use UUID-based filenames: `{uuid}.jpg`

Each image pair shares the same UUID across `reference_images/` and `images/`:
- `reference_images/abc123.jpg` — the reference image
- `images/abc123.jpg` — the user-submitted image

---

## Sampled Data (`data/sampled/`)

Sampled subsets are created by the sampling scripts for use in prompt optimization. Each sampled dataset has:

- **train/**: Used for prompt optimization
- **validation/**: Used for post-optimization evaluation
- **test/**: Held-out set for final generalization check

Each split contains:
- A mapping CSV with the same format as above
- A mapping JSON (optimization-compatible format with image path lists)
- Copies of the actual images in `reference_images/` and `images/` subdirectories

### Sampling Methods

1. **Label-based (stratified)**: Preserves the ground truth distribution (Match/Mismatch/Inconclusive ratio) from the full dataset. Recommended for training.

2. **Confusion-matrix based**: Samples by prediction outcome (TP/FP/FN/TN) relative to a prior model run. Useful for targeted analysis.

---

## Category-Centric Data Resolution

All scripts (pipeline, sampling, GEPA optimization) follow a **category-centric** approach to locate data. Every run requires **either** a `--category` name **or** an explicit `--mapping_csv` path:

| Method | How it works | Example |
|--------|-------------|---------|
| **By category** | Auto-resolves to `<data_dir>/<category>_mapping.csv` | `--category smartwatch --data_dir ./data/images/smartwatch` → `./data/images/smartwatch/smartwatch_mapping.csv` |
| **By mapping CSV** | Uses the exact path provided | `--mapping_csv ./data/sampled/custom_split/my_mapping.csv` |

If neither is provided, scripts will error with a clear message.

The mapping CSV must have this format:

```csv
id,category,reference_image_filename,image_filename,ground_truth
abc123,footwear,abc123.jpg,abc123.jpg,Match
def456,footwear,def456.jpg,def456.jpg,Mismatch
ghi789,footwear,ghi789.jpg,ghi789.jpg,Inconclusive
```

For the GEPA grid runners, each data split (train, eval, test, full) independently resolves its mapping CSV using the same logic:
- Training: `--category` or `--mapping_csv`
- Validation: `--eval_category` (falls back to `--category`) or `--eval_mapping_csv`
- Test: `--test_category` (falls back to `--eval_category` → `--category`) or `--test_mapping_csv`
- Full: `--full_data_category` (falls back to `--category`) or `--full_data_mapping_csv`

---

## Adding a New Category

1. Create a subdirectory under `data/images/` with the category name
2. Place reference images in `reference_images/` and user images in `images/`
3. Create a `{category}_mapping.csv` with the columns above
4. Run sampling to create train/validation/test splits
