# Sampling

Data sampling creates train/validation/test splits from a source dataset. Proper sampling is critical for prompt optimization — the training split composition directly affects what the optimizer learns to prioritize.

Two strategies are available:

1. **Label-based** — sample by ground truth label (Match, Mismatch, Inconclusive)
2. **Confusion-matrix-based** — sample by model prediction outcome (TP, FP, FN, TN)

Both produce the same output structure: `train/`, `validation/`, `test/` subdirectories with per-split mapping CSVs, evaluation JSONs, and copied images.
<!--  -->
---

## 1. Label-Based Sampling

**Script:** `scripts/sampling/sample_data_by_label.py`  
**Config location:** `config/label/`

Samples rows from a mapping CSV based on ground truth label distribution. No model predictions needed — only ground truth labels.

### Config Format

```json
{
  "train":      {"fraction_match": 0.8, "fraction_mismatch": 0.8, "fraction_inconclusive": 0.8},
  "validation": {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1},
  "test":       {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1}
}
```

**Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `fraction_match` | float (0–1) | Fraction of Match-labeled rows to draw for this split |
| `fraction_mismatch` | float (0–1) | Fraction of Mismatch-labeled rows to draw for this split |
| `fraction_inconclusive` | float (0–1) | Fraction of Inconclusive-labeled rows to draw for this split |

Fractions are applied to the total pool of each label. Splits are drawn sequentially (train → validation → test) with no overlap. The sum of fractions across splits should be ≤ 1.0 per label.

### Inputs

| Argument | Required | Description |
|----------|----------|-------------|
| `--config_file` | Yes | Path to the label sampling config JSON |
| `--mapping_csv` | Yes | Input CSV with columns: `id`, `ground_truth`, `reference_image_filename`, `image_filename` |
| `--image_dir` | Yes | Directory containing `reference_images/` and `images/` subdirectories |
| `--output_dir` | Yes | Output root (creates `train/`, `validation/`, `test/` subdirs) |
| `--output_name` | Yes | Prefix for output filenames (e.g., `smartwatch_1`) |

### Step-by-Step

1. **Prepare your data directory** with `reference_images/` and `images/` subdirectories and a mapping CSV:
   ```
   data/images/smartwatch/
   ├── smartwatch_mapping.csv
   ├── reference_images/
   │   └── <uuid>.jpg
   └── images/
       └── <uuid>.jpg
   ```

2. **Choose or create a config** in `config/label/`. The default `sampling_config_label_1.json` uses 80/10/10 splits:
   ```json
   {
     "train":      {"fraction_match": 0.8, "fraction_mismatch": 0.8, "fraction_inconclusive": 0.8},
     "validation": {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1},
     "test":       {"fraction_match": 0.1, "fraction_mismatch": 0.1, "fraction_inconclusive": 0.1}
   }
   ```

3. **Run the sampler:**
   ```bash
   .venv/bin/python scripts/sampling/sample_data_by_label.py \
       --config_file config/label/sampling_config_label_1.json \
       --mapping_csv data/images/smartwatch/smartwatch_mapping.csv \
       --image_dir data/images/smartwatch \
       --output_dir data/sampled/label/smartwatch_1 \
       --output_name smartwatch_1
   ```

4. **Check the output:**
   ```
   data/sampled/label/smartwatch_1/
   ├── train/
   │   ├── smartwatch_1_mapping.csv
   │   ├── smartwatch_1_mapping.json
   │   ├── reference_images/
   │   └── images/
   ├── validation/
   └── test/
   ```

**Notes:**
- `--image_dir` should point directly to the category folder (not the parent `data/images/`)
- The mapping CSV must have columns: `id`, `ground_truth`, `reference_image_filename`, `image_filename`
- Images are copied (not moved) to the output directory

---

## 2. Confusion-Matrix-Based Sampling

**Script:** `scripts/sampling/sample_data_by_confusion_matrix.py`  
**Config location:** `config/match/` or `config/mismatch/`

Samples rows based on model prediction outcomes (TP/FP/FN/TN). This allows targeted sampling — e.g., oversampling false positives to focus optimization on precision.

**Prerequisite:** You must first run the evaluation pipeline (`scripts/pipeline/run_binary_match_pipeline_with_eval.py` or `run_binary_mismatch_pipeline_with_eval.py`) with `--output_file_path` to produce a results JSON file. This file contains per-item predictions and ground truth, which the confusion-matrix sampler uses to classify each item into TP/FP/FN/TN quadrants. The `output_results_filepath` in the sampling config must point to this results file.

### Config Format

```json
{
  "train": {
    "output_results_filepath": "path/to/benchmark_results.json",
    "fraction_tp": 0.32,
    "fraction_fp": 0.38,
    "fraction_fn": 0.38,
    "fraction_tn": 0.32
  },
  "validation": {
    "output_results_filepath": "path/to/benchmark_results.json",
    "fraction_tp": 0.33,
    "fraction_fp": 0.32,
    "fraction_fn": 0.32,
    "fraction_tn": 0.33
  },
  "test": {
    "output_results_filepath": "path/to/benchmark_results.json",
    "fraction_tp": 0.35,
    "fraction_fp": 0.30,
    "fraction_fn": 0.30,
    "fraction_tn": 0.35
  }
}
```

**Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `output_results_filepath` | string | Path to benchmark results JSON (must contain `individual_results` with `id`, `ground_truth`, `model_output`) |
| `fraction_tp` | float (0–1) | Fraction of True Positive samples to draw |
| `fraction_fp` | float (0–1) | Fraction of False Positive samples to draw |
| `fraction_fn` | float (0–1) | Fraction of False Negative samples to draw |
| `fraction_tn` | float (0–1) | Fraction of True Negative samples to draw |

**Confusion matrix definition** depends on `--positive_class`:

| `--positive_class` | TP | FP | FN | TN |
|---------------------|----|----|----|----|
| `match` (default) | model=Match, GT=Match | model=Match, GT≠Match | model≠Match, GT=Match | model≠Match, GT≠Match |
| `mismatch` | model=Mismatch, GT=Mismatch | model=Mismatch, GT≠Mismatch | model≠Mismatch, GT=Mismatch | model≠Mismatch, GT≠Mismatch |

If all splits point to the same results file, draws are sequential (no overlap). If splits point to different files, per-split deduplication is applied.

### Inputs

| Argument | Required | Description |
|----------|----------|-------------|
| `--config_file` | Yes | Path to the confusion matrix sampling config JSON |
| `--base_data_dir` | Yes | Base image directory with `<category>/<reference_images\|images>/` layout |
| `--output_dir` | Yes | Output root (creates `train/`, `validation/`, `test/` subdirs) |
| `--user_defined_category` | Yes | Name prefix for output CSV/JSON files |
| `--master_csv` | No | Path to master CSV |
| `--positive_class` | No | Positive class for confusion matrix: `match` (default) or `mismatch` |

### Step-by-Step

1. **Run the evaluation pipeline first** to produce a results JSON:
   ```bash
   .venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
       --category smartwatch \
       --data_dir data/images/smartwatch \
       --system_prompt prompts/binary_match_or_not.txt \
       --eval_cloud_function_name img_match_metric_match_only_guarded \
       --project my-gcp-project \
       --location us-central1 \
       --eval_cloud_function_location us-central1 \
       --limit 0 \
       --output_file_path benchmark_runs/smartwatch_match_baseline_1.json
   ```

2. **Create or update the sampling config** in `config/match/`. Point `output_results_filepath` to the results JSON:
   ```json
   {
     "train": {
       "output_results_filepath": "benchmark_runs/smartwatch_match_baseline_1.json",
       "fraction_tp": 0.32, "fraction_fp": 0.38, "fraction_fn": 0.38, "fraction_tn": 0.32
     },
     "validation": {
       "output_results_filepath": "benchmark_runs/smartwatch_match_baseline_1.json",
       "fraction_tp": 0.33, "fraction_fp": 0.32, "fraction_fn": 0.32, "fraction_tn": 0.33
     },
     "test": {
       "output_results_filepath": "benchmark_runs/smartwatch_match_baseline_1.json",
       "fraction_tp": 0.35, "fraction_fp": 0.30, "fraction_fn": 0.30, "fraction_tn": 0.35
     }
   }
   ```

3. **Run the sampler:**
   ```bash
   .venv/bin/python scripts/sampling/sample_data_by_confusion_matrix.py \
       --config_file config/match/sampling_config_confusion_matrix_1.json \
       --base_data_dir data/images/smartwatch \
       --output_dir data/sampled/match/smartwatch_cm_1 \
       --user_defined_category smartwatch_cm_1 \
       --master_csv data/images/smartwatch/smartwatch_mapping.csv \
       --positive_class match
   ```

4. **For mismatch sampling**, use the mismatch pipeline results and `--positive_class mismatch`:
   ```bash
   .venv/bin/python scripts/sampling/sample_data_by_confusion_matrix.py \
       --config_file config/mismatch/sampling_config_confusion_matrix_1.json \
       --base_data_dir data/images/smartwatch \
       --output_dir data/sampled/mismatch/smartwatch_cm_1 \
       --user_defined_category smartwatch_cm_1 \
       --master_csv data/images/smartwatch/smartwatch_mapping.csv \
       --positive_class mismatch
   ```

**Notes:**
- `--base_data_dir` should point directly to the category folder containing `reference_images/` and `images/`
- `--master_csv` is the same mapping CSV used in label-based sampling
- The results JSON must contain `individual_results` with `id`, `ground_truth`, and `model_output` fields
- If many items in the results file are API errors (e.g., 429 rate limits), they are classified as "Unknown" predictions and placed in the TN quadrant. Consider re-running the pipeline with `--workers 1` for cleaner results

---

## Choosing a Strategy

| Scenario | Strategy | Rationale |
|----------|----------|-----------|
| Initial dataset split | **Label-based** | No model predictions needed; ensures balanced GT label representation |
| Post-baseline optimization | **Confusion-matrix** | Oversample error quadrants (FP/FN) to focus GEPA on the model's weaknesses |
| Precision-focused training | **Confusion-matrix** with high `fraction_fp` | Forces optimizer to see more false positives, reducing FP rate |
| Recall-focused training | **Confusion-matrix** with high `fraction_fn` | Forces optimizer to see more false negatives, reducing FN rate |

---

## Output Structure

Both sampling scripts produce identical output layouts:

```
output_dir/
├── train/
│   ├── <name>_mapping.csv
│   ├── <name>_mapping.json
│   ├── reference_images/
│   │   └── <image files>
│   └── images/
│       └── <image files>
├── validation/
│   └── (same structure)
└── test/
    └── (same structure)
```

---

## Typical Workflow

1. **Start with label-based sampling** to create an initial balanced split
2. **Run baseline evaluation** with the base prompt on the train split
3. **Use confusion-matrix sampling** on the baseline results to create error-focused splits
4. **Run GEPA optimization** on the CM-sampled train split
5. **Evaluate** on the held-out test split (which was never seen during optimization)

---

## Config Directory Layout

```
config/
├── label/
│   └── sampling_config_label_1.json             # Label-based sampling
├── match/
│   └── sampling_config_confusion_matrix_1.json   # CM sampling, positive_class=match
└── mismatch/
    └── sampling_config_confusion_matrix_1.json   # CM sampling, positive_class=mismatch
