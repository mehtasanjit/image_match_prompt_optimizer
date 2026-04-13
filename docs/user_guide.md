# User Guide: From Setup to Running GEPA Grid Optimization

This guide walks through every step from a fresh clone to running automated prompt optimization with evaluation.

---

## Prerequisites

- **GCP Project** with Vertex AI API enabled
- **Python 3.11+** with a virtual environment
- **gcloud CLI** authenticated (`gcloud auth application-default login`)

---

## Step 1: Set Up the Environment

```bash
# Clone the repo and cd into it
cd <repo_root>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlflow google-genai litellm gepa Pillow pandas requests
```

---

## Step 2: Prepare Your Data

Each product category needs:
1. A directory with images organized as:
   ```
   data/images/<category>/
   ├── <category>_mapping.csv
   ├── reference_images/
   │   ├── uuid1.jpg
   │   └── uuid2.jpg
   └── images/
       ├── uuid1.jpg
       └── uuid2.jpg
   ```

2. A mapping CSV with these columns:
   ```csv
   id,category,reference_image_filename,image_filename,ground_truth
   abc123,smartwatch,abc123.jpg,abc123.jpg,Match
   def456,smartwatch,def456.jpg,def456.jpg,Mismatch
   ghi789,smartwatch,ghi789.jpg,ghi789.jpg,Inconclusive
   ```

See [data.md](data.md) for full details.

---

## Step 3: Create Train/Validation/Test Splits

There are two paths for creating data splits, depending on whether you have prior benchmark results:

### Path A: Label-based sampling (no prior benchmark needed)

Splits data by ground truth label distribution. Start here if this is your first run.

Create a sampling config JSON (e.g. `config/label/smartwatch_split.json`):

```json
{
  "train": { "Match": 0.6, "Mismatch": 0.6, "Inconclusive": 0.6 },
  "validation": { "Match": 0.2, "Mismatch": 0.2, "Inconclusive": 0.2 },
  "test": { "Match": 0.2, "Mismatch": 0.2, "Inconclusive": 0.2 }
}
```

Then run the label-based sampler:

```bash
./venv/bin/python scripts/sampling/sample_data_by_label.py \
    --config_file ./config/label/smartwatch_split.json \
    --mapping_csv ./data/images/smartwatch/smartwatch_mapping.csv \
    --image_dir ./data/images/smartwatch \
    --output_dir ./data/sampled/label/smartwatch_split \
    --output_name smartwatch_split
```

### Path B: Confusion-matrix sampling (requires prior benchmark)

Samples by prediction outcome (TP/FP/FN/TN) to focus training on the model's actual failure modes. This requires:
1. First deploy Cloud Functions (Step 4 below)
2. Run a benchmark pipeline on the full dataset to get per-item predictions
3. Then use confusion-matrix sampling to create targeted splits

```bash
./venv/bin/python scripts/sampling/sample_data_by_confusion_matrix.py \
    --config_file ./config/match/smartwatch_cm_split.json \
    --base_data_dir ./data/images/smartwatch \
    --output_dir ./data/sampled/match/smartwatch_cm_split \
    --user_defined_category smartwatch_cm_split \
    --master_csv ./data/images/smartwatch/smartwatch_mapping.csv
```

**When to use which:**
- **Path A** (label-based): First optimization run, general-purpose splits
- **Path B** (confusion-matrix): Subsequent runs, when you want to focus GEPA on specific failure patterns (e.g. oversample false positives)

Both paths create the same output structure:
```
data/sampled/<method>/<split_name>/
├── train/
│   ├── <split_name>_mapping.csv
│   ├── reference_images/
│   └── images/
├── validation/
│   └── ...
└── test/
    └── ...
```

See [sampling.md](sampling.md) for full details on both methods.

---

## Step 4: Deploy Evaluation Cloud Functions

Deploy the weighted cost functions as GCP Cloud Functions. Each metric returns a score key:

| Stage | Cloud Function | Score Key |
|-------|---------------|-----------|
| Match | `img_match_weighted_guarded` | `match_score` |
| Mismatch | `img_mismatch_weighted_guarded` | `mismatch_score` |

See [cost_function_design.md](cost_function_design.md) for the scoring logic and [metrics.md](metrics.md) for all available metrics.

---

## Step 5: Verify the Pipeline (Optional but Recommended)

Before running GEPA optimization, verify the eval pipeline works end-to-end on a small sample:

```bash
./venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch_split \
    --data_dir ./data/sampled/label/smartwatch_split/validation \
    --system_prompt ./prompts/binary_match_or_not.txt \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location global \
    --eval_cloud_function_location us-central1 \
    --model gemini-3-flash-preview \
    --limit 5 \
    --workers 3
```

This should produce per-item results and aggregated metrics. If it works, your Cloud Functions, data, and prompts are correctly configured.

---

## Step 6: Run GEPA Grid Optimization

### Basic Grid Run (Match)

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --location global \
    --target_model gemini-3-flash-preview \
    --temperature 1.0 \
    --top_p 0.42 \
    --critic_model gemini-3.1-pro-preview \
    --critic_model_location global \
    --critic_reasoning_effort high \
    --eval_cf_names img_match_weighted_guarded \
    --eval_score_key match_score \
    --num_iterations 3 \
    --category smartwatch_split \
    --data_dir ./data/sampled/label/smartwatch_split/train \
    --eval_data_dir ./data/sampled/label/smartwatch_split/validation \
    --test_data_dir ./data/sampled/label/smartwatch_split/test \
    --initial_prompt ./prompts/binary_match_or_not.txt \
    --subsample_fraction 0.3 \
    --eval_workers 10 \
    --output_file ./grid_runs/match/smartwatch_split/run_1.json
```

### Using Explicit Mapping CSVs (No Category)

If your mapping CSV names don't match the `<category>_mapping.csv` convention:

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --location global \
    --target_model gemini-3-flash-preview \
    --eval_cf_names img_match_weighted_guarded \
    --eval_score_key match_score \
    --num_iterations 3 \
    --data_dir ./data/sampled/custom/train \
    --mapping_csv ./data/sampled/custom/train/my_train_data.csv \
    --eval_data_dir ./data/sampled/custom/validation \
    --eval_mapping_csv ./data/sampled/custom/validation/my_val_data.csv \
    --test_data_dir ./data/sampled/custom/test \
    --test_mapping_csv ./data/sampled/custom/test/my_test_data.csv \
    --initial_prompt ./prompts/binary_match_or_not.txt \
    --output_file ./grid_runs/match/custom/run_1.json
```

### Grid with Multiple CFs and Iterations

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --location global \
    --target_model gemini-3-flash-preview \
    --eval_cf_names img_match_weighted_guarded,img_match_weighted_balanced,img_match_weighted_aggressive \
    --eval_score_key match_score \
    --num_iterations 5,10,15 \
    --category smartwatch_split \
    --data_dir ./data/sampled/label/smartwatch_split/train \
    --eval_data_dir ./data/sampled/label/smartwatch_split/validation \
    --test_data_dir ./data/sampled/label/smartwatch_split/test \
    --initial_prompt ./prompts/binary_match_or_not.txt \
    --eval_workers 10 \
    --output_file ./grid_runs/match/smartwatch_split/grid_3x3.json
```

This runs 3 × 3 = 9 grid cells, each with full GEPA optimization + evaluation.

### Stepwise GEPA with Blend

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --location global \
    --target_model gemini-3-flash-preview \
    --eval_cf_names img_match_weighted_guarded \
    --eval_score_key match_score \
    --num_iterations 12 \
    --step_size 4 \
    --no_chaining \
    --subsample_fraction 0.5 \
    --category smartwatch_split \
    --data_dir ./data/sampled/label/smartwatch_split/train \
    --eval_data_dir ./data/sampled/label/smartwatch_split/validation \
    --initial_prompt ./prompts/binary_match_or_not.txt \
    --output_file ./grid_runs/match/smartwatch_split/stepwise_blend.json
```

---

## Step 7: Analyze Results

### 7a. Quick look at the output JSON

The output JSON (`grid_runs/match/.../run_1.json`) contains:

```json
{
  "grid_spec": [...],
  "best_by_eval_precision": { "grid_cell": {...}, "eval_match_precision": 0.95 },
  "best_by_eval_f1": { "grid_cell": {...}, "eval_match_f1": 0.82 },
  "results": [
    {
      "grid_cell": { "eval_cf_name": "...", "num_iterations": 3 },
      "gepa": {
        "initial_eval_score": 0.45,
        "final_eval_score": 0.72,
        "final_prompt": "... optimized prompt text ..."
      },
      "eval_train": { "aggregated_metrics": {...} },
      "eval_validation": { "aggregated_metrics": {...} },
      "eval_test": { "aggregated_metrics": {...} }
    }
  ]
}
```

Key metrics to look at:
- `eval_validation.aggregated_metrics.metrics_match.precision` — primary optimization target
- `eval_test.aggregated_metrics.metrics_match.precision` — generalization check
- Compare train vs validation precision for overfitting detection

### 7b. Precision-Recall tradeoff analysis

Use the analysis script to scan all grid run results in a folder, group them by validation precision brackets, and find the best recall at each precision level:

```bash
./venv/bin/python scripts/post_optimization/analyze_grid_precision_recall_tradeoff.py \
    --input_dir ./grid_runs/match/smartwatch_split \
    --verbose
```

This outputs:
- **Tradeoff table**: Best recall per precision bracket (0.95-1.00, 0.90-0.95, etc.)
- **All cells table** (`--verbose`): Every grid cell sorted by validation precision
- **CF summary** (`--verbose`): Aggregated stats per Cloud Function variant
- **Recommendation**: The cell with smallest precision spread across train/val/test (most generalizable)

Optionally save as JSON:
```bash
./venv/bin/python scripts/post_optimization/analyze_grid_precision_recall_tradeoff.py \
    --input_dir ./grid_runs/match/smartwatch_split \
    --output_json ./grid_runs/match/smartwatch_split/tradeoff_analysis.json
```

---

## Step 8: Extract the Best Prompt

Use the extraction script to pull the optimized prompt from grid results:

```bash
./venv/bin/python scripts/post_optimization/extract_prompt_from_grid_run.py \
    --grid_run_folder ./grid_runs/match/smartwatch_split \
    --cf_key guarded \
    --num_iterations 16 \
    --output_file ./prompts/optimized_smartwatch_match_guarded_16.txt
```

The `--cf_key` matches the Cloud Function name suffix (e.g. `guarded` matches `img_match_weighted_guarded`). If `--num_iterations` is omitted, all iteration counts are matched.

---

## Step 9: Blend Prompts from Multiple Runs (Optional)

This step is optional. If you ran multiple grid experiments (different seeds, CFs, or categories) and want to combine their best prompts into a single unified prompt, use the standalone blending script. It uses `gemini-3.1-pro-preview` via the Google GenAI SDK (Vertex AI) with `thinking_level=HIGH` — not through LiteLLM:

```bash
./venv/bin/python scripts/post_optimization/blend_prompts.py \
    --input_files \
        ./grid_runs/match/smartwatch_split/run_1.json \
        ./grid_runs/match/smartwatch_split/run_2.json \
    --output_prompt ./prompts/smartwatch_match_blended.txt \
    --project my-gcp-project
```

This:
1. Extracts final prompts + eval metrics from each grid JSON
2. Sends them to `gemini-3.1-pro-preview` (with `thinking_level=HIGH`) with the blend template
3. Saves the merged prompt

You can provide a custom blend template:
```bash
    --blend_template ./prompts/blend_prompts_template.txt
```

---

## Step 10: Use the Optimized Prompt

Run the optimized prompt on your full dataset:

```bash
./venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir ./data/images/smartwatch \
    --system_prompt ./prompts/optimized_smartwatch_match_guarded_16.txt \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location global \
    --eval_cloud_function_location us-central1 \
    --model gemini-3-flash-preview \
    --limit 0 \
    --workers 10
```

### Discarding Inconclusive Items Early (Optional)

If your dataset contains low-quality or ambiguous images that frequently produce `Inconclusive` results, you can enable a **two-step pipeline** using the image quality precheck prompts:

- `prompts/binary_match_or_not_with_image_quality_precheck.txt`
- `prompts/binary_mismatch_or_not_with_image_quality_precheck.txt`

When the `--inconclusive_system_prompt` flag is provided, the pipeline first runs the precheck prompt on each image pair. Items classified as `Inconclusive` at this stage are skipped entirely — they never reach the main classification step. This reduces noise, saves inference cost, and improves overall precision by filtering out items the model cannot confidently classify.

```bash
./venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir ./data/images/smartwatch \
    --system_prompt ./prompts/optimized_smartwatch_match_guarded_16.txt \
    --inconclusive_system_prompt ./prompts/binary_match_or_not_with_image_quality_precheck.txt \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location global \
    --eval_cloud_function_location us-central1 \
    --model gemini-3-flash-preview \
    --limit 0 \
    --workers 10
```

See [pipeline.md](pipeline.md) for details on the inconclusive pre-filter mechanism.

### Running the Multi-Step Pipeline (Optional)

Instead of running match and mismatch pipelines separately, you can use the **multi-step pipeline** to produce a 3-class output (Match / Mismatch / Inconclusive) in a single run. It chains two binary prompts sequentially:

1. Run first prompt → if positive class detected → done
2. Otherwise → run second prompt → if positive class detected → done
3. Otherwise → Inconclusive

Items resolved at step 1 save one model call.

```bash
./venv/bin/python scripts/pipeline/run_multi_step_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir ./data/images/smartwatch \
    --system_prompt_match ./prompts/optimized_smartwatch_match_guarded_16.txt \
    --system_prompt_mismatch ./prompts/binary_mismatch_or_not.txt \
    --first_step match \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --model gemini-3-flash-preview \
    --limit 0 \
    --workers 10 \
    --output_file_path benchmark_runs/smartwatch_multi_step.json
```

If your mapping CSV doesn't follow the `<category>_mapping.csv` convention, use `--mapping_csv` explicitly:

```bash
./venv/bin/python scripts/pipeline/run_multi_step_pipeline_with_eval.py \
    --mapping_csv ./data/images/smartwatch/my_custom_mapping.csv \
    --data_dir ./data/images/smartwatch \
    --system_prompt_match ./prompts/optimized_smartwatch_match_guarded_16.txt \
    --system_prompt_mismatch ./prompts/binary_mismatch_or_not.txt \
    --first_step match \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --model gemini-3-flash-preview \
    --limit 0 \
    --workers 10
```

See [pipeline.md](pipeline.md) for the full multi-step pipeline documentation.

---

## CLI Quick Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--project` | GCP project ID | (required) |
| `--location` | GCP location for target model | `us-central1` |
| `--target_model` | Vision model to optimize | `gemini-2.5-flash` |
| `--temperature` | Sampling temperature (used for training AND evaluation) | `0.0` |
| `--top_p` | Top-p sampling (used for training AND evaluation) | `0.95` |
| `--thinking_budget` | Thinking token budget for target model (Gemini 2.5 family only; **not applicable for Gemini 3 models** which use `thinking_level` instead) | `0` |
| `--critic_model` | Reflection model for GEPA | `gemini-3.1-pro-preview` |
| `--critic_model_location` | Location for critic model | `global` |
| `--critic_reasoning_effort` | Thinking level: `high`, `low`, or `""` | `high` |
| `--category` | Auto-resolves mapping CSVs as `<dir>/<category>_mapping.csv` | (optional) |
| `--mapping_csv` | Explicit path to training mapping CSV | (optional) |
| `--eval_mapping_csv` | Explicit path to validation mapping CSV | (optional) |
| `--test_mapping_csv` | Explicit path to test mapping CSV | (optional) |
| `--random_seed` | Random seed for sub-sampling. **Change this between runs** to ensure different sub-samples are picked (see note below) | `42` |
| `--subsample_fraction` | Fraction of training data for GEPA (0.0-1.0) | `1.0` |
| `--step_size` | Stepwise GEPA iterations per step (0 = standard) | `0` |
| `--no_chaining` | Blend mode: all steps independent, then blend | (flag) |
| `--eval_workers` | Concurrent workers for eval pipeline | `10` |
| `--num_iterations` | Comma-separated iteration counts for grid | (required) |
| `--eval_cf_names` | Comma-separated Cloud Function names for grid | (required) |
| `--reflection_prompt_template` | Custom reflection prompt for the critic model (see below) | (optional) |

---

## Random Seed Behavior

The `--random_seed` controls which sub-sample of training data is used for GEPA optimization. **You should change the seed between runs** if you want different sub-samples to be selected.

- With `--subsample_fraction 0.3 --random_seed 42`, the same 30% of data is always selected
- Change to `--random_seed 734` to get a different 30% sub-sample
- In **stepwise mode**, each step automatically gets a different seed derived from the base seed: `step_seed = random_seed + step_idx * 7`. This ensures different sub-samples per step even within a single run, regardless of whether chaining or blend mode is used

---

## Custom Reflection Prompt Template

The GEPA critic model uses a **reflection prompt template** to analyze failures and propose prompt improvements. A precision-focused template is provided at `prompts/gepa_reflection_prompt_match_precision_focused.txt`.

The template must contain these placeholders (replaced by GEPA at runtime):
- `<curr_param>` — the current system prompt being optimized
- `<side_info>` — real examples with predictions, ground truth, and scores

**Important**: GEPA validates for exactly these placeholder names and will raise a `ValueError` if they are missing.

Usage:
```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    ... \
    --reflection_prompt_template ./prompts/gepa_reflection_prompt_match_precision_focused.txt
```

If not provided, GEPA uses its default built-in reflection prompt.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'mlflow_gepa'` | Running from wrong directory | Run from repo root |
| `Mapping CSV not found` | Wrong `--category` or missing `--mapping_csv` | Check CSV filename matches `<category>_mapping.csv` |
| `Publisher Model not found` | Wrong model location | Set `--location global` for Gemini 3 models |
| `No module named 'gepa.lm'` | gepa < 0.1.1 | ThinkingOptimizer has a fallback; works with any gepa version |
| `text fields too large` | MLflow autolog capturing image bytes | Already handled by `predict.py` disabling autolog |