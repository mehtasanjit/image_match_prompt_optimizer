# Pipeline

The evaluation pipeline compares product images using a vision model and scores predictions against ground truth via Cloud Function metrics.

**Script:** `scripts/pipeline/run_binary_match_pipeline_with_eval.py`

The same script is used for both Match and Mismatch evaluation — the difference is which system prompt and which Cloud Function metric you provide.

---

## How It Works

For each image pair in the mapping CSV:

1. **Load images** — reads reference image from `reference_images/` and candidate image from `images/`
2. **Construct prompt** — assembles the system prompt + product category + images into a Gemini API call
3. **Call model** — sends to Gemini and captures the response JSON, token usage, and latency
4. **Score** — sends the model response + ground truth to the evaluation Cloud Function, which returns a score
5. **Aggregate** — collects all per-item results and produces a summary JSON

### Optional: Inconclusive Pre-Filter

If `--inconclusive_system_prompt` is provided, the pipeline runs a two-step process:
1. First call with the inconclusive prompt — if the model returns `"Inconclusive"`, the item is recorded as Inconclusive and skips step 2
2. Second call with the main system prompt — only runs if step 1 did not return Inconclusive

---

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes | — | Directory containing `reference_images/`, `images/`, and optionally the mapping CSV |
| `--system_prompt` | Yes | — | Path to the system prompt `.txt` file |
| `--eval_cloud_function_name` | Yes | — | Name of the deployed Cloud Function metric |
| `--project` | Yes | — | GCP Project ID |
| `--location` | Yes | — | GCP location for Gemini API (e.g., `us-central1`) |
| `--eval_cloud_function_location` | Yes | — | GCP location where the Cloud Function is deployed |
| `--category` | No | `None` | Product category name. Used in the model prompt and for auto-resolving the mapping CSV as `<data_dir>/<category>_mapping.csv`. If omitted, `--mapping_csv` is required and the prompt shows "Not Specified" |
| `--mapping_csv` | No | `<data_dir>/<category>_mapping.csv` | Path to mapping CSV. Required columns: `id`, `ground_truth`, `reference_image_filename`, `image_filename` |
| `--model` | No | `gemini-2.5-flash` | Model name |
| `--temperature` | No | `0.0` | Sampling temperature |
| `--top_p` | No | `0.95` | Top-p sampling |
| `--thinking_budget` | No | `0` | Thinking budget (Gemini 2.x models) |
| `--thinking_level` | No | `None` | Thinking level for Gemini 3 Preview (e.g., `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`) |
| `--limit` | No | `5` | Number of items to evaluate. Use `0` for all |
| `--output_file_path` | No | `None` | Path to save the summary JSON. Directories are created automatically |
| `--workers` | No | `1` | Number of concurrent workers for parallel evaluation |
| `--few_shot_examples_file` | No | `None` | Path to JSON file with few-shot examples |
| `--inconclusive_system_prompt` | No | `None` | Path to inconclusive pre-filter prompt |

---

## Running the Match Pipeline

Determines whether the candidate image shows the **same product** as the reference image.

**Outputs:** `Match` or `Not_Match` (or `Inconclusive` with image quality pre-check prompt)

### Minimal Example

```bash
.venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir data/sampled/smartwatch_1/train \
    --system_prompt prompts/binary_match_or_not.txt \
    --eval_cloud_function_name img_match_metric_match_only_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_match_guarded_1.json
```

### With Inconclusive Pre-Filter

```bash
.venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir data/sampled/smartwatch_1/train \
    --system_prompt prompts/binary_match_or_not.txt \
    --inconclusive_system_prompt prompts/binary_match_or_not_with_image_quality_precheck.txt \
    --eval_cloud_function_name img_match_metric_match_only_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_match_prefilter_1.json
```

### With Explicit CSV and No Category

```bash
.venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --mapping_csv data/sampled/smartwatch_1/train/smartwatch_1_mapping.csv \
    --data_dir data/sampled/smartwatch_1/train \
    --system_prompt prompts/binary_match_or_not.txt \
    --eval_cloud_function_name img_match_metric_match_only_moderate \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_match_moderate_1.json
```

In this case, "Product Category" in the model prompt will show "Not Specified".

---

## Running the Mismatch Pipeline

Determines whether the candidate image shows a **clearly different product**. Used as Stage 2 after items are classified as `Not_Match` in Stage 1.

**Outputs:** `Mismatch` or `Not_Mismatch` (or `Inconclusive` with pre-check prompt)

### Example

```bash
.venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir data/sampled/smartwatch_1/train \
    --system_prompt prompts/binary_mismatch_or_not.txt \
    --eval_cloud_function_name img_match_metric_mismatch_only_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_mismatch_guarded_1.json
```

### With Image Quality Pre-Check

```bash
.venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir data/sampled/smartwatch_1/train \
    --system_prompt prompts/binary_mismatch_or_not.txt \
    --inconclusive_system_prompt prompts/binary_mismatch_or_not_with_image_quality_precheck.txt \
    --eval_cloud_function_name img_match_metric_mismatch_only_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_mismatch_prefilter_1.json
```

---

## Running the Multi-Step Pipeline

**Script:** `scripts/pipeline/run_multi_step_pipeline_with_eval.py`

Runs two binary prompts in sequence to produce a 3-class output (Match / Mismatch / Inconclusive). This is useful when you want a single pipeline to produce all three labels without running match and mismatch separately.

### How It Works

**Default flow** (`--first_step match`):
1. Run match prompt → if `Match` → final = **Match**, DONE
2. Otherwise → run mismatch prompt → if `Mismatch` → final = **Mismatch**
3. Otherwise → final = **Inconclusive**

**Reversed flow** (`--first_step mismatch`):
1. Run mismatch prompt → if `Mismatch` → final = **Mismatch**, DONE
2. Otherwise → run match prompt → if `Match` → final = **Match**
3. Otherwise → final = **Inconclusive**

Items resolved at step 1 save one model call (latency and cost savings).

### Arguments

All arguments from the binary pipeline apply, plus:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--system_prompt_match` | Yes | — | Path to match system prompt |
| `--system_prompt_mismatch` | Yes | — | Path to mismatch system prompt |
| `--first_step` | No | `match` | Which prompt runs first (`match` or `mismatch`) |
| `--category` | No | `None` | Product category name. Used in the prompt and for auto-resolving `<data_dir>/<category>_mapping.csv` |
| `--mapping_csv` | No | `<data_dir>/<category>_mapping.csv` | Explicit path to mapping CSV. Required if `--category` is omitted |
| `--data_dir` | Yes | — | Directory with `reference_images/`, `images/`, and mapping CSV |
| `--eval_cloud_function_name` | Yes | — | Cloud Function for 3-class evaluation |
| `--project` | Yes | — | GCP Project ID |
| `--location` | Yes | — | GCP location for Gemini API |
| `--eval_cloud_function_location` | Yes | — | GCP location for Cloud Function |
| `--model` | No | `gemini-2.5-flash` | Model name |
| `--temperature` | No | `0.0` | Sampling temperature |
| `--top_p` | No | `0.95` | Top-p sampling |
| `--thinking_budget` | No | `0` | Thinking budget (Gemini 2.5) |
| `--thinking_level` | No | `None` | Thinking level (Gemini 3) |
| `--limit` | No | `5` | Number of items (0 = all) |
| `--output_file_path` | No | `None` | Output JSON path |
| `--workers` | No | `1` | Concurrent workers |

### Example: Match-First

```bash
.venv/bin/python scripts/pipeline/run_multi_step_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir data/sampled/smartwatch_1/test \
    --system_prompt_match prompts/binary_match_or_not.txt \
    --system_prompt_mismatch prompts/binary_mismatch_or_not.txt \
    --first_step match \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_multi_step_1.json
```

### Example: With Explicit CSV (No Category)

```bash
.venv/bin/python scripts/pipeline/run_multi_step_pipeline_with_eval.py \
    --mapping_csv data/sampled/smartwatch_1/test/smartwatch_1_mapping.csv \
    --data_dir data/sampled/smartwatch_1/test \
    --system_prompt_match prompts/binary_match_or_not.txt \
    --system_prompt_mismatch prompts/binary_mismatch_or_not.txt \
    --first_step mismatch \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location us-central1 \
    --eval_cloud_function_location us-central1 \
    --limit 0 \
    --output_file_path benchmark_runs/smartwatch_multi_step_2.json
```

### Output Format

The output JSON includes per-item results with multi-step metadata:

```json
{
  "config": { ... },
  "aggregated_metrics": {
    "n_total": 100,
    "accuracy": 0.85,
    "step1_resolved_count": 72,
    "step2_resolved_count": 28,
    "metrics_per_class": {
      "Match": { "precision": 0.92, "recall": 0.88, "f1": 0.90 },
      "Mismatch": { "precision": 0.85, "recall": 0.80, "f1": 0.82 },
      "Inconclusive": { "precision": 0.70, "recall": 0.75, "f1": 0.72 }
    },
    "confusion_matrix": { ... }
  },
  "results": [
    {
      "id": "uuid",
      "ground_truth": "Match",
      "multi_step": {
        "first_step": "match",
        "steps_used": 1,
        "step1_label": "match",
        "final_label": "Match"
      },
      "latency_sec": 2.1,
      "telemetry": { ... }
    }
  ]
}
```

---

## Match vs Mismatch: What Changes

| | Match Pipeline | Mismatch Pipeline |
|---|---|---|
| **System prompt** | `binary_match_or_not.txt` | `binary_mismatch_or_not.txt` |
| **Pre-check prompt** | `binary_match_or_not_with_image_quality_precheck.txt` | `binary_mismatch_or_not_with_image_quality_precheck.txt` |
| **CF metric** | `img_match_metric_match_only_*` | `img_match_metric_mismatch_only_*` |
| **Score key returned** | `match_score` | `mismatch_score` |
| **Model output classes** | Match, Not_Match, (Inconclusive) | Mismatch, Not_Mismatch, (Inconclusive) |

Everything else — the script, data format, image loading, model calling — is identical.

---

## Assumptions and Defaults

- **Mapping CSV resolution:** If `--mapping_csv` is not provided, the script looks for `<data_dir>/<category>_mapping.csv`
- **Category in prompt:** Always sent to the model. Defaults to `"Not Specified"` if `--category` is omitted
- **Score key:** The pipeline reads `match_score` from the CF response. For mismatch metrics, the CF returns `mismatch_score` — this means the pipeline currently reads the match key. If running mismatch evaluation standalone, the score in the output JSON will be `0.0` unless the mismatch CF also returns `match_score`. This is acceptable because GEPA reads the score from the CF directly, not through this pipeline
- **Output directories:** Created automatically if they don't exist
- **Model default:** `gemini-2.5-flash` with `temperature=0.0`, `top_p=0.95`, `thinking_budget=0`
- **Limit default:** `5` items (for quick testing). Use `--limit 0` for full evaluation
- **Workers default:** `1` (sequential). Increase for parallel evaluation but watch API rate limits
- **Cloud Function URL format:** `https://<eval_cloud_function_location>-<project>.cloudfunctions.net/<eval_cloud_function_name>`
- **Image loading:** Images are opened with PIL, converted to PNG bytes via base64 round-trip
- **JSON validation:** Model output is checked for valid JSON locally before scoring

---

## Output Format

The summary JSON (if `--output_file_path` is provided) contains:

```json
{
  "config": { ... },           // Run configuration (all CLI args)
  "individual_results": [      // Per-item results
    {
      "id": "uuid",
      "ground_truth": "Match",
      "model_output": "{...}",
      "is_valid_json": true,
      "score": 1.0,
      "eval_payload": { "match_score": 1.0 },
      "latency_sec": 2.3,
      "telemetry": {
        "prompt_token_count": 1500,
        "candidates_token_count": 120,
        ...
      },
      "inconclusive_prefilter": false
    },
    ...
  ]
}
```

---

## Prerequisites

1. **Deployed Cloud Functions** — run the deploy scripts first (see [metrics.md](metrics.md))
2. **Sampled data** — create train/validation/test splits (see [sampling.md](sampling.md))
3. **System prompts** — in `prompts/` directory
4. **GCP authentication** — `gcloud auth login` and appropriate project permissions
5. **Python dependencies** — `google-genai`, `pandas`, `Pillow`, `requests`

---

## Post-Optimization Scripts

After running GEPA grid optimization, three utility scripts in `scripts/post_optimization/` help with analysis, extraction, and blending:

### Precision-Recall Tradeoff Analysis

Scans all grid run JSON files in a folder, groups cells by validation precision brackets, and picks the best recall at each precision level. Also recommends the most generalizable cell (smallest precision spread across train/val/test).

```bash
./venv/bin/python scripts/post_optimization/analyze_grid_precision_recall_tradeoff.py \
    --input_dir ./grid_runs/match/smartwatch_split \
    --verbose \
    --output_json ./grid_runs/match/smartwatch_split/tradeoff.json
```

### Extract Prompt from Grid Run

Searches grid result JSON files for a cell matching a CF name suffix and iteration count, then saves the optimized prompt to a file.

```bash
./venv/bin/python scripts/post_optimization/extract_prompt_from_grid_run.py \
    --grid_run_folder ./grid_runs/match/smartwatch_split \
    --cf_key guarded \
    --num_iterations 16 \
    --output_file ./prompts/optimized_match_guarded_16.txt
```

### Standalone Prompt Blending

Takes multiple grid output JSON files, extracts final prompts from each, and blends them using `gemini-3.1-pro-preview` with `thinking_level=HIGH`.

```bash
./venv/bin/python scripts/post_optimization/blend_prompts.py \
    --input_files ./grid_runs/match/run_1.json ./grid_runs/match/run_2.json \
    --output_prompt ./prompts/blended_match.txt \
    --project my-gcp-project
```

See [user_guide.md](user_guide.md) for the full end-to-end workflow (Steps 7-9).
