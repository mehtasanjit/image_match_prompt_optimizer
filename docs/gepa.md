# GEPA Prompt Optimization

## Overview

GEPA (Genetic-Pareto) is MLflow's automated prompt optimization algorithm. It uses iterative mutation, reflection, and Pareto-aware candidate selection to improve prompts. A reflection model (LLM) analyzes system behavior on failures and proposes improvements, while Pareto selection ensures optimization across multiple objectives simultaneously.

This project uses GEPA to iteratively refine system prompts for product image comparison, maximizing precision on a custom weighted cost function deployed as a GCP Cloud Function.

The optimization loop works as follows:

1. **Predict**: The target model (e.g. `gemini-2.5-flash`) processes each training example using the current prompt — comparing a reference image against a candidate image and outputting a JSON classification.
2. **Score**: Each prediction is evaluated by a Cloud Function scorer that computes a weighted score (e.g. `match_score` or `mismatch_score`).
3. **Reflect**: A critic/reflection model (e.g. `gemini-3.1-pro-preview`) analyzes failures and proposes prompt mutations.
4. **Select**: Pareto-aware candidate selection picks the best prompt variant.
5. **Update**: The improved prompt is registered and the cycle repeats.

## Architecture

```
scripts/
├── mlflow_gepa/                                # GEPA optimization
│   ├── __init__.py
│   ├── config.py                               # GEPAConfig dataclass
│   ├── data_loader.py                          # CSV → GEPA DataFrame
│   ├── predict.py                              # predict_fn for GEPA (vision model call)
│   ├── scorer.py                               # MLflow scorer wrapping Cloud Function
│   ├── thinking_optimizer.py                   # Subclass enabling thinking_level=HIGH
│   ├── run_gepa.py                             # Basic GEPA orchestrator
│   ├── run_gepa_stepwise.py                    # Stepwise GEPA (chaining + blend)
│   ├── run_gepa_binary_match_grid.py           # Grid search (match)
│   └── run_gepa_binary_mismatch_grid.py        # Grid search (mismatch)
│
└── post_optimization/                          # Post-optimization analysis & extraction
    ├── analyze_grid_precision_recall_tradeoff.py  # P-R tradeoff table + recommendation
    ├── extract_prompt_from_grid_run.py            # Extract prompt by CF key + iterations
    └── blend_prompts.py                           # Standalone prompt blending across runs
```

## Three Optimization Modes

### 1. Basic GEPA (`run_gepa.py`)

Standard `mlflow.genai.optimize_prompts()` — runs N iterations on the full training set with a single cost function.

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa.py \
    --project my-gcp-project \
    --eval_cf_name img_match_weighted_guarded \
    --eval_score_key match_score \
    --category smartwatch \
    --data_dir data/images/smartwatch \
    --initial_prompt prompts/binary_match_or_not.txt \
    --num_iterations 16
```

### 2. Stepwise GEPA (`run_gepa_stepwise.py`)

Divides total iterations into smaller steps, each on a different sub-sample of training data. Supports two sub-modes:

#### 2a. Chaining (default)
Output prompt from step K becomes input for step K+1. Analogous to SGD — sequential refinement with data diversity.

#### 2b. No-chaining / Blend (`--no_chaining`)
All steps start from the same initial prompt, producing K independent prompts. These are then **blended** by `gemini-3.1-pro-preview` (with `thinking_level=HIGH`) into a single merged prompt. Analogous to ensemble → distillation.

The blend step uses a template from `prompts/blend_prompts_template.txt` (match) or `prompts/blend_prompts_mismatch_template.txt` (mismatch) that instructs the blending model to:
- Preserve the overall prompt structure
- Merge instructions from all source prompts
- Prioritize precision-focused guardrails
- Never invent new content beyond what exists in the source prompts

#### 2c. Error-Focused Chaining (`--error_focused`)

A variant of chaining that uses **hard example mining between steps**. Instead of random sub-sampling, it runs the current prompt on all training data after each step, classifies predictions as TP/FP/FN/TN, and enriches the next step's training sample with error cases.

**Flow per step:**
1. **Step 0**: Random sub-sample (no prior predictions exist)
2. **Steps 1+**: Run predict_fn on full training data → classify each item as TP/FP/FN/TN → subsample with configurable quadrant fractions

Default fractions: 100% of FP, 100% of FN, 30% of TP, 30% of TN — focusing GEPA on the current failure modes.

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --eval_cf_names img_match_weighted_guarded \
    --eval_score_key match_score \
    --num_iterations 16 \
    --step_size 4 \
    --error_focused \
    --fp_fraction 1.0 --fn_fraction 1.0 \
    --tp_fraction 0.3 --tn_fraction 0.3 \
    --category smartwatch \
    --data_dir data/images/smartwatch/train \
    --eval_data_dir data/images/smartwatch/eval \
    --initial_prompt prompts/binary_match_or_not.txt \
    --output_file grid_runs/match/smartwatch/error_focused.json
```

**Key differences from standard stepwise:**

| | Standard stepwise | Error-focused |
|---|---|---|
| Between-step sampling | Random (different seed per step) | TP/FP/FN/TN-aware |
| What changes per step | Random seed → different subset | Error distribution → different failures |
| Cost | Low (no inter-step eval) | Higher (full predict pass between steps) |
| Signal quality | Random diversity | Targeted error focus |

**Incompatibility**: `--error_focused` requires chaining (`--no_chaining` is not allowed). Error-focused mode depends on the prompt changing between steps — with no-chaining, all steps use the same initial prompt, producing identical error distributions.

### 3. Grid Search (`run_gepa_binary_match_grid.py` / `run_gepa_binary_mismatch_grid.py`)

Runs GEPA across a Cartesian product of Cloud Function names × iteration counts, then evaluates each optimized prompt on held-out train/validation/test splits using the full pipeline.

```bash
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --eval_cf_names img_match_weighted_guarded,img_match_weighted_balanced \
    --eval_score_key match_score \
    --num_iterations 12,16 \
    --category smartwatch \
    --data_dir data/images/smartwatch/train \
    --eval_data_dir data/images/smartwatch/eval \
    --eval_category smartwatch \
    --initial_prompt prompts/binary_match_or_not.txt \
    --step_size 4 --no_chaining \
    --eval_workers 10 \
    --output_file grid_runs/match/smartwatch/results.json
```

The grid runner:
- Supports all three modes (basic, stepwise chaining, stepwise blend) via `--step_size` and `--no_chaining` flags
- Evaluates each optimized prompt on train, validation, and optional test splits
- Tracks best cells by match/mismatch precision, F1, and GEPA score
- Writes incremental results after each grid cell

## Key Components

### GEPAConfig (`config.py`)

Centralizes all parameters: GCP project, target model, critic model, Cloud Function details, prompt paths, data paths, and optimizer settings.

Key fields:
- `category` — product category name (e.g. "smartwatch")
- `eval_score_key` — JSON key returned by the Cloud Function (`match_score` or `mismatch_score`)
- `mapping_csv` — explicit path to mapping CSV (optional; defaults to `{data_dir}/{category}_mapping.csv`)

### Data Loader (`data_loader.py`)

Data resolution is **category-centric**: every run requires either a `--category` name or an explicit `--mapping_csv` path.

| Method | Resolution | Example |
|--------|-----------|---------|
| **By category** | `<data_dir>/<category>_mapping.csv` | `--category smartwatch --data_dir ./data/images/smartwatch` |
| **By mapping CSV** | Exact path | `--mapping_csv ./data/sampled/split_1/train/split_1_mapping.csv` |

The mapping CSV must contain these columns:

```csv
id,category,reference_image_filename,image_filename,ground_truth
abc123,smartwatch,abc123.jpg,abc123.jpg,Match
def456,smartwatch,def456.jpg,def456.jpg,Mismatch
```

The data loader reads this CSV and builds a GEPA-compatible DataFrame with:
- `inputs`: `{"reference_image_path": "...", "image_path": "..."}`
- `outputs`: `{"ground_truth": "Match"}`

Images are resolved from `reference_images/` and `images/` subdirectories within `data_dir`.

For the grid runners, each data split independently resolves its mapping CSV:
- Training: `--category` or `--mapping_csv`
- Validation: `--eval_category` → `--category` fallback, or `--eval_mapping_csv`
- Test: `--test_category` → `--eval_category` → `--category`, or `--test_mapping_csv`
- Full: `--full_data_category` → `--category`, or `--full_data_mapping_csv`

See [data.md](data.md) for full data organization details.

### Predict Function (`predict.py`)

Builds a multimodal request with:
- System prompt loaded from MLflow prompt registry
- User content: Product Category + Reference Image + Candidate Image

Returns full JSON output (including reasoning) so the GEPA reflection model can analyze failure patterns.

**Thinking budget vs thinking level**: The `--thinking_budget` parameter (token count) applies only to Gemini 2.5 family models used as the target model. For Gemini 3 family models, thinking is controlled via `thinking_level` (HIGH/LOW) — this is configured separately for the critic model via `--critic_reasoning_effort`.

**Important**: Disables `mlflow.gemini.autolog()` to prevent image bytes from being captured in traces, which would cause "text fields too large" errors during reflection.

### Scorer (`scorer.py`)

Wraps Cloud Function calls behind MLflow's `@scorer` decorator:
1. Extracts `product_match` label from model JSON output
2. Normalizes to Match/Mismatch/Inconclusive
3. Posts `{response, target}` to the Cloud Function
4. Returns the numeric score from the specified score key

### Reflection Model Provider

GEPA's reflection model is specified via a URI in the format `<provider>:/<model>`. MLflow natively supports multiple providers including direct Vertex AI access:

| Provider | URI Format | Required Env Vars |
|----------|-----------|-------------------|
| **Vertex AI** (direct) | `vertex_ai:/gemini-3.1-pro-preview` | `VERTEX_PROJECT`, `VERTEX_LOCATION` |
| **OpenAI** (or LiteLLM proxy) | `openai:/gpt-4o` | `OPENAI_API_KEY`, `OPENAI_API_BASE` |
| **Gemini** (API key) | `gemini:/gemini-2.5-flash` | `GEMINI_API_KEY` |

**Recommended for GCP**: Use `vertex_ai:/` directly — no proxy needed:

```python
optimizer = GepaPromptOptimizer(
    reflection_model="vertex_ai:/gemini-3.1-pro-preview",
    max_metric_calls=100,
)
```

**Note:** `VERTEX_PROJECT` and `VERTEX_LOCATION` env vars are set automatically by `run_gepa.py` from `--project` and `--critic_model_location` (default `global`). These env vars are used **only for the critic/reflection model** (via MLflow's `vertex_ai:/` provider → LiteLLM). The target model's predict function (`predict.py`) uses its own `google.genai.Client` initialized with `config.project` and `config.location`, independent of these env vars.

**Alternative**: If using a LiteLLM proxy (e.g. for model routing), set:
- `OPENAI_API_KEY=sk-litellm-dummy`
- `OPENAI_API_BASE=http://localhost:4000/v1`
- Use `reflection_model="openai:/gemini-3.1-pro-preview"`

### Thinking Level for the Reflection Model

The reflection/critic model uses **`thinking_level=HIGH`** by default for deeper reasoning when analyzing failures and proposing prompt mutations. This is achieved via `ThinkingGepaPromptOptimizer` (`scripts/mlflow_gepa/thinking_optimizer.py`), a custom subclass of MLflow's `GepaPromptOptimizer`.

**Why a custom subclass?** MLflow's `GepaPromptOptimizer` only accepts the reflection model as a string URI (e.g. `"vertex_ai:/gemini-3.1-pro-preview"`) and provides no way to pass generation parameters like `thinking_level`. Under the hood, GEPA uses `gepa.lm.LM` which forwards `**kwargs` to `litellm.completion`. LiteLLM maps `reasoning_effort="high"` to Gemini 3's native `thinking_level=HIGH`.

**How the patch works:**

1. `ThinkingGepaPromptOptimizer` overrides the `optimize()` method
2. Before calling the parent's `optimize()`, it creates a `gepa.lm.LM` instance with `reasoning_effort="high"`
3. It patches `gepa.optimize()` to intercept the kwargs and replace the string-based `reflection_lm` with the thinking-enabled LM instance
4. The parent's `optimize()` runs normally, but GEPA now uses the thinking-enabled model for all reflection calls

```python
from mlflow_gepa.thinking_optimizer import ThinkingGepaPromptOptimizer

optimizer = ThinkingGepaPromptOptimizer(
    reflection_model="vertex_ai:/gemini-3.1-pro-preview",
    max_metric_calls=100,
    reasoning_effort="high",  # "high" (default), "low", or None to disable
)
```

To disable thinking, pass `reasoning_effort=None` — this falls back to standard `GepaPromptOptimizer` behavior.

## Score Keys

All match-stage metrics use `match_score`. All mismatch-stage metrics use `mismatch_score`. These are the JSON keys returned by the deployed Cloud Function scorers.

## Binary Classification Stages

GEPA optimization is run independently for each binary stage:

| Stage | Positive Class | Negative Class | Score Key | Prompt |
|-------|---------------|----------------|-----------|--------|
| Match | Match | Not_Match | `match_score` | `binary_match_or_not.txt` |
| Mismatch | Mismatch | Not_Mismatch | `mismatch_score` | `binary_mismatch_or_not.txt` |

See [binary_vs_multiclass_design.md](binary_vs_multiclass_design.md) for rationale on the two-stage design.

## Data Flow

### Single GEPA Run (`run_gepa.py`)

```
CSV (mapping)
    │
    ▼
data_loader.load_eval_data()  →  DataFrame[inputs, outputs]
    │
    ▼
mlflow.genai.optimize_prompts(
    predict_fn = predict.predict_fn,     # calls target model
    train_data = DataFrame,
    scorers   = [scorer.weighted_scorer], # calls Cloud Function
    optimizer = ThinkingGepaPromptOptimizer(
        reflection_model = "vertex_ai:/gemini-3.1-pro-preview",
        reasoning_effort = "high"        # thinking_level=HIGH
    )
)
    │
    ▼
Optimized prompt registered in MLflow
```

### Grid Runner Flow (`run_gepa_binary_match_grid.py`)

When a user runs the grid runner, the following happens end-to-end:

```
┌─────────────────────────────────────────────────────────────┐
│  1. PARSE & LOAD                                            │
│                                                             │
│  Parse CLI args → build grid (CF names × iteration counts)  │
│  Load train/eval/test splits from mapping CSVs              │
│  Compute effective_max_train from subsample_fraction        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. FOR EACH GRID CELL  (cf_name, num_iterations)           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2a. GEPA OPTIMIZATION                                │   │
│  │                                                       │   │
│  │  Build GEPAConfig with cell-specific params           │   │
│  │  Set VERTEX_PROJECT / VERTEX_LOCATION env vars        │   │
│  │                                                       │   │
│  │  If step_size > 0:                                    │   │
│  │    run_stepwise(config, step_size, chaining?)          │   │
│  │    → K sub-steps, each on different data subsample    │   │
│  │    → If no_chaining: blend K prompts with Gemini 3.1  │   │
│  │  Else:                                                │   │
│  │    run_gepa(config)                                   │   │
│  │    → Standard GEPA with all iterations at once        │   │
│  │                                                       │   │
│  │  Output: final_prompt_text + GEPA scores              │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2b. EVALUATE ON TRAIN SPLIT                           │   │
│  │                                                       │   │
│  │  For each item in train_data (concurrent workers):    │   │
│  │    process_item() → model prediction → CF scoring     │   │
│  │  summarize_eval_results() → precision/recall/F1       │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2c. EVALUATE ON VALIDATION SPLIT                      │   │
│  │                                                       │   │
│  │  Same as train eval, on held-out validation data      │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2d. EVALUATE ON TEST SPLIT (optional)                 │   │
│  │                                                       │   │
│  │  Same as above, if --test_data_dir provided           │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2e. WRITE INTERMEDIATE RESULTS                        │   │
│  │                                                       │   │
│  │  Append cell result to output JSON after each cell    │   │
│  │  Track best cells by precision, F1, GEPA score        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. FINAL OUTPUT (grid_results.json)                        │
│                                                             │
│  {                                                          │
│    "grid_spec": [...],                                      │
│    "best_by_eval_precision": { cell, precision, F1 },       │
│    "best_by_eval_f1": { cell, precision, F1 },              │
│    "best_by_gepa_final_score": { cell, score },             │
│    "results": [                                             │
│      {                                                      │
│        "grid_cell": { cf_name, score_key, iterations },     │
│        "gepa": { initial/final scores, prompt text },       │
│        "eval_train": { aggregated_metrics },                │
│        "eval_validation": { aggregated_metrics },           │
│        "eval_test": { aggregated_metrics }                  │
│      },                                                     │
│      ...                                                    │
│    ]                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- Each grid cell gets a unique MLflow prompt name (`{prompt_name}_{cf_name}_{iterations}`) to avoid registry collisions
- Intermediate results are written after each cell, so partial results are available even if a cell fails
- Failed cells log the error and continue to the next cell
- **Consistent temperature/top_p**: The same `--temperature` and `--top_p` values are used for GEPA training (predict_fn), and for all post-optimization evaluation splits (train, validation, test, full). This ensures the optimized prompt is evaluated under the same generation conditions it was trained with
