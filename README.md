# Image Match Prompt Optimizer

Automated prompt optimization for product image comparison using [MLflow GEPA](https://mlflow.org/docs/latest/llms/gepa/index.html) and [Google Gemini](https://ai.google.dev/).

Compares a **reference image** against a **candidate image** to classify: **Match** (same product), **Mismatch** (different product), or **Inconclusive** (insufficient evidence). The framework automatically discovers high-precision prompts through iterative optimization with custom cost functions.

---

## How It Works

```
Reference Image + Candidate Image
            │
            ▼
    ┌───────────────┐       ┌──────────────┐       ┌──────────────┐
    │  Predict       │──────▶│  Score        │──────▶│  Reflect     │
    │  (Gemini Flash)│       │  (Cloud Func) │       │  (Gemini Pro)│
    └───────────────┘       └──────────────┘       └──────┬───────┘
            ▲                                              │
            └──────────── Improved Prompt ◀────────────────┘
                        (repeat N iterations)
```

1. A vision model classifies image pairs using the current prompt
2. A Cloud Function scores each prediction with configurable precision/recall weights
3. A critic model analyzes failures and proposes prompt improvements
4. Grid search across cost functions × iterations finds the best prompt

## Key Features

- **Binary classification** — Match vs Not_Match framing for maximum training signal
- **Configurable cost functions** — deploy different TP/FP/FN/TN weights as Cloud Functions to control the precision-recall tradeoff
- **Grid search** — automated search across cost function variants × iteration counts
- **Stepwise optimization** — chaining, blend, and error-focused (hard example mining) modes
- **Two-stage architecture** — independent Match and Mismatch optimization with multi-step pipeline
- **Category-agnostic** — same pipeline works for any product category; supply labeled data and run

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GCP
gcloud auth application-default login

# 3. Verify the pipeline works
./venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
    --category smartwatch \
    --data_dir ./data/images/smartwatch \
    --system_prompt ./prompts/binary_match_or_not.txt \
    --eval_cloud_function_name img_match_weighted_guarded \
    --project my-gcp-project \
    --location global \
    --eval_cloud_function_location us-central1 \
    --limit 5

# 4. Run GEPA grid optimization
./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
    --project my-gcp-project \
    --location global \
    --target_model gemini-3-flash-preview \
    --eval_cf_names img_match_weighted_guarded \
    --eval_score_key match_score \
    --num_iterations 16 \
    --category smartwatch \
    --data_dir ./data/sampled/label/smartwatch/train \
    --eval_data_dir ./data/sampled/label/smartwatch/validation \
    --initial_prompt ./prompts/binary_match_or_not.txt \
    --subsample_fraction 0.3 \
    --eval_workers 10 \
    --output_file ./grid_runs/match/smartwatch/run_1.json

# 5. Analyze results
./venv/bin/python scripts/post_optimization/analyze_grid_precision_recall_tradeoff.py \
    --input_dir ./grid_runs/match/smartwatch --verbose
```

See the [User Guide](docs/user_guide.md) for the full 10-step walkthrough.

---

## Project Structure

```
├── requirements.txt
├── docs/
│   ├── user_guide.md                  # 10-step guide from setup to optimized prompt
│   ├── gepa.md                        # GEPA architecture: 3 modes, data flow, components
│   ├── pipeline.md                    # Eval pipeline: match, mismatch, multi-step
│   ├── data.md                        # Data format and category-centric resolution
│   ├── sampling.md                    # Label-based and confusion-matrix sampling
│   ├── metrics.md                     # Cloud Function metric variants
│   ├── cost_function_design.md        # Weighted cost function design rationale
│   ├── binary_vs_multiclass_design.md # Why binary > 3-class for optimization
│   └── problem_statement.md           # Problem framing and success criteria
│
├── scripts/
│   ├── mlflow_gepa/                   # GEPA optimization scripts
│   │   ├── config.py                  # GEPAConfig dataclass
│   │   ├── data_loader.py             # CSV → GEPA DataFrame
│   │   ├── predict.py                 # Vision model predict function
│   │   ├── scorer.py                  # Cloud Function scorer
│   │   ├── thinking_optimizer.py      # Critic model with thinking_level=HIGH
│   │   ├── run_gepa.py                # Basic GEPA orchestrator
│   │   ├── run_gepa_stepwise.py       # Stepwise: chaining, blend, error-focused
│   │   ├── run_gepa_binary_match_grid.py      # Grid runner (match)
│   │   └── run_gepa_binary_mismatch_grid.py   # Grid runner (mismatch)
│   │
│   ├── pipeline/                      # Evaluation pipelines
│   │   ├── run_binary_match_pipeline_with_eval.py
│   │   ├── run_binary_mismatch_pipeline_with_eval.py
│   │   ├── run_multi_step_pipeline_with_eval.py
│   │   ├── summarize_binary_match_pipeline_eval.py
│   │   └── summarize_binary_mismatch_pipeline_eval.py
│   │
│   ├── sampling/                      # Data sampling scripts
│   │   ├── sample_data_by_label.py
│   │   └── sample_data_by_confusion_matrix.py
│   │
│   ├── post_optimization/             # Post-optimization analysis
│   │   ├── analyze_grid_precision_recall_tradeoff.py
│   │   ├── extract_prompt_from_grid_run.py
│   │   └── blend_prompts.py
│   │
│   └── deploy_custom_metrics/         # Cloud Function deploy scripts
│       ├── deploy_match_only_metrics.sh
│       └── deploy_mismatch_only_metrics.sh
│
├── prompts/                           # System prompts and templates
│   ├── binary_match_or_not.txt
│   ├── binary_mismatch_or_not.txt
│   ├── binary_match_or_not_with_image_quality_precheck.txt
│   ├── binary_mismatch_or_not_with_image_quality_precheck.txt
│   ├── blend_prompts_template.txt
│   ├── blend_prompts_mismatch_template.txt
│   └── gepa_reflection_prompt_match_precision_focused.txt
│
├── custom_metrics/                    # Cloud Function source code
│   ├── custom_metric_match_only/
│   ├── custom_metric_mismatch_only/
│   ├── custom_metric_weighted_match_only_guarded/
│   ├── custom_metric_weighted_match_only_balanced/
│   ├── custom_metric_weighted_match_only_aggressive/
│   └── ... (14 variants total)
│
├── data/                              # Image data (not included in repo)
├── grid_runs/                         # Grid run output JSONs
├── benchmark_runs/                    # Pipeline benchmark outputs
└── config/                            # Sampling configuration files
```

---

## Optimization Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Basic** | `--step_size 0` (default) | Standard GEPA: N iterations on full/subsampled data |
| **Stepwise Chaining** | `--step_size 4` | Each step refines the previous step's prompt on a different data subsample |
| **Stepwise Blend** | `--step_size 4 --no_chaining` | Independent steps from initial prompt, then blended by Gemini 3.1 Pro |
| **Error-Focused** | `--step_size 4 --error_focused` | Chaining with hard example mining: classifies TP/FP/FN/TN between steps, enriches next step with error cases |

---

## Cost Functions

Precision is controlled by the cost function's FP:FN weight ratio:

| Variant | FP | FN | Precision Target | Best For |
|---------|----|----|-----------------|----------|
| Balanced | -3 | -3 | ~60% | Baseline, F1 optimization |
| Moderate | -7 | -1 | ~87% | General use |
| **Guarded** | **-9** | **-2** | **~90%** | **Recommended default** |
| Highly Aggressive | -15 | -0.25 | ~93% | Maximum precision |

The **guarded** variant includes a recall floor (FN penalty + TN reward) to prevent the optimizer from achieving high precision by never predicting Match.

---

## Prerequisites

- **Python 3.11+**
- **GCP Project** with Vertex AI API enabled
- **gcloud CLI** authenticated
- Deployed Cloud Function metrics (see [metrics.md](docs/metrics.md))

---

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user_guide.md) | Complete 10-step walkthrough |
| [GEPA Architecture](docs/gepa.md) | Optimization modes, components, data flow |
| [Pipeline](docs/pipeline.md) | Match, mismatch, and multi-step pipelines |
| [Data Format](docs/data.md) | Mapping CSV format, image organization |
| [Sampling](docs/sampling.md) | Label-based and confusion-matrix sampling |
| [Cost Functions](docs/cost_function_design.md) | Weighted scoring design |
| [Metrics](docs/metrics.md) | All Cloud Function variants |
| [Binary vs Multiclass](docs/binary_vs_multiclass_design.md) | Why binary framing works better |

---

## Built With

- [MLflow GEPA](https://mlflow.org/docs/latest/llms/gepa/index.html) — prompt optimization
- [Google Gemini](https://ai.google.dev/) — vision model (target) and reasoning model (critic)
- [GCP Cloud Functions](https://cloud.google.com/functions) — evaluation metrics
- [LiteLLM](https://docs.litellm.ai/) — model routing for GEPA reflection