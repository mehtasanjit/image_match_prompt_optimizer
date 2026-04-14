# GEPA Prompt Optimization

## Overview

GEPA (Genetic-Pareto) is an open-source prompt optimization library that MLflow wraps via its `GepaPromptOptimizer` class. GEPA uses iterative mutation, reflection, and Pareto-aware candidate selection to improve prompts. A reflection model (LLM) analyzes system behavior on failures and proposes improvements, while Pareto selection ensures optimization across multiple objectives simultaneously.

This project uses GEPA to iteratively refine system prompts for product image comparison, maximizing precision on a custom weighted cost function deployed as a GCP Cloud Function.

The optimization loop works at two levels:

**Outer loop (our framework)**: Controlled by `run_gepa.py` / `run_gepa_stepwise.py`. Loads data, initializes MLflow, sets up the scorer, and calls `mlflow.genai.optimize_prompts()`. For stepwise mode, runs multiple GEPA optimizations sequentially with different data compositions.

**Inner loop (GEPA library)**: Controlled entirely by the `gepa` package. Runs hundreds of proposal-evaluate-reflect cycles within a single `gepa.optimize()` call. Each cycle:

1. **Select** a candidate from the Pareto frontier (a pool of diverse prompts, each best at something)
2. **Predict** on a minibatch of 3 training examples using the target model (e.g. `gemini-3-flash-preview`), capturing full execution traces (model reasoning, scores)
3. **Reflect** using the critic model (e.g. `gemini-3.1-pro-preview` with thinking_level=HIGH) — it reads the 3 traces, diagnoses *specific* failure patterns, and proposes a new prompt with targeted modifications
4. **Validate** the proposed prompt on the same 3 items — if it improves, run full validation on ALL training items
5. **Score** each prediction via the Cloud Function scorer, which applies the weighted cost function (e.g. TP=+1, FP=-9, FN=-2, TN=+0.5)
6. **Update** the Pareto frontier if the new prompt is not dominated by existing candidates
7. **Repeat** until the metric call budget (`num_iterations × dataset_size`) is exhausted

At the end, GEPA returns the candidate with the highest average score across all training items. See the "GEPA Algorithm (Deep Dive)" section below for the full algorithm with annotated logs.

## The GEPA Algorithm (Deep Dive)

GEPA is a **separate library** (`gepa` pip package) that MLflow wraps via `GepaPromptOptimizer`. MLflow provides the orchestration (prompt registry, scoring, data loading); GEPA provides the actual optimization algorithm.

**Paper**: [GEPA: Genetic-Pareto Optimization for Text Evolution](https://arxiv.org/abs/2507.19457)

### Core Concepts

| Concept | Description |
|---|---|
| **Candidate** | A dictionary mapping component names to text values. E.g., `{"system_prompt": "You are..."}`. In our case, a single prompt being optimized. |
| **Pareto Frontier** | A set of candidates where each is the best at *something* — e.g., excelling on different subsets of examples. Any candidate that is not dominated (beaten on all objectives) survives. |
| **Metric Call** | One predict+score evaluation on a single training example. The fundamental unit of compute budget. |
| **Reflection LM** | The critic model (e.g., `gemini-3.1-pro-preview`) that analyzes execution traces and proposes prompt improvements. |
| **Actionable Side Information (ASI)** | Execution traces, error messages, reasoning logs captured during evaluation — the raw material the reflection LM uses to diagnose failures. |

### The Optimization Loop

Each GEPA iteration chooses between two strategies:

```
┌──────────────────────────────────────────────────────────────────┐
│                     GEPA Iteration                                │
│                                                                   │
│  ┌─────────────────────┐     ┌──────────────────────────┐        │
│  │ Strategy A:          │     │ Strategy B:               │        │
│  │ REFLECTIVE MUTATION  │ OR  │ SYSTEM-AWARE MERGE        │        │
│  └──────────┬──────────┘     └────────────┬─────────────┘        │
│             │                              │                      │
│             ▼                              ▼                      │
│  1. Select candidate        1. Select two candidates              │
│     from Pareto frontier       from Pareto frontier               │
│             │                              │                      │
│  2. Evaluate on MINIBATCH   2. Merge modules based on             │
│     (3 examples, with          evolution history                  │
│      full trace capture)                   │                      │
│             │                              │                      │
│  3. REFLECT: Reflection LM  3. Evaluate merged candidate         │
│     analyzes traces,            on stratified subsample           │
│     diagnoses failures                     │                      │
│             │                              │                      │
│  4. PROPOSE: Generate        4. If score ≥ parents' scores        │
│     improved candidate          → proceed to full eval            │
│             │                              │                      │
│  5. Evaluate proposal on                   │                      │
│     SAME minibatch                         │                      │
│             │                              │                      │
│  6. If improved on minibatch               │                      │
│     → proceed to full eval                 │                      │
│             │                              │                      │
│             └──────────────┬───────────────┘                      │
│                            ▼                                      │
│              ┌──────────────────────┐                             │
│              │  FULL VALIDATION      │                             │
│              │  (entire val set)     │                             │
│              │  Update Pareto        │                             │
│              │  frontier if better   │                             │
│              └──────────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step (Reflective Mutation Path)

1. **Candidate Selection**: Pick a candidate from the Pareto frontier using novelty-weighted sampling (balances exploration vs exploitation)

2. **Minibatch Evaluation**: Run the candidate on a small batch of training examples (`reflection_minibatch_size`, default=3). Crucially, this captures **full execution traces** — the model's reasoning, intermediate outputs, and scores — not just the final label.

3. **Reflection**: The `reflection_lm` receives the traces and diagnoses *why* the candidate failed on specific examples. It sees:
   - The current prompt text
   - The model's full output (including reasoning)
   - The ground truth
   - The score
   
   It then proposes **targeted textual modifications** — not random mutations, but directed changes based on observed failure patterns.

4. **Proposal Evaluation**: The proposed new candidate is evaluated on the **same minibatch**. This is a cheap sanity check — if the proposal doesn't improve on 3 examples, it's unlikely to improve on the full dataset.

5. **Acceptance Gate**: If the proposal improves the minibatch score, it proceeds to full validation. If not, it's rejected and the iteration ends (metric calls were still consumed).

6. **Full Validation**: The accepted candidate is evaluated on the entire validation set. The Pareto frontier is updated — the new candidate is added if it's not dominated by any existing candidate.

### System-Aware Merge (Strategy B)

Periodically, GEPA picks two candidates from the Pareto frontier and merges them:

- If candidate A was refined on algebraic problems and candidate B excels at geometry, the merge creates a hybrid combining the strengths of both
- Module selection is based on **evolution history** — if a component was actively refined in one candidate but left unchanged in another, the refined version is preferred
- The merged candidate must score ≥ both parents to be accepted

### From Many Candidates to One Output

Throughout optimization, GEPA maintains a **Pareto frontier** — multiple candidates that each excel on different subsets of training examples. But at the end, it must return a **single best prompt**.

**During optimization** — the Pareto frontier drives exploration:
- Candidates are selected from the frontier for mutation using novelty-weighted sampling
- A candidate that's best on 3 specific hard cases survives even if its overall score is mediocre
- Different `frontier_type` settings control how the frontier is tracked:
  - **`instance` (default — what we use)**: Per-example Pareto dominance. A candidate survives if it's the best on even one training example. This means candidates that excel on specific hard cases are preserved even if their average score is mediocre. With our single-scorer setup, this keeps prompts that handle niche failure modes (e.g., "packaging-only images" or "extreme angles") alive in the pool for potential merging.
  - `objective`: Per-objective (relevant for multi-objective optimization with multiple scorers)
  - `hybrid`/`cartesian`: Combinations of the above

**After optimization** — final selection is simple:
- Every candidate that was ever added to the Pareto frontier has a `val_aggregate_score` — the average weighted score across ALL validation items
- The final output is the candidate with the **highest `val_aggregate_score`**
- This is a straightforward argmax, not Pareto selection

So the Pareto frontier's role is to **maintain diversity during search** (preventing premature convergence to a local optimum), while the final selection is purely **"which candidate has the best average score across all data?"**

In our case, the `val_aggregate_score` is the average Cloud Function score (guarded CF: TP=+1, FP=-9, FN=-2, TN=+0.5) across all training items. The candidate that maximizes this weighted average — balancing precision and recall as encoded by the CF weights — wins.

**Source code reference** ([`src/gepa/core/result.py`](https://github.com/gepa-ai/gepa/blob/c861dcd8841b7748733a573655f65d9638de99b6/src/gepa/core/result.py)):

```python
# GEPAResult class in gepa library
@property
def best_idx(self):
    # Returns index of candidate with highest average validation score
    return max(range(len(self.val_aggregate_scores)),
               key=lambda i: self.val_aggregate_scores[i])

@property
def best_candidate(self):
    return self.candidates[self.best_idx]
```

In our code, MLflow exposes this via `result.final_eval_score` (maps to `val_aggregate_scores[best_idx]`) and `result.optimized_prompts[0].template` (maps to `best_candidate`).

### How `max_metric_calls` Relates to `--num_iterations` and Dataset Size

In our framework:

```
max_metric_calls = --num_iterations × len(training_data)
```

**Example**: `--num_iterations 16` with 96 training items:
```
max_metric_calls = 16 × 96 = 1,536
```

But GEPA does NOT do 16 sequential passes over 96 items. Instead, it spends 1,536 metric calls across:

| Activity | Calls per occurrence | How many times |
|---|---|---|
| Minibatch evaluation | 3 (default) | Many (one per iteration) |
| Minibatch validation of proposal | 3 | Many (one per accepted proposal) |
| Full validation | N (all val items) | Few (only when proposal improves minibatch) |
| Rejected proposals | 3-6 | Variable |

So the actual number of GEPA iterations (proposal cycles) **far exceeds** `--num_iterations`. With `max_metric_calls=1536`:

- If full validation runs on 96 items and happens ~5 times: 480 calls
- Remaining 1056 calls ÷ 6 calls per iteration ≈ **176 proposal cycles**
- Of those, maybe 20-30 are accepted → 20-30 actual prompt improvements

**GEPA's recommendation**: Run with `max_metric_calls` at least 15-30× the validation set size. For 96 items, that's 1,440-2,880 — equivalent to `--num_iterations 15-30`.

### Why This Matters for Our Framework

| Parameter | What it controls | Effect |
|---|---|---|
| `--num_iterations` | Total metric call budget (×dataset) | More budget → more proposal cycles → better prompts, but more cost |
| `--subsample_fraction` | Dataset size seen by GEPA | Smaller dataset → more iterations per budget → faster but potentially overfit |
| `--step_size` | Budget per stepwise step | Each step gets `step_size × dataset` metric calls |
| `--error_focused` | What data GEPA sees | Changes composition, not budget |

The interplay between dataset size and iteration count matters:

```
# 320 items, 16 iterations = 5,120 metric calls = many proposal cycles
--data_dir (320 items) --num_iterations 16

# 96 items, 16 iterations = 1,536 metric calls = fewer proposal cycles  
--data_dir (96 items) --num_iterations 16

# 96 items, 48 iterations = 4,608 metric calls ≈ same as first case
--data_dir (96 items) --num_iterations 48
```

For small datasets (<100 items), increase `--num_iterations` to compensate for the reduced metric call budget.

### Mapping GEPA Concepts to Our Implementation

| GEPA Internal Concept | Our Implementation | Where |
|---|---|---|
| **seed_candidate** | Initial prompt loaded from `--initial_prompt` file, registered in MLflow prompt registry | `run_gepa.py` line: `mlflow.register_prompt(template=initial_prompt_text)` |
| **trainset** | DataFrame from `data_loader.load_eval_data()` — each row has `inputs` (image paths) and `outputs` (ground truth) | `data_loader.py` |
| **valset** | Same as trainset (GEPA uses trainset for both minibatch eval and full validation in our setup) | MLflow passes `train_data` to GEPA as both train and val |
| **adapter / predict_fn** | `predict.predict_fn(reference_image_path, image_path)` — calls Gemini Flash with current prompt, returns full JSON | `predict.py` |
| **metric / scorer** | `scorer.weighted_scorer` — calls Cloud Function with `{response, target}`, returns weighted score | `scorer.py` |
| **reflection_lm** | `vertex_ai:/gemini-3.1-pro-preview` with `reasoning_effort="high"` via `ThinkingGepaPromptOptimizer` | `thinking_optimizer.py` |
| **max_metric_calls** | `num_iterations × len(train_data)` — computed by our grid runner, passed to `GepaPromptOptimizer` | `run_gepa.py`: `max_metric_calls = config.num_iterations * len(train_data)` |
| **reflection_minibatch_size** | Default 3 (GEPA default, we don't override) — 3 image pairs evaluated per reflection cycle. See note below. | GEPA internal default |
| **Pareto frontier** | Managed entirely by GEPA internally — we only see the final best candidate | GEPA internal |
| **System-Aware Merge** | Happens automatically within GEPA — we have a single component (`system_prompt`), so merge is less impactful | GEPA internal |
| **Full validation** | GEPA evaluates accepted proposals on the entire `train_data` DataFrame — this is where most metric calls go | GEPA internal |
| **ASI / execution traces** | Our `predict_fn` returns the full model JSON (including `reason` field) — GEPA captures this as traces for the reflection LM | `predict.py` returns full JSON, not just the label |
| **Callbacks** | Not used in our implementation — could add for logging/monitoring | Available via `gepa.optimize(callbacks=[...])` |

**Note on `reflection_minibatch_size`**: We don't expose this parameter — it uses GEPA's default of **3**. Each GEPA iteration samples exactly **1 minibatch of 3 items** (not multiple minibatches). The cycle is: sample 3 → evaluate with traces → reflect → propose → validate on same 3 → accept/reject → (if accepted) full validation on all items. One minibatch per cycle.

**How many minibatch cycles per `gepa.optimize()` call?** Hundreds. Each cycle consumes metric calls as follows:

| Outcome | Metric calls consumed | Breakdown |
|---|---|---|
| **Rejected proposal** | **6** | 3 (evaluate current candidate on minibatch) + 3 (validate proposal on same minibatch). The proposal didn't improve the minibatch score → rejected, no full validation. |
| **Accepted proposal** | **6 + N** | 3 (minibatch eval) + 3 (minibatch validation) + N (full validation on all items). The proposal improved the minibatch → accepted → full dataset evaluation. |

Note: The reflection LLM call (critic model) between minibatch eval and proposal is NOT a metric call — it doesn't consume budget.

With `max_metric_calls = 2,928` and 183 training items: ~240-340 minibatch cycles run, of which ~5-8 get accepted. See the "Under the Hood" section for a worked example with actual log output.

**Is 3 enough?** For vision tasks with expensive per-item calls (2 images + prompt → Gemini Flash), 3 is a good tradeoff:
- Gives the critic enough failure diversity per cycle (e.g., 1 FP + 1 FN + 1 TN)
- Keeps per-iteration cost low (6 metric calls for a rejected proposal, ~N+6 for accepted)
- More iterations with diverse minibatches > fewer iterations with larger batches
- GEPA docs recommend 3-5 for sample efficiency

GEPA supports overriding this via `EngineConfig(reflection_minibatch_size=N)`, but we don't currently expose it as a CLI arg. If you want to experiment with larger minibatches (e.g., 5-10 for categories with many similar failure modes), the GEPA config can be extended.

**Key insight**: We only control the outer loop (what data GEPA sees, how many metric calls, which scorer). GEPA controls the inner loop (minibatch selection, reflection strategy, Pareto management, acceptance decisions). Our stepwise/error-focused modes add a layer *above* GEPA — they run multiple GEPA optimizations sequentially with different data compositions.

### Under the Hood: What Happens in One Step (Annotated Log)

This traces through one complete error-focused step to show exactly what happens at each layer.

**Phase 1 — Our code (error-focused sampling):**

The stepwise runner runs the current prompt on all 319 training items, classifies each prediction:

```
Classification distribution: TP=86 FP=20 FN=97 TN=116
  TP: 43/86 (50%)      ← Keep half the correct matches (anchoring examples)
  FP: 20/20 (100%)     ← Keep ALL false positives (most valuable for precision)
  FN: 97/97 (100%)     ← Keep ALL false negatives (missed matches)
  TN: 23/116 (20%)     ← Keep 20% of correct rejections (background)
Error-aware subsample: 183 items
Step 2: max_metric_calls=2928 (16 iters × 183 samples)
```

183 items are passed to GEPA. The metric call budget is `16 × 183 = 2,928`.

**Phase 2 — MLflow wrapper:**

```
Testing model prediction with the first sample  ← MLflow sanity-checks predict_fn works
ThinkingGepaPromptOptimizer: reasoning_effort=high
Patched gepa.optimize: reflection_lm replaced   ← Our patch swaps string URI for thinking LM
```

MLflow sets up the GEPA adapter (predict_fn, scorer, prompt registry) and calls `gepa.optimize()`.

**Phase 3 — GEPA Iteration 0 (baseline):**

```
Iteration 0: Base program full valset score: -1.7459 over 183/183 examples
```

GEPA evaluates the seed prompt on ALL 183 items → **183 metric calls consumed**. Each `AFC is enabled` log line = one Gemini Flash predict call. The score is the average weighted score across all items (negative because FPs outweigh TPs under the guarded cost function).

**Phase 4 — GEPA Iteration 1 (first proposal cycle):**

Step 1 — **Candidate selection**:
```
Iteration 1: Selected program 0 score: -1.7459
```
Picks the only candidate from the Pareto frontier (the seed prompt).

Step 2 — **Minibatch evaluation** (3 items, with trace capture):
```
AFC is enabled...  ← 3 concurrent predict calls on minibatch
AFC is enabled...
AFC is enabled...
```
3 metric calls consumed. GEPA captures full execution traces (model reasoning, scores) for these 3 items.

Step 3 — **Reflection** (critic model):
```
LiteLLM completion() model=gemini-3.1-pro-preview; provider=vertex_ai
Wrapper: Completed Call, calling success_handler   ← ~38 seconds
```
The reflection LM (Gemini 3.1 Pro with `thinking_level=HIGH`) receives the 3 traces and the current prompt. It analyzes failure patterns and proposes a new prompt. This is NOT a metric call — it's a separate LLM call to the critic model.

Step 4 — **Proposal** (the new prompt text):
```
Iteration 1: Proposed new text for ...: ### Role & Persona
You are an AI assistant specializing in product image comparison...
```
The full proposed prompt is logged. Notice it added category-specific directives (smartwatch crowns, strap matching, perspective differences) based on the 3 failure traces.

Step 5 — **Minibatch validation** (same 3 items):
```
AFC is enabled...  ← 3 predict calls with new prompt
AFC is enabled...
AFC is enabled...
```
3 metric calls consumed. GEPA evaluates the proposal on the SAME 3 minibatch items.

Step 6 — **Acceptance decision**:
```
Iteration 1: New subsample score 3.0 is better than old score -6.0.
Continue to full eval and add to candidate pool.
```
Score improved from -6.0 to 3.0 on the minibatch → **ACCEPTED** → proceeds to full validation.

Step 7 — **Full validation** (all 183 items):
```
AFC is enabled...   ← 183 concurrent predict calls
AFC is enabled...
... (many more)
```
183 metric calls consumed. The new prompt is evaluated on all 183 items.

**Metric call budget after Iteration 1:**
```
Iteration 0:  183 calls (baseline)
Iteration 1:    3 calls (minibatch eval)
            +   3 calls (minibatch validation)
            + 183 calls (full validation)
            = 189 calls
──────────────────────────────
Total used: 372 / 2,928 budget
Remaining:  2,556 calls
```

GEPA continues with Iteration 2, 3, ... until the 2,928 budget is exhausted. Most iterations will be **rejected** (proposal doesn't beat minibatch score → 6 calls wasted, no full validation). Accepted proposals trigger 183-call full validations, which dominate the budget.

**Rough estimate**: With 2,928 budget and ~5-8 accepted proposals (each costing ~189 calls = 945-1,512 calls), the remaining ~1,400-2,000 calls support ~230-330 rejected proposal cycles (6 calls each). Total: **~240-340 GEPA iterations**, of which only **5-8 actually improve the prompt**.

---

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
1. **Step 0**: Random sub-sample using `--subsample_fraction` (no prior predictions exist)
2. **Steps 1+**: Run predict_fn on **all** training data in `--data_dir` → classify each item as TP/FP/FN/TN → subsample with configurable quadrant fractions

**Important**: In error-focused mode, the inter-step classification always runs on the **entire** training split (not the subsampled portion). This ensures the optimizer discovers errors on items it hasn't seen yet. The `--subsample_fraction` only controls the step 0 random sample size.

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
- **`--data_dir`** and **`--eval_data_dir`** are required; **`--test_data_dir`** and **`--full_data_dir`** are optional
- Evaluates each optimized prompt on train, validation, and optional test/full splits
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
