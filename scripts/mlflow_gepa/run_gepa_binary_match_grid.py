"""
Meta-iterator for running GEPA optimization across a grid of
eval_cf_names x num_iterations, then evaluating each final
optimized prompt on a held-out eval dataset.

Usage example (run from repo root):
    ./venv/bin/python scripts/mlflow_gepa/run_gepa_binary_match_grid.py \
        --project my-gcp-project \
        --eval_cf_names img_match_weighted_guarded,img_match_weighted_balanced \
        --eval_score_key match_score \
        --num_iterations 15,18,20 \
        --category smartwatch \
        --data_dir ./data/images/smartwatch/train \
        --eval_data_dir ./data/images/smartwatch/eval \
        --eval_category smartwatch \
        --initial_prompt ./prompts/binary_match_or_not.txt \
        --target_model gemini-3-flash-preview \
        --location global \
        --eval_cf_location us-central1 \
        --eval_workers 10 \
        --output_file grid_runs/match/smartwatch/results.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import threading
import concurrent.futures

# Ensure mlflow_gepa is importable as a top-level package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Ensure pipeline/ is importable for process_item and summarize_eval_results.
# The grid runner bridges GEPA optimization with the eval pipeline scripts
# (scripts/pipeline/), which are a sibling package in the same repo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from google import genai

from mlflow_gepa.config import GEPAConfig
from mlflow_gepa.run_gepa import run as run_gepa
from mlflow_gepa.run_gepa_stepwise import run_stepwise

# Import eval pipeline components (binary match variant).
# These come from scripts/pipeline/ — a sibling directory, not part of mlflow_gepa.
try:
    from run_binary_match_pipeline_with_eval import process_item, csv_to_eval_data
    from summarize_binary_match_pipeline_eval import summarize_eval_results
except ImportError as e:
    raise ImportError(
        "Grid runner requires the pipeline scripts (scripts/pipeline/) to be available. "
        "Ensure run_binary_match_pipeline_with_eval.py and "
        "summarize_binary_match_pipeline_eval.py are in scripts/pipeline/."
    ) from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_eval_data(eval_data_dir: str, category: str, eval_limit: int, mapping_csv: str = ""):
    """Load the eval dataset from mapping CSV."""
    if mapping_csv:
        csv_path = mapping_csv
    else:
        csv_path = os.path.join(eval_data_dir, f"{category}_mapping.csv")
    data = csv_to_eval_data(csv_path)
    if eval_limit > 0:
        data = data[:eval_limit]
    logger.info("Loaded %d eval items from %s", len(data), csv_path)
    return data


def _run_eval_pipeline(
    system_prompt_text: str,
    eval_data: list,
    eval_data_dir: str,
    category_name: str,
    eval_cf_name: str,
    project: str,
    location: str,
    eval_cf_location: str,
    model: str,
    workers: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    few_shot_examples: list = None,
) -> dict:
    """
    Run the eval pipeline on the eval dataset using the given prompt.
    Uses the same temperature and top_p as the GEPA training run for consistency.
    Returns aggregated_metrics dict (no individual_results).
    """
    client = genai.Client(project=project, location=location, vertexai=True)
    total_items = len(eval_data)

    def worker(item, i):
        logger.info("Eval item %d/%d | ID: %s", i + 1, total_items, item.get("id"))
        return process_item(
            item=item,
            data_dir=eval_data_dir,
            system_prompt=system_prompt_text,
            client=client,
            project=project,
            eval_cloud_function_name=eval_cf_name,
            eval_cloud_function_location=eval_cf_location,
            category_name=category_name,
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=0,
            few_shot_examples=few_shot_examples,
        )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item, i) for i, item in enumerate(eval_data)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    config_dict = {
        "category": category_name,
        "data_dir": eval_data_dir,
        "model": model,
        "eval_cloud_function_name": eval_cf_name,
        "temperature": temperature,
        "top_p": top_p,
    }

    summary = summarize_eval_results(results, config_dict)

    # Strip individual_results to keep output compact
    aggregated = {
        "config": summary.get("config", {}),
        "aggregated_metrics": summary.get("aggregated_metrics", {}),
    }
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization across a grid of eval_cf_names × num_iterations, "
        "then evaluate each final prompt on a held-out eval set.",
    )

    # GCP
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP location for GenAI")

    # Target model
    parser.add_argument("--target_model", default="gemini-2.5-flash", help="Vision model to optimize")
    parser.add_argument("--temperature", type=float, default=0.0, help="GEPA training temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="GEPA training top_p")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Thinking budget (0 = disabled)")

    # Critic model
    parser.add_argument("--critic_model", default="gemini-3.1-pro-preview", help="GEPA reflection model")
    parser.add_argument("--critic_model_location", default="global", help="GCP location for the critic/reflection model")
    parser.add_argument("--critic_reasoning_effort", default="high",
                        help="Thinking level for the critic model: 'high', 'low', or '' to disable (default: high)")

    # Grid dimensions
    parser.add_argument(
        "--eval_cf_names", required=True,
        help="Comma-separated Cloud Function names to grid over",
    )
    parser.add_argument(
        "--eval_score_key", required=True,
        help="Single score key (applied to all) or comma-separated (1 per eval_cf_name)",
    )
    parser.add_argument(
        "--num_iterations", required=True,
        help="Comma-separated iteration counts to grid over (e.g. 15,18,20)",
    )
    parser.add_argument("--eval_cf_location", default="us-central1", help="Cloud Function location")

    # Prompt
    parser.add_argument("--prompt_name", default="image_match_system_prompt", help="MLflow prompt registry name")
    parser.add_argument(
        "--initial_prompt",
        default="prompts/binary_match_or_not.txt",
        help="Path to initial system prompt file",
    )

    # GEPA training data
    parser.add_argument("--category", default=None,
                        help="Product category name. Used for auto-resolving mapping CSVs as "
                             "<data_dir>/<category>_mapping.csv. If not provided, --mapping_csv is required.")
    parser.add_argument("--data_dir", required=True, help="Training data directory")
    parser.add_argument("--mapping_csv", default="", help="Explicit path to training mapping CSV (optional)")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Max training samples (0 = all)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    # Eval / test datasets
    parser.add_argument("--eval_data_dir", required=True, help="Validation dataset directory (images + mapping CSV)")
    parser.add_argument("--eval_category", default=None, help="Eval category name (defaults to --category)")
    parser.add_argument("--eval_mapping_csv", default="", help="Explicit path to eval mapping CSV (optional)")
    parser.add_argument("--eval_limit", type=int, default=0, help="Limit eval items (0 = all)")
    parser.add_argument("--eval_workers", type=int, default=10, help="Concurrent workers for eval pipeline")
    parser.add_argument("--test_data_dir", default=None, help="Test dataset directory (images + mapping CSV). Optional.")
    parser.add_argument("--test_category", default=None, help="Test category name (defaults to --eval_category or --category)")
    parser.add_argument("--test_mapping_csv", default="", help="Explicit path to test mapping CSV (optional)")
    parser.add_argument("--test_limit", type=int, default=0, help="Limit test items (0 = all)")

    # Eval Cloud Function for the eval pipeline (may differ from GEPA training CF)
    parser.add_argument(
        "--eval_pipeline_cf_name", default=None,
        help="CF name for eval pipeline scoring. Defaults to each grid eval_cf_name.",
    )

    # Output
    parser.add_argument("--output_file", default="grid_results.json", help="Path to output JSON")
    parser.add_argument("--experiment_name", default="IMAGE_MATCH_GEPA_GRID", help="MLflow experiment name")

    # Few-shot examples
    parser.add_argument("--few_shot_examples_file", default=None,
                        help="Path to JSON file with few-shot examples for eval pipeline")

    # Reflection prompt template
    parser.add_argument("--reflection_prompt_template", default=None,
                        help="Path to custom reflection prompt template for GEPA critic model")

    # Stepwise GEPA
    parser.add_argument("--step_size", type=int, default=0,
                        help="Stepwise GEPA: iterations per step. 0 = disabled (standard GEPA).")
    parser.add_argument("--subsample_fraction", type=float, default=1.0,
                        help="Fraction of training data to use for GEPA (0.0-1.0). Default 1.0 = all data. "
                             "In stepwise mode, this is the per-step subsample fraction.")
    parser.add_argument("--no_chaining", action="store_true",
                        help="Stepwise GEPA: run all steps independently from initial prompt, then blend results.")
    parser.add_argument("--error_focused", action="store_true",
                        help="Stepwise GEPA: use error-aware sampling between steps. "
                             "Runs predictions on full training data after each step, classifies as TP/FP/FN/TN, "
                             "and enriches the next step's sample with error cases. Incompatible with --no_chaining.")
    parser.add_argument("--fp_fraction", type=float, default=1.0,
                        help="Error-focused: fraction of false positives to include (0.0-1.0). Default 1.0 = all.")
    parser.add_argument("--fn_fraction", type=float, default=1.0,
                        help="Error-focused: fraction of false negatives to include (0.0-1.0). Default 1.0 = all.")
    parser.add_argument("--tp_fraction", type=float, default=0.3,
                        help="Error-focused: fraction of true positives to include (0.0-1.0). Default 0.3.")
    parser.add_argument("--tn_fraction", type=float, default=0.3,
                        help="Error-focused: fraction of true negatives to include (0.0-1.0). Default 0.3.")
    parser.add_argument("--full_data_dir", default=None,
                        help="Directory with full dataset for evaluating blended/final prompt. Images + mapping CSV.")
    parser.add_argument("--full_data_category", default=None,
                        help="Category name for full dataset (defaults to --category).")
    parser.add_argument("--full_data_mapping_csv", default="",
                        help="Explicit path to full dataset mapping CSV (optional).")

    # Repetitions
    parser.add_argument("--num_repetitions", type=int, default=1,
                        help="Number of repetitions per grid cell. Each repetition uses a different "
                             "derived seed for sub-sampling. Default 1 (no repetitions).")
    parser.add_argument("--num_repetition_workers", type=int, default=1,
                        help="Max concurrent grid cells to run in parallel. Default 1 (sequential). "
                             "Values > 1 run cells concurrently using threads. Be mindful of API quotas.")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger("mlflow_gepa").setLevel(logging.DEBUG)

    # Resolve per-split categories: each split can have its own category
    train_category = args.category or ""
    eval_category = args.eval_category or args.category or ""
    test_category = args.test_category or args.eval_category or args.category or ""

    # Validate: each split needs either its category or explicit mapping_csv
    if not train_category and not args.mapping_csv:
        parser.error("Training split requires --category or --mapping_csv.")
    if not eval_category and not args.eval_mapping_csv:
        parser.error("Eval split requires --category, --eval_category, or --eval_mapping_csv.")
    if args.test_data_dir and not test_category and not args.test_mapping_csv:
        parser.error("Test split requires --category, --test_category, or --test_mapping_csv.")

    full_category = args.full_data_category or args.category or ""
    if args.full_data_dir and not full_category and not args.full_data_mapping_csv:
        parser.error("Full data split requires --category, --full_data_category, or --full_data_mapping_csv.")

    # Parse grid dimensions
    cf_names = [s.strip() for s in args.eval_cf_names.split(",")]
    iterations_list = [int(s.strip()) for s in args.num_iterations.split(",")]

    score_keys_raw = [s.strip() for s in args.eval_score_key.split(",")]
    if len(score_keys_raw) == 1:
        score_keys = [score_keys_raw[0]] * len(cf_names)
    elif len(score_keys_raw) == len(cf_names):
        score_keys = score_keys_raw
    else:
        parser.error(
            f"--eval_score_key must be a single value or match the count of --eval_cf_names "
            f"({len(cf_names)}). Got {len(score_keys_raw)}."
        )

    # Build grid (with repetitions)
    num_reps = max(1, args.num_repetitions)
    grid = []
    for idx, cf_name in enumerate(cf_names):
        for num_iter in iterations_list:
            for rep_idx in range(num_reps):
                rep_seed = args.random_seed + rep_idx * 17
                grid.append({
                    "eval_cf_name": cf_name,
                    "eval_score_key": score_keys[idx],
                    "num_iterations": num_iter,
                    "repetition": rep_idx,
                    "seed": rep_seed,
                })

    if num_reps > 1:
        logger.info("Grid has %d cells: %d CF(s) × %d iteration count(s) × %d repetition(s)",
                     len(grid), len(cf_names), len(iterations_list), num_reps)
    else:
        logger.info("Grid has %d cells: %d CF(s) × %d iteration count(s)", len(grid), len(cf_names), len(iterations_list))

    # Pre-load all splits once
    train_data = _load_eval_data(args.data_dir, train_category, 0, args.mapping_csv)
    logger.info("Train split: %d items from %s", len(train_data), args.data_dir)

    # Compute max_train_samples from subsample_fraction (applies to both standard and stepwise modes)
    effective_max_train = args.max_train_samples
    if args.subsample_fraction < 1.0:
        computed = int(len(train_data) * args.subsample_fraction)
        effective_max_train = max(1, computed)
        logger.info("Subsample fraction %.2f → max_train_samples=%d (from %d total)",
                     args.subsample_fraction, effective_max_train, len(train_data))

    eval_data = _load_eval_data(args.eval_data_dir, eval_category, args.eval_limit, args.eval_mapping_csv)
    logger.info("Validation split: %d items from %s", len(eval_data), args.eval_data_dir)

    # Load few-shot examples if provided
    few_shot_examples = None
    if args.few_shot_examples_file:
        try:
            with open(args.few_shot_examples_file, "r") as f:
                few_shot_data = json.load(f)
                few_shot_examples = few_shot_data.get("examples", [])
                logger.info("Loaded %d few-shot examples from %s", len(few_shot_examples), args.few_shot_examples_file)
        except Exception as e:
            logger.error("Failed to load few-shot examples: %s", e)

    test_data = None
    if args.test_data_dir:
        test_data = _load_eval_data(args.test_data_dir, test_category, args.test_limit, args.test_mapping_csv)
        logger.info("Test split: %d items from %s", len(test_data), args.test_data_dir)

    # Auto-create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    results_lock = threading.Lock()

    def _run_single_cell(cell_idx, cell):
        """Process a single grid cell. Returns the cell result dict."""
        cf_name = cell["eval_cf_name"]
        score_key = cell["eval_score_key"]
        num_iter = cell["num_iterations"]
        rep_idx = cell.get("repetition", 0)
        cell_seed = cell.get("seed", args.random_seed)

        if num_reps > 1:
            cell_label = f"[{cell_idx + 1}/{len(grid)}] cf={cf_name} iters={num_iter} rep={rep_idx} seed={cell_seed}"
        else:
            cell_label = f"[{cell_idx + 1}/{len(grid)}] cf={cf_name} iters={num_iter}"
        logger.info("=" * 70)
        logger.info("GRID CELL %s", cell_label)
        logger.info("=" * 70)

        # Use a unique prompt name per cell to avoid MLflow registry collisions
        if num_reps > 1:
            cell_prompt_name = f"{args.prompt_name}_{cf_name}_{num_iter}_rep{rep_idx}"
        else:
            cell_prompt_name = f"{args.prompt_name}_{cf_name}_{num_iter}"

        gepa_config = GEPAConfig(
            project=args.project,
            location=args.location,
            target_model=args.target_model,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_budget=args.thinking_budget,
            critic_model=args.critic_model,
            critic_model_location=args.critic_model_location,
            critic_reasoning_effort=args.critic_reasoning_effort,
            eval_cf_name=cf_name,
            eval_cf_location=args.eval_cf_location,
            eval_score_key=score_key,
            prompt_name=cell_prompt_name,
            initial_prompt_path=args.initial_prompt,
            category=args.category,
            data_dir=args.data_dir,
            mapping_csv=args.mapping_csv,
            max_train_samples=effective_max_train,
            random_seed=cell_seed,
            num_iterations=num_iter,
            experiment_name=args.experiment_name,
            reflection_prompt_template_path=args.reflection_prompt_template or "",
        )

        # ── 1. Run GEPA (standard or stepwise) ──
        gepa_start = time.time()
        try:
            if args.step_size > 0:
                chaining = not args.no_chaining
                logger.info("Using STEPWISE GEPA: step_size=%d, subsample=%.0f%%, chaining=%s, error_focused=%s",
                            args.step_size, args.subsample_fraction * 100, chaining, args.error_focused)
                result, final_prompt_text = run_stepwise(
                    gepa_config,
                    step_size=args.step_size,
                    subsample_fraction=args.subsample_fraction,
                    chaining=chaining,
                    error_focused=args.error_focused,
                    fp_fraction=args.fp_fraction,
                    fn_fraction=args.fn_fraction,
                    tp_fraction=args.tp_fraction,
                    tn_fraction=args.tn_fraction,
                    error_eval_workers=args.eval_workers,
                    positive_class="match",
                )
            else:
                result, final_prompt_text = run_gepa(gepa_config)
            initial_eval_score = getattr(result, "initial_eval_score", None)
            final_eval_score = getattr(result, "final_eval_score", None)
        except Exception as e:
            logger.error("GEPA run failed for %s: %s", cell_label, e)
            return {"grid_cell": cell, "error": str(e)}
        gepa_duration = time.time() - gepa_start

        logger.info("GEPA done in %.1fs | initial=%.4f final=%.4f",
                     gepa_duration, initial_eval_score or 0, final_eval_score or 0)

        eval_cf_for_pipeline = args.eval_pipeline_cf_name or cf_name

        # ── 2a. Run eval pipeline on TRAIN split ──
        logger.info("Running eval on TRAIN split (%d items) ...", len(train_data))
        train_eval_start = time.time()
        try:
            train_summary = _run_eval_pipeline(
                system_prompt_text=final_prompt_text,
                eval_data=train_data,
                eval_data_dir=args.data_dir,
                category_name=train_category,
                eval_cf_name=eval_cf_for_pipeline,
                project=args.project,
                location=args.location,
                eval_cf_location=args.eval_cf_location,
                model=args.target_model,
                workers=args.eval_workers,
                temperature=args.temperature,
                top_p=args.top_p,
                few_shot_examples=few_shot_examples,
            )
        except Exception as e:
            logger.error("Train eval failed for %s: %s", cell_label, e)
            train_summary = {"error": str(e)}
        train_eval_duration = time.time() - train_eval_start
        logger.info("Train eval done in %.1fs", train_eval_duration)

        # ── 2b. Run eval pipeline on VALIDATION split ──
        logger.info("Running eval on VALIDATION split (%d items) ...", len(eval_data))
        eval_start = time.time()
        try:
            eval_summary = _run_eval_pipeline(
                system_prompt_text=final_prompt_text,
                eval_data=eval_data,
                eval_data_dir=args.eval_data_dir,
                category_name=eval_category,
                eval_cf_name=eval_cf_for_pipeline,
                project=args.project,
                location=args.location,
                eval_cf_location=args.eval_cf_location,
                model=args.target_model,
                workers=args.eval_workers,
                temperature=args.temperature,
                top_p=args.top_p,
                few_shot_examples=few_shot_examples,
            )
        except Exception as e:
            logger.error("Validation eval failed for %s: %s", cell_label, e)
            eval_summary = {"error": str(e)}
        eval_duration = time.time() - eval_start
        logger.info("Validation eval done in %.1fs", eval_duration)

        # ── 2c. Run eval pipeline on TEST split (if provided) ──
        test_summary = None
        test_eval_duration = 0
        if test_data:
            logger.info("Running eval on TEST split (%d items) ...", len(test_data))
            test_eval_start = time.time()
            try:
                test_summary = _run_eval_pipeline(
                    system_prompt_text=final_prompt_text,
                    eval_data=test_data,
                    eval_data_dir=args.test_data_dir,
                    category_name=test_category,
                    eval_cf_name=eval_cf_for_pipeline,
                    project=args.project,
                    location=args.location,
                    eval_cf_location=args.eval_cf_location,
                    model=args.target_model,
                    workers=args.eval_workers,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    few_shot_examples=few_shot_examples,
                )
            except Exception as e:
                logger.error("Test eval failed for %s: %s", cell_label, e)
                test_summary = {"error": str(e)}
            test_eval_duration = time.time() - test_eval_start
            logger.info("Test eval done in %.1fs", test_eval_duration)

        # Collect stepwise history if available
        stepwise_history = getattr(result, "_stepwise_history", None) if result else None

        # ── 2d. Run eval pipeline on FULL dataset (if provided) ──
        full_summary = None
        full_eval_duration = 0
        if args.full_data_dir:
            full_data = _load_eval_data(args.full_data_dir, full_category, 0, args.full_data_mapping_csv)
            logger.info("Running eval on FULL dataset (%d items) ...", len(full_data))
            full_eval_start = time.time()
            try:
                full_summary = _run_eval_pipeline(
                    system_prompt_text=final_prompt_text,
                    eval_data=full_data,
                    eval_data_dir=args.full_data_dir,
                    category_name=full_category,
                    eval_cf_name=eval_cf_for_pipeline,
                    project=args.project,
                    location=args.location,
                    eval_cf_location=args.eval_cf_location,
                    model=args.target_model,
                    workers=args.eval_workers,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    few_shot_examples=few_shot_examples,
                )
            except Exception as e:
                logger.error("Full dataset eval failed for %s: %s", cell_label, e)
                full_summary = {"error": str(e)}
            full_eval_duration = time.time() - full_eval_start
            logger.info("Full dataset eval done in %.1fs", full_eval_duration)

        # Collect stepwise metadata
        stepwise_mode = getattr(result, "_mode", None) if result else None

        cell_result = {
            "grid_cell": cell,
            "gepa": {
                "initial_eval_score": initial_eval_score,
                "final_eval_score": final_eval_score,
                "duration_sec": round(gepa_duration, 1),
                "prompt_name": cell_prompt_name,
                "final_prompt": final_prompt_text,
                "mode": stepwise_mode or ("stepwise" if args.step_size > 0 else "standard"),
                **({"step_size": args.step_size, "subsample_fraction": args.subsample_fraction, "stepwise_history": stepwise_history} if stepwise_history else {}),
            },
            "eval_train": train_summary,
            "eval_train_duration_sec": round(train_eval_duration, 1),
            "eval_validation": eval_summary,
            "eval_validation_duration_sec": round(eval_duration, 1),
        }
        if test_summary is not None:
            cell_result["eval_test"] = test_summary
            cell_result["eval_test_duration_sec"] = round(test_eval_duration, 1)
        if full_summary is not None:
            cell_result["eval_full"] = full_summary
            cell_result["eval_full_duration_sec"] = round(full_eval_duration, 1)
        return cell_result

    # ── Execute grid cells (sequential or concurrent) ──
    num_cell_workers = max(1, args.num_repetition_workers)

    if num_cell_workers <= 1:
        # Sequential execution (default, same as original behavior)
        for cell_idx, cell in enumerate(grid):
            cell_result = _run_single_cell(cell_idx, cell)
            all_results.append(cell_result)
            _write_output(args.output_file, all_results, grid)
    else:
        # Concurrent execution across grid cells
        logger.info("Running grid cells with %d concurrent workers", num_cell_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cell_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single_cell, idx, cell): idx
                for idx, cell in enumerate(grid)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                cell_result = future.result()
                with results_lock:
                    all_results.append(cell_result)
                    _write_output(args.output_file, all_results, grid)

    # Final write
    _write_output(args.output_file, all_results, grid)
    logger.info("Grid search complete. Results written to %s", args.output_file)


def _write_output(path: str, results: list, grid: list):
    # Find best cell by eval set match precision
    best_by_precision = None
    best_precision_val = -1.0
    # Find best cell by eval set match F1
    best_by_f1 = None
    best_f1_val = -1.0
    # Find best cell by GEPA final_eval_score
    best_by_gepa = None
    best_gepa_score = -1.0

    for r in results:
        if "error" in r:
            continue

        agg = r.get("eval_validation", {}).get("aggregated_metrics", {})
        match_precision = agg.get("metrics_match", {}).get("precision", -1.0)
        match_recall = agg.get("metrics_match", {}).get("recall")
        match_f1 = agg.get("metrics_match", {}).get("f1", -1.0)

        cell_summary = {
            "grid_cell": r["grid_cell"],
            "eval_match_precision": match_precision,
            "eval_match_recall": match_recall,
            "eval_match_f1": match_f1,
            "gepa_final_eval_score": r.get("gepa", {}).get("final_eval_score"),
        }

        if match_precision > best_precision_val:
            best_precision_val = match_precision
            best_by_precision = cell_summary.copy()

        if match_f1 > best_f1_val:
            best_f1_val = match_f1
            best_by_f1 = cell_summary.copy()

        # GEPA final eval score
        gepa_score = r.get("gepa", {}).get("final_eval_score") or -1.0
        if gepa_score > best_gepa_score:
            best_gepa_score = gepa_score
            best_by_gepa = cell_summary.copy()

    output = {
        "grid_spec": grid,
        "total_cells": len(grid),
        "completed_cells": len(results),
        "best_by_eval_precision": best_by_precision,
        "best_by_eval_f1": best_by_f1,
        "best_by_gepa_final_score": best_by_gepa,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()