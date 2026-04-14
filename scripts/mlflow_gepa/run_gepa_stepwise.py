"""
Stepwise GEPA optimizer with per-step sub-sampling.

Supports three modes:
1. **Chaining** (default): Output prompt from step K becomes input for step K+1.
   Analogous to SGD — sequential refinement with data diversity.

2. **No-chaining** (--no_chaining): All steps start from the same initial prompt.
   Produces K independent prompts, which are then **blended** by a strong model
   into a single merged prompt. Analogous to ensemble → distillation.

3. **Error-focused** (--error_focused): Chaining with error-aware sampling.
   After each step, runs the current prompt on all training data, classifies
   predictions as TP/FP/FN/TN, and enriches the next step's sample with
   error cases (FP and FN). Incompatible with --no_chaining.

Usage:
    # Chaining mode
    result, final_prompt = run_stepwise(config, step_size=4, subsample_fraction=0.7)

    # No-chaining mode (parallel exploration + blend)
    result, final_prompt = run_stepwise(config, step_size=4, subsample_fraction=0.7, chaining=False)

    # Error-focused mode (hard example mining between steps)
    result, final_prompt = run_stepwise(config, step_size=4, error_focused=True,
                                         fp_fraction=1.0, fn_fraction=1.0,
                                         tp_fraction=0.3, tn_fraction=0.3)
"""

import concurrent.futures
import json
import logging
import os
import tempfile
import time

import pandas as pd
import mlflow
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

from google import genai
from google.genai import types as genai_types

from mlflow_gepa.config import GEPAConfig
from mlflow_gepa import data_loader, predict, scorer
from mlflow_gepa.thinking_optimizer import ThinkingGepaPromptOptimizer

# Re-export for direct attribute access
import mlflow_gepa.predict as _predict_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)



# Default path to blend prompt template (relative to external/ root)
_BLEND_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "blend_prompts_template.txt"
)


def _load_blend_template(path: str = None) -> str:
    """Load the blend prompt template from file."""
    template_path = path or _BLEND_PROMPT_TEMPLATE_PATH
    with open(template_path, "r") as f:
        return f.read()


def _blend_prompts(
    prompts: list[str],
    step_scores: list[dict],
    config: GEPAConfig,
) -> str:
    """Blend multiple prompts into one using a strong model."""
    logger.info("Blending %d prompts using %s ...", len(prompts), config.critic_model)

    prompts_section = ""
    for i, prompt_text in enumerate(prompts):
        prompts_section += f"#### Prompt {i+1}\n```\n{prompt_text}\n```\n\n"

    scores_str = json.dumps(step_scores, indent=2)

    blend_template = _load_blend_template()
    blend_instruction = blend_template.format(
        num_prompts=len(prompts),
        step_scores=scores_str,
        prompts_section=prompts_section,
    )

    client = genai.Client(
        project=config.project,
        location="global",
        vertexai=True,
    )

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config=genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.HIGH,
            ),
        ),
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=blend_instruction)],
        )],
    )

    blended = response.text.strip()
    # Strip markdown code fences if the model wrapped it
    if blended.startswith("```"):
        blended = blended.split("\n", 1)[1] if "\n" in blended else blended[3:]
    if blended.endswith("```"):
        blended = blended[:-3].rstrip()

    logger.info("Blended prompt length: %d chars", len(blended))
    return blended


def _extract_label(model_output: str) -> str:
    """Extract the product_match label from model JSON output. Returns lowercase."""
    if not model_output:
        return ""
    cleaned = model_output.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("product_match", "").strip().lower()
    except (json.JSONDecodeError, TypeError):
        pass
    return cleaned.strip().lower()


def _classify_prediction(predicted_label: str, ground_truth: str) -> str:
    """Classify a prediction as TP/FP/FN/TN for binary Match classification.

    Match = positive class. Everything else (Not_Match, Mismatch, Inconclusive) = negative.
    """
    pred_positive = predicted_label in ("match",)
    gt_positive = ground_truth.lower() in ("match",)

    if pred_positive and gt_positive:
        return "TP"
    elif pred_positive and not gt_positive:
        return "FP"
    elif not pred_positive and gt_positive:
        return "FN"
    else:
        return "TN"


def _run_predictions_on_data(
    full_train_data: pd.DataFrame,
    current_prompt_text: str,
    config: GEPAConfig,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Run predict_fn on all training data and classify each as TP/FP/FN/TN.

    Returns the full_train_data DataFrame with added columns:
    - predicted_output: raw model JSON string
    - predicted_label: extracted label (lowercase)
    - classification: TP/FP/FN/TN
    """
    logger.info("Running predictions on %d items for error classification ...", len(full_train_data))

    # Temporarily register the current prompt so predict_fn can load it
    temp_prompt_name = f"{config.prompt_name}_error_eval"
    if _predict_module._config is not None:
        _predict_module._config.prompt_name = temp_prompt_name
    mlflow.register_prompt(
        name=temp_prompt_name,
        template=current_prompt_text,
        commit_message="Temporary prompt for inter-step error classification",
    )

    results = [None] * len(full_train_data)

    def _predict_row(idx):
        row = full_train_data.iloc[idx]
        inputs = row["inputs"]
        try:
            output = predict.predict_fn(
                reference_image_path=inputs["reference_image_path"],
                image_path=inputs["image_path"],
            )
        except Exception as e:
            logger.warning("Prediction failed for row %d: %s", idx, e)
            output = json.dumps({"product_match": "Inconclusive", "reason": str(e)})
        return idx, output

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_predict_row, i) for i in range(len(full_train_data))]
        for future in concurrent.futures.as_completed(futures):
            idx, output = future.result()
            results[idx] = output

    # Add classification columns
    df = full_train_data.copy()
    df["predicted_output"] = results
    df["predicted_label"] = df["predicted_output"].apply(_extract_label)
    df["classification"] = df.apply(
        lambda row: _classify_prediction(
            row["predicted_label"],
            row["outputs"]["ground_truth"],
        ),
        axis=1,
    )

    counts = df["classification"].value_counts().to_dict()
    logger.info("Classification distribution: TP=%d FP=%d FN=%d TN=%d",
                counts.get("TP", 0), counts.get("FP", 0),
                counts.get("FN", 0), counts.get("TN", 0))

    return df


def _error_aware_subsample(
    classified_data: pd.DataFrame,
    fp_fraction: float = 1.0,
    fn_fraction: float = 1.0,
    tp_fraction: float = 0.3,
    tn_fraction: float = 0.3,
    random_seed: int = 42,
    max_samples: int = 0,
) -> pd.DataFrame:
    """Subsample training data based on TP/FP/FN/TN classification.

    Takes a higher fraction of error cases (FP/FN) and a lower fraction
    of correct cases (TP/TN) to focus GEPA on current failure modes.

    Returns a DataFrame with the same columns as the input minus the
    classification helper columns (predicted_output, predicted_label, classification).
    """
    sampled_parts = []
    fractions = {"TP": tp_fraction, "FP": fp_fraction, "FN": fn_fraction, "TN": tn_fraction}

    for cls, frac in fractions.items():
        subset = classified_data[classified_data["classification"] == cls]
        if len(subset) == 0:
            continue
        n = max(1, int(len(subset) * frac))
        n = min(n, len(subset))
        sampled = subset.sample(n=n, random_state=random_seed, replace=False)
        sampled_parts.append(sampled)
        logger.info("  %s: %d/%d (%.0f%%)", cls, n, len(subset), frac * 100)

    if not sampled_parts:
        logger.warning("No samples after error-aware subsampling! Using full data.")
        result = classified_data.copy()
    else:
        result = pd.concat(sampled_parts, ignore_index=True)

    # Apply max_samples cap if set
    if max_samples > 0 and len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=random_seed, replace=False).reset_index(drop=True)
        logger.info("  Capped to %d samples (max_samples)", max_samples)

    # Drop helper columns — return clean GEPA-compatible DataFrame
    result = result.drop(columns=["predicted_output", "predicted_label", "classification"], errors="ignore")
    result = result.reset_index(drop=True)

    logger.info("Error-aware subsample: %d items", len(result))
    return result


def run_stepwise(
    config: GEPAConfig,
    step_size: int = 3,
    subsample_fraction: float = 0.8,
    chaining: bool = True,
    error_focused: bool = False,
    fp_fraction: float = 1.0,
    fn_fraction: float = 1.0,
    tp_fraction: float = 0.3,
    tn_fraction: float = 0.3,
    error_eval_workers: int = 10,
) -> tuple:
    """
    Execute stepwise GEPA optimization with per-step sub-sampling.

    Args:
        config: GEPAConfig with total num_iterations, data paths, etc.
        step_size: Number of GEPA iterations per step.
        subsample_fraction: Fraction of training data to use per step (0.0-1.0).
                           Used for random sampling (step 0, or when error_focused=False).
        chaining: If True, each step's output prompt feeds into the next step.
                  If False, all steps start from the initial prompt, and results
                  are blended into a single merged prompt at the end.
        error_focused: If True, run predictions between steps and sample by
                      TP/FP/FN/TN classification. Requires chaining=True.
        fp_fraction: Fraction of false positives to include (0.0-1.0).
        fn_fraction: Fraction of false negatives to include (0.0-1.0).
        tp_fraction: Fraction of true positives to include (0.0-1.0).
        tn_fraction: Fraction of true negatives to include (0.0-1.0).
        error_eval_workers: Concurrent workers for inter-step prediction evaluation.

    Returns:
        Tuple of (final_result, final_prompt_text).
        final_result is the result object from the last GEPA step (or blend step).

    Raises:
        ValueError: If error_focused=True and chaining=False (incompatible).
    """
    if error_focused and not chaining:
        raise ValueError(
            "--error_focused is incompatible with --no_chaining. "
            "Error-focused sampling requires chaining because each step's errors "
            "depend on the previous step's prompt. With no-chaining, all steps "
            "use the same initial prompt, producing identical error distributions."
        )
    total_iterations = config.num_iterations
    num_steps = max(1, total_iterations // step_size)
    remainder = total_iterations % step_size
    if error_focused:
        mode_label = "ERROR-FOCUSED CHAINING"
    elif chaining:
        mode_label = "CHAINING"
    else:
        mode_label = "PARALLEL (no-chaining → blend)"

    logger.info("=" * 70)
    logger.info("STEPWISE GEPA [%s]: %d total iterations = %d steps × %d iters/step%s",
                mode_label, total_iterations, num_steps, step_size,
                f" + 1 remainder step of {remainder}" if remainder else "")
    logger.info("Sub-sample fraction: %.1f%%", subsample_fraction * 100)
    logger.info("=" * 70)

    # Set Vertex AI env vars so MLflow's vertex_ai:/ provider picks them up.
    # Force-set (not setdefault) because critic model location may differ.
    os.environ["VERTEX_PROJECT"] = config.project
    os.environ["VERTEX_LOCATION"] = config.critic_model_location

    # 1. Initialise modules
    predict.init(config)
    scorer.init(config.eval_cf_url, score_key=config.eval_score_key)

    # 2. Load training data
    # When error_focused, load ALL training data so error classification sees
    # every item. The subsample_fraction only controls step 0's random sample.
    # Without error_focused, max_train_samples pre-limits the pool.
    load_limit = 0 if error_focused else config.max_train_samples
    full_train_data = data_loader.load_eval_data(
        category=config.category,
        data_dir=config.data_dir,
        mapping_csv=config.mapping_csv,
        limit=load_limit,
        random_seed=config.random_seed,
    )
    full_size = len(full_train_data)
    subsample_size = max(1, int(full_size * subsample_fraction))
    logger.info("Full training data: %d rows, sub-sample size: %d per step",
                full_size, subsample_size)
    if error_focused:
        logger.info("Error-focused mode: error classification will run on all %d items; "
                     "step 0 random sample = %d items", full_size, subsample_size)

    # 3. Load initial prompt
    with open(config.initial_prompt_path, "r") as f:
        initial_prompt_text = f.read()

    current_prompt_text = initial_prompt_text

    # Load reflection prompt template if provided
    reflection_prompt_template = None
    if hasattr(config, "reflection_prompt_template_path") and config.reflection_prompt_template_path:
        with open(config.reflection_prompt_template_path, "r") as f:
            reflection_prompt_template = f.read()
        logger.info("Using custom reflection prompt template from: %s",
                     config.reflection_prompt_template_path)

    # Set experiment
    mlflow.set_experiment(config.experiment_name)

    # Track scores and prompts across steps
    step_scores = []
    step_prompts = []  # For no-chaining blend
    initial_score = None
    final_result = None

    # Build step schedule: num_steps steps of step_size, plus optional remainder
    step_schedule = [step_size] * num_steps
    if remainder > 0:
        step_schedule.append(remainder)

    for step_idx, iters_this_step in enumerate(step_schedule):
        step_num = step_idx + 1
        total_steps = len(step_schedule)

        # Different seed per step for sub-sampling diversity
        step_seed = config.random_seed + step_idx * 7  # deterministic but different

        logger.info("-" * 60)
        logger.info("STEP %d/%d: %d iterations, seed=%d, mode=%s",
                     step_num, total_steps, iters_this_step, step_seed,
                     "chain" if chaining else "independent")
        logger.info("-" * 60)

        # In no-chaining mode, always start from the initial prompt
        if not chaining:
            current_prompt_text = initial_prompt_text

        # Sub-sample training data for this step
        if error_focused and step_idx > 0:
            # Error-focused: classify predictions, then sample by TP/FP/FN/TN
            logger.info("Step %d: running error-focused classification on full data ...", step_num)
            classified = _run_predictions_on_data(
                full_train_data, current_prompt_text, config,
                max_workers=error_eval_workers,
            )
            step_train_data = _error_aware_subsample(
                classified,
                fp_fraction=fp_fraction,
                fn_fraction=fn_fraction,
                tp_fraction=tp_fraction,
                tn_fraction=tn_fraction,
                random_seed=step_seed,
            )
        else:
            # Standard random subsample (step 0, or non-error-focused mode)
            step_train_data = full_train_data.sample(
                n=subsample_size,
                random_state=step_seed,
                replace=False,
            ).reset_index(drop=True)
        logger.info("Step %d: using %d/%d training examples",
                     step_num, len(step_train_data), full_size)

        # Register current prompt with a step-specific name
        step_prompt_name = f"{config.prompt_name}_step{step_num}"

        # Update predict module's config to use step-specific prompt name
        if _predict_module._config is not None:
            _predict_module._config.prompt_name = step_prompt_name

        prompt_version = mlflow.register_prompt(
            name=step_prompt_name,
            template=current_prompt_text,
            commit_message=f"Step {step_num}/{total_steps} input prompt ({'chain' if chaining else 'independent'})",
        )
        prompt_uri = f"prompts:/{step_prompt_name}/{prompt_version.version}"
        logger.info("Step %d: registered prompt URI: %s", step_num, prompt_uri)

        # Build optimizer for this step
        max_metric_calls = iters_this_step * len(step_train_data)
        logger.info("Step %d: max_metric_calls=%d (%d iters × %d samples)",
                     step_num, max_metric_calls, iters_this_step, len(step_train_data))

        gepa_kwargs = {}
        if reflection_prompt_template:
            gepa_kwargs["reflection_prompt_template"] = reflection_prompt_template

        optimizer = ThinkingGepaPromptOptimizer(
            reflection_model=f"vertex_ai:/{config.critic_model}",
            max_metric_calls=max_metric_calls,
            reasoning_effort=config.critic_reasoning_effort or None,
            gepa_kwargs=gepa_kwargs if gepa_kwargs else None,
        )

        # Run GEPA for this step
        logger.info("Step %d: calling mlflow.genai.optimize_prompts ...", step_num)
        result = mlflow.genai.optimize_prompts(
            predict_fn=predict.predict_fn,
            train_data=step_train_data,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=[scorer.weighted_scorer],
        )

        step_initial = getattr(result, "initial_eval_score", None)
        step_final = getattr(result, "final_eval_score", None)

        if step_idx == 0:
            initial_score = step_initial

        step_scores.append({
            "step": step_num,
            "iterations": iters_this_step,
            "seed": step_seed,
            "train_size": len(step_train_data),
            "initial_score": step_initial,
            "final_score": step_final,
        })

        logger.info("Step %d: score %.4f -> %.4f",
                     step_num, step_initial or 0, step_final or 0)

        # Load the optimized prompt
        optimized_prompt = mlflow.load_prompt(step_prompt_name)
        step_output_text = optimized_prompt.template

        # In chaining mode, carry forward to next step
        if chaining:
            current_prompt_text = step_output_text

        # Collect all step prompts for potential blending
        step_prompts.append(step_output_text)
        final_result = result

        logger.info("Step %d complete.", step_num)

    # ── Post-processing ──

    if chaining:
        # In chaining mode, final prompt is the last step's output
        final_prompt_text = current_prompt_text
    else:
        # In no-chaining mode, blend all step prompts into one
        logger.info("=" * 70)
        logger.info("BLENDING %d independent step prompts ...", len(step_prompts))
        logger.info("=" * 70)

        final_prompt_text = _blend_prompts(
            prompts=step_prompts,
            step_scores=step_scores,
            config=config,
        )

        # Save blended prompt to prompts/ directory
        blend_filename = f"system_prompt_blended_{config.prompt_name}.txt"
        blend_path = os.path.join("prompts", blend_filename)
        os.makedirs("prompts", exist_ok=True)
        with open(blend_path, "w") as f:
            f.write(final_prompt_text)
        logger.info("Blended prompt saved to: %s", blend_path)

    # Register the final prompt under the main prompt name
    mlflow.register_prompt(
        name=config.prompt_name,
        template=final_prompt_text,
        commit_message=f"Final {'chained' if chaining else 'blended'} prompt after {len(step_schedule)} steps",
    )

    # Summary
    logger.info("=" * 70)
    logger.info("STEPWISE GEPA COMPLETE [%s]", mode_label)
    logger.info("=" * 70)
    logger.info("Steps: %d", len(step_schedule))
    logger.info("Initial score (step 1): %.4f", initial_score or 0)
    logger.info("Final score (last step): %.4f", step_scores[-1]["final_score"] or 0)
    for s in step_scores:
        logger.info("  Step %d: %.4f -> %.4f (seed=%d, n=%d, iters=%d)",
                     s["step"], s["initial_score"] or 0, s["final_score"] or 0,
                     s["seed"], s["train_size"], s["iterations"])

    # Attach metadata to result for downstream use
    if final_result:
        final_result._stepwise_history = step_scores
        final_result._step_prompts = step_prompts
        final_result._mode = "chaining" if chaining else "blend"

    print(f"\n=== Stepwise Results [{mode_label}] ===")
    print(f"Initial eval score: {initial_score}")
    print(f"Final eval score:   {step_scores[-1]['final_score']}")
    print(f"Steps completed:    {len(step_schedule)}")
    if not chaining:
        print(f"Blended from:       {len(step_prompts)} independent prompts")

    return final_result, final_prompt_text