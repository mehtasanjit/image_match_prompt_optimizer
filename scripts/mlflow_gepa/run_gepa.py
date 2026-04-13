"""
Main GEPA orchestrator for prompt optimization.

Wires together data_loader, config, predict, and scorer modules,
registers the initial prompt in MLflow, sets up the experiment,
and calls mlflow.genai.optimize_prompts().
"""

import argparse
import logging
import os

import mlflow
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

from mlflow_gepa.config import GEPAConfig
from mlflow_gepa import data_loader, predict, scorer
from mlflow_gepa.thinking_optimizer import ThinkingGepaPromptOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)



def run(config: GEPAConfig) -> tuple:
    """
    Execute a full GEPA optimization run.

    Steps:
        1. Initialise predict and scorer modules.
        2. Load and (optionally) sub-sample training data.
        3. Register the initial prompt in MLflow.
        4. Build the GepaPromptOptimizer.
        5. Call mlflow.genai.optimize_prompts().
        6. Print results and the final optimized prompt.

    Args:
        config: Fully populated GEPAConfig instance.
    """
    logger.info("--- GEPA run starting ---")
    logger.info("Target model: %s (location=%s)", config.target_model, config.location)
    logger.info("Critic model: %s", config.critic_model)
    logger.info("Eval CF URL:  %s", config.eval_cf_url)

    # Set Vertex AI env vars so MLflow's vertex_ai:/ provider picks them up.
    # Force-set VERTEX_LOCATION (not setdefault) because the critic model
    # location may differ from the target model location set earlier.
    os.environ["VERTEX_PROJECT"] = config.project
    os.environ["VERTEX_LOCATION"] = config.critic_model_location

    # 1. Initialise modules
    predict.init(config)
    scorer.init(config.eval_cf_url, score_key=config.eval_score_key)

    # 2. Load training data
    train_data = data_loader.load_eval_data(
        category=config.category,
        data_dir=config.data_dir,
        mapping_csv=config.mapping_csv,
        limit=config.max_train_samples,
        random_seed=config.random_seed,
    )
    logger.info("Training data: %d rows", len(train_data))

    # 3. Register initial prompt
    with open(config.initial_prompt_path, "r") as f:
        initial_prompt_template = f.read()

    prompt_version = mlflow.register_prompt(
        name=config.prompt_name,
        template=initial_prompt_template,
        commit_message="Initial system prompt for product image matching",
    )
    prompt_uri = f"prompts:/{config.prompt_name}/{prompt_version.version}"
    logger.info("Registered prompt URI: %s", prompt_uri)

    # 4. Build optimizer
    effective_max_metric_calls = config.num_iterations * len(train_data)
    logger.info("max_metric_calls: %d (iterations=%d x samples=%d)",
                effective_max_metric_calls, config.num_iterations, len(train_data))

    gepa_kwargs = {}
    if hasattr(config, "reflection_prompt_template_path") and config.reflection_prompt_template_path:
        with open(config.reflection_prompt_template_path, "r") as f:
            gepa_kwargs["reflection_prompt_template"] = f.read()
        logger.info("Using custom reflection prompt template from: %s", config.reflection_prompt_template_path)

    optimizer = ThinkingGepaPromptOptimizer(
        reflection_model=f"vertex_ai:/{config.critic_model}",
        max_metric_calls=effective_max_metric_calls,
        reasoning_effort=config.critic_reasoning_effort or None,
        gepa_kwargs=gepa_kwargs if gepa_kwargs else None,
    )

    # 5. Run optimization
    mlflow.set_experiment(config.experiment_name)

    logger.info("Calling mlflow.genai.optimize_prompts ...")
    result = mlflow.genai.optimize_prompts(
        predict_fn=predict.predict_fn,
        train_data=train_data,
        prompt_uris=[prompt_uri],
        optimizer=optimizer,
        scorers=[scorer.weighted_scorer],
    )

    # 6. Report results
    logger.info("--- Optimization complete ---")
    print("\n=== Results ===")
    print(f"Initial eval score: {result.initial_eval_score}")
    print(f"Final eval score:   {result.final_eval_score}")

    final_prompt = mlflow.load_prompt(config.prompt_name)
    print("\n=== Final Optimized Prompt ===")
    print(final_prompt.template)

    return result, final_prompt.template


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------

def _parse_args() -> GEPAConfig:
    """Build a GEPAConfig from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MLflow GEPA prompt optimization for product image matching.",
    )

    # GCP
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP location for GenAI client")

    # Target model
    parser.add_argument("--target_model", default="gemini-2.5-flash", help="Vision model to optimize")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Thinking budget (0 = disabled)")

    # Critic model
    parser.add_argument("--critic_model", default="gemini-3.1-pro-preview", help="Reflection model for GEPA")
    parser.add_argument("--critic_model_location", default="global", help="GCP location for the critic/reflection model")
    parser.add_argument("--critic_reasoning_effort", default="high",
                        help="Thinking level for the critic model: 'high', 'low', or '' to disable (default: high)")

    # Evaluation Cloud Function
    parser.add_argument("--eval_cf_name", required=True, help="Cloud Function name for evaluation")
    parser.add_argument("--eval_cf_location", default="us-central1", help="Cloud Function location")
    parser.add_argument("--eval_score_key", default="match_score",
                        help="JSON key returned by the eval CF containing the numeric score")

    # Prompt
    parser.add_argument("--prompt_name", default="image_match_system_prompt", help="MLflow prompt registry name")
    parser.add_argument(
        "--initial_prompt",
        default="prompts/binary_match_or_not.txt",
        help="Path to initial system prompt file",
    )

    # Data
    parser.add_argument("--category", default=None,
                        help="Product category (e.g. smartwatch, sandal). Used for auto-resolving mapping CSVs. "
                             "If not provided, --mapping_csv is required.")
    parser.add_argument("--data_dir", required=True, help="Directory with mapping CSV and images")
    parser.add_argument("--mapping_csv", default="", help="Explicit path to mapping CSV (optional)")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Max training samples (0 = all)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sub-sampling")

    # GEPA optimizer
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of GEPA iterations")
    parser.add_argument("--experiment_name", default="IMAGE_MATCH_GEPA", help="MLflow experiment name")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging for scorer/predict")

    args = parser.parse_args()

    # Apply debug logging if requested
    if args.debug:
        logging.getLogger("mlflow_gepa").setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled for mlflow_gepa")

    return GEPAConfig(
        project=args.project,
        location=args.location,
        target_model=args.target_model,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_budget=args.thinking_budget,
        critic_model=args.critic_model,
        critic_model_location=args.critic_model_location,
        critic_reasoning_effort=args.critic_reasoning_effort,
        eval_cf_name=args.eval_cf_name,
        eval_cf_location=args.eval_cf_location,
        eval_score_key=args.eval_score_key,
        prompt_name=args.prompt_name,
        initial_prompt_path=args.initial_prompt,
        category=args.category,
        data_dir=args.data_dir,
        mapping_csv=args.mapping_csv,
        max_train_samples=args.max_train_samples,
        random_seed=args.random_seed,
        num_iterations=args.num_iterations,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    config = _parse_args()
    run(config)