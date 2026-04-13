"""
Configuration dataclass for MLflow GEPA prompt optimization.

Centralizes all parameters needed across the GEPA pipeline:
data loading, model invocation, scoring, and optimization.
"""

from dataclasses import dataclass, field


@dataclass
class GEPAConfig:
    """All tuneable knobs for a single GEPA optimization run."""

    # ── GCP ──────────────────────────────────────────────────
    project: str = ""
    location: str = "us-central1"

    # ── Target model (the model whose prompt we optimize) ────
    target_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    top_p: float = 0.95
    thinking_budget: int = 0

    # ── Critic / reflection model (used by GEPA internally) ─
    critic_model: str = "gemini-3.1-pro-preview"
    critic_model_location: str = "global"
    critic_reasoning_effort: str = "high"  # "high", "low", or "" to disable

    # ── Evaluation Cloud Function ────────────────────────────
    eval_cf_name: str = ""
    eval_cf_location: str = "us-central1"
    eval_score_key: str = "match_score"

    # ── Prompt ───────────────────────────────────────────────
    prompt_name: str = "image_match_system_prompt"
    initial_prompt_path: str = (
        "prompts/binary_match_or_not.txt"
    )

    # ── Data ─────────────────────────────────────────────────
    category: str = ""
    data_dir: str = ""
    mapping_csv: str = ""
    max_train_samples: int = 0
    random_seed: int = 42

    # ── GEPA optimizer ───────────────────────────────────────
    num_iterations: int = 10
    experiment_name: str = "IMAGE_MATCH_GEPA"
    reflection_prompt_template_path: str = ""

    # ── Derived properties ───────────────────────────────────
    @property
    def eval_cf_url(self) -> str:
        """Full HTTPS URL for the evaluation Cloud Function."""
        return (
            f"https://{self.eval_cf_location}-{self.project}"
            f".cloudfunctions.net/{self.eval_cf_name}"
        )

    @property
    def max_metric_calls(self) -> int:
        """Upper bound on scorer invocations passed to GepaPromptOptimizer."""
        return self.num_iterations * self.max_train_samples