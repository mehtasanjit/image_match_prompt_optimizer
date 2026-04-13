"""
Custom GepaPromptOptimizer subclass that enables thinking for the
reflection/critic model.

MLflow's GepaPromptOptimizer only passes the reflection model as a
string to GEPA, with no way to configure generation parameters like
thinking_level. This subclass overrides the optimize method to inject
an LM callable with `reasoning_effort="high"` via LiteLLM, which maps
to Gemini 3's native `thinking_level=HIGH`.

Requires: gepa >= 0.1.1 (for gepa.lm.LM) OR litellm (direct fallback).

Usage:
    optimizer = ThinkingGepaPromptOptimizer(
        reflection_model="vertex_ai:/gemini-3.1-pro-preview",
        max_metric_calls=100,
        reasoning_effort="high",  # "high", "low", or None to disable
    )
"""

import logging
import os
from typing import Any
from unittest.mock import patch

from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
from mlflow.genai.optimize.optimizers.base import _EvalFunc
from mlflow.genai.optimize.types import PromptOptimizerOutput

logger = logging.getLogger(__name__)


def _create_thinking_lm(
    model: str,
    reasoning_effort: str,
    vertex_project: str = "",
    vertex_location: str = "global",
):
    """Create a LanguageModel callable with reasoning_effort configured.

    Tries gepa.lm.LM first (requires gepa >= 0.1.1), falls back to a
    direct litellm.completion wrapper.
    """
    try:
        from gepa.lm import LM
        logger.info("Using gepa.lm.LM for thinking LM (gepa >= 0.1.1)")
        return LM(
            model,
            reasoning_effort=reasoning_effort,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
    except ImportError:
        pass

    # Fallback: create a minimal callable that conforms to LanguageModel protocol
    logger.info("gepa.lm.LM not available, using direct litellm wrapper")
    import litellm

    def thinking_lm(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response = litellm.completion(
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            num_retries=3,
            drop_params=True,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
        return response.choices[0].message.content

    return thinking_lm


class ThinkingGepaPromptOptimizer(GepaPromptOptimizer):
    """GepaPromptOptimizer with thinking/reasoning enabled for the reflection model.

    For Gemini 3+ models, LiteLLM maps ``reasoning_effort`` to the native
    ``thinking_level`` parameter:
    - ``reasoning_effort="high"`` → ``thinking_level=HIGH``
    - ``reasoning_effort="low"`` → ``thinking_level=LOW``

    This subclass intercepts the call to ``gepa.optimize()`` and replaces
    the string-based ``reflection_lm`` with a callable configured with
    the appropriate reasoning_effort.

    Args:
        reflection_model: Model URI in format "<provider>:/<model>".
        max_metric_calls: Max evaluation calls during optimization.
        display_progress_bar: Show progress bar.
        reasoning_effort: Thinking level for the reflection model.
            "high" = deep reasoning (thinking_level=HIGH).
            "low" = minimal reasoning (thinking_level=LOW).
            None = disabled (uses standard GepaPromptOptimizer behavior).
        gepa_kwargs: Additional kwargs passed to gepa.optimize().
    """

    def __init__(
        self,
        reflection_model: str,
        max_metric_calls: int = 100,
        display_progress_bar: bool = False,
        reasoning_effort: str | None = "high",
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            reflection_model=reflection_model,
            max_metric_calls=max_metric_calls,
            display_progress_bar=display_progress_bar,
            gepa_kwargs=gepa_kwargs,
        )
        self.reasoning_effort = reasoning_effort

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
    ) -> PromptOptimizerOutput:
        """Override to inject thinking-enabled LM into gepa.optimize() call.

        The parent's optimize() merges kwargs as:
            self.gepa_kwargs | {"reflection_lm": string, ...}
        The right side wins, so we can't inject via gepa_kwargs.

        Instead, we patch gepa.optimize to intercept the kwargs and
        replace reflection_lm with our thinking-enabled callable.
        """
        if not self.reasoning_effort:
            return super().optimize(eval_fn, train_data, target_prompts, enable_tracking)

        import gepa
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        provider, model = _parse_model_uri(self.reflection_model)
        litellm_model = f"{provider}/{model}"

        # Read project/location from env (set by run_gepa.py before calling optimizer)
        vertex_project = os.environ.get("VERTEX_PROJECT", "")
        vertex_location = os.environ.get("VERTEX_LOCATION", "global")

        thinking_lm = _create_thinking_lm(
            litellm_model,
            self.reasoning_effort,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )

        logger.info(
            "ThinkingGepaPromptOptimizer: reasoning_effort=%s "
            "(model=%s, location=%s → thinking_level=%s)",
            self.reasoning_effort,
            litellm_model,
            vertex_location,
            self.reasoning_effort.upper(),
        )

        original_optimize = gepa.optimize

        def patched_optimize(**kwargs):
            kwargs["reflection_lm"] = thinking_lm
            logger.info("Patched gepa.optimize: reflection_lm replaced with thinking LM")
            return original_optimize(**kwargs)

        with patch.object(gepa, "optimize", patched_optimize):
            return super().optimize(eval_fn, train_data, target_prompts, enable_tracking)