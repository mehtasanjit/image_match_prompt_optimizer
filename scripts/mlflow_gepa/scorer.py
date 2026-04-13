"""
Custom MLflow scorer for product image match evaluation.

Normalises the LLM output label into the three categories the
Cloud Function expects - Match, Mismatch, Inconclusive - then
posts each prediction/ground-truth pair to the deployed Cloud
Function and returns the weighted score.
"""

import json
import logging

import requests
from mlflow.genai.scorers import scorer

logger = logging.getLogger(__name__)

# -- module-level state (set by init()) --
_eval_cf_url: str = ""
_score_key: str = "match_score"

# -- label normalisation map (case-insensitive via .lower()) --
_LABEL_MAP: dict[str, str] = {
    "match": "Match",
    "mismatch": "Mismatch",
    "not_match": "Mismatch",
    "not_mismatch": "Inconclusive",
    "not_inconclusive": "Not_Inconclusive",
    "inconclusive": "Inconclusive",
}


# -----------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------

def init(eval_cf_url: str, score_key: str | None = None) -> None:
    """Bind module to a specific Cloud Function endpoint.

    Args:
        eval_cf_url: Full HTTPS URL of the evaluation Cloud Function.
        score_key: JSON key the CF returns with the numeric score.
                   Defaults to 'match_score'.
    """
    global _eval_cf_url, _score_key
    _eval_cf_url = eval_cf_url
    if score_key is not None:
        _score_key = score_key
    logger.info("scorer.init -> eval_cf_url=%s, score_key=%s", _eval_cf_url, _score_key)


def _normalise_label(raw: str) -> str:
    """Map a label to Match / Mismatch / Inconclusive.

    Known labels (case-insensitive):
        Match        -> Match
        Mismatch     -> Mismatch
        Not_Match    -> Mismatch
        Inconclusive -> Inconclusive

    Anything else defaults to Inconclusive.
    """
    key = raw.strip().lower()
    normalised = _LABEL_MAP.get(key, "Inconclusive")
    if key not in _LABEL_MAP:
        logger.warning("_normalise_label: unrecognised label '%s' -> Inconclusive", raw.strip())
    return normalised


def _extract_label(raw_output: str) -> str:
    """Pull 'product_match' from a JSON string and normalise it.

    If the output is not valid JSON or lacks the key, the raw string
    itself is normalised as a last resort.
    """
    cleaned = raw_output.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            label = parsed.get("product_match", "")
            if label:
                return _normalise_label(str(label))
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback - try normalising the raw string directly
    return _normalise_label(cleaned)


# -----------------------------------------------------------------
# MLflow scorer
# -----------------------------------------------------------------

@scorer(name="weighted_precision_score")
def weighted_scorer(*, inputs, outputs, expectations):
    """Score a single prediction against ground truth via the Cloud Function.

    Args:
        inputs: dict with reference_image_path, image_path (unused here).
        outputs: raw LLM output string (full JSON including reasoning).
        expectations: dict with 'ground_truth' key.

    Returns:
        float score from the Cloud Function, or -1.0 on error.
    """
    prediction = _extract_label(str(outputs))

    # Debug: log raw expectations to diagnose missing ground_truth
    logger.debug("scorer: raw expectations type=%s, value=%s", type(expectations).__name__, str(expectations)[:200])

    # MLflow GEPA wraps outputs column under 'expected_response'
    exp = expectations.get("expected_response", expectations)
    if isinstance(exp, dict):
        gt_raw = exp.get("ground_truth", "Inconclusive")
    else:
        gt_raw = str(exp)
    target = _normalise_label(str(gt_raw))

    if not _eval_cf_url:
        logger.error("scorer: _eval_cf_url is not set - call scorer.init() first")
        return -1.0

    payload = {"response": prediction, "target": target}

    try:
        resp = requests.post(_eval_cf_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        score = float(result.get(_score_key, -1.0))
        logger.debug(
            "scorer: prediction=%s, target=%s, score=%s",
            prediction, target, score,
        )
        return score
    except Exception as exc:
        logger.error("scorer: Cloud Function call failed - %s", exc)
        return -1.0