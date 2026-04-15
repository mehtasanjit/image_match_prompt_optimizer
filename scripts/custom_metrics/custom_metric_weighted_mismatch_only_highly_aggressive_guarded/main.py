import functions_framework
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Highly Aggressive Guarded (~93% precision + recall floor) | TP=+1, FP=-15, FN=-3, TN=+0.5 | FP:FN ratio 5:1
WEIGHTS = {"tp": 1, "fp": -15, "fn": -3, "tn": 0.5}


def evaluate(model_response_text, ground_truth):
    extracted = ""
    try:
        cleaned = str(model_response_text).strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            extracted = parsed.get("product_match", "").strip()
    except Exception:
        extracted = str(model_response_text).strip()

    if not extracted:
        return 0.0

    pred_mismatch = extracted.lower() == "mismatch"
    gt_mismatch = str(ground_truth).lower().strip() == "mismatch"

    if pred_mismatch and gt_mismatch:
        return WEIGHTS["tp"]
    elif pred_mismatch and not gt_mismatch:
        return WEIGHTS["fp"]
    elif not pred_mismatch and gt_mismatch:
        return WEIGHTS["fn"]
    else:
        return WEIGHTS["tn"]


@functions_framework.http
def product_mismatch_weighted_metric(request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {"error": "Invalid JSON"}, 400

        response_text = request_json.get("response", "")
        ground_truth = request_json.get("target", "")

        if not response_text or not ground_truth:
            return {"mismatch_score": 0.0}

        score = evaluate(response_text, ground_truth)
        logger.info(f"highly_aggressive_guarded | Score: {score} | GT: {ground_truth}")
        return {"mismatch_score": score}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}, 500