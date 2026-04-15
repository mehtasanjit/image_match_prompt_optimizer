import functions_framework
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Moderate (~87% precision target) | TP=+1, FP=-7, FN=-1, TN=0 | FP:FN ratio 7:1
WEIGHTS = {"tp": 1, "fp": -7, "fn": -1, "tn": 0}


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
        logger.info(f"moderate | Score: {score} | GT: {ground_truth}")
        return {"mismatch_score": score}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}, 500