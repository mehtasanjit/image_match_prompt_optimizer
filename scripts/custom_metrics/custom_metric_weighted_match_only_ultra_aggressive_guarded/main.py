import functions_framework
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ultra Aggressive Guarded (~95% precision + recall floor) | TP=+1, FP=-25, FN=-4, TN=+0.5 | FP:FN ratio 6.25:1
WEIGHTS = {"tp": 1, "fp": -25, "fn": -4, "tn": 0.5}


def evaluate(model_response_text, ground_truth):
    extracted_match = ""
    try:
        cleaned = str(model_response_text).strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            extracted_match = parsed.get("product_match", "").strip()
    except Exception:
        extracted_match = str(model_response_text).strip()

    if not extracted_match:
        return 0.0

    pred_match = extracted_match.lower() == "match"
    gt_match = str(ground_truth).lower().strip() == "match"

    if pred_match and gt_match:
        return WEIGHTS["tp"]
    elif pred_match and not gt_match:
        return WEIGHTS["fp"]
    elif not pred_match and gt_match:
        return WEIGHTS["fn"]
    else:
        return WEIGHTS["tn"]


@functions_framework.http
def product_match_weighted_metric(request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {"error": "Invalid JSON"}, 400

        response_text = request_json.get("response", "")
        ground_truth = request_json.get("target", "")

        if not response_text or not ground_truth:
            return {"match_score": 0.0}

        score = evaluate(response_text, ground_truth)
        logger.info(f"ultra_aggressive_guarded | Score: {score} | GT: {ground_truth}")
        return {"match_score": score}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}, 500