import functions_framework
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_product_mismatch(model_response_text, ground_truth):
    """
    Evaluates alignment between the model's reported 'product_match'
    and the ground truth for binary Mismatch classification.

    Binary framing:
      - Positive class: Mismatch
      - Negative class: Not_Mismatch (includes Match, Inconclusive, anything else)

    Returns 1.0 if model and GT agree, 0.0 otherwise.
    """
    # 1. Extract 'product_match' from the model response JSON
    extracted_label = ""
    try:
        cleaned_text = str(model_response_text).strip()
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()

        response_json = json.loads(cleaned_text)
        if isinstance(response_json, dict):
            extracted_label = response_json.get("product_match", "").strip()
    except Exception as e:
        logger.warning(
            f"JSON Parsing Failed: {e}. Falling back to raw text. "
            f"Text: {str(model_response_text)[:100]}..."
        )
        extracted_label = str(model_response_text).strip()

    if not extracted_label:
        return 0.0

    # 2. Normalize to binary: Mismatch vs Not_Mismatch
    model_norm = extracted_label.lower().strip()
    gt_norm = str(ground_truth).lower().strip()

    is_model_mismatch = model_norm == "mismatch"
    is_gt_mismatch = gt_norm == "mismatch"

    # Binary scoring
    if is_model_mismatch and is_gt_mismatch:
        return 1.0  # TP
    elif not is_model_mismatch and not is_gt_mismatch:
        return 1.0  # TN
    else:
        return 0.0  # FP or FN


@functions_framework.http
def product_mismatch_custom_metric(request):
    """
    HTTP Cloud Function for binary Mismatch evaluation.

    Expected Input JSON Payload:
    - response: The model's output text (JSON string with product_match field).
    - target: The ground truth label.

    Returns:
    - mismatch_score: 1.0 or 0.0
    """
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {"error": "Invalid JSON"}, 400

        response_text = request_json.get("response", "")
        ground_truth = request_json.get("target", "")

        if not response_text or not ground_truth:
            logger.warning("Missing 'response' or 'target' in payload.")
            return {"mismatch_score": 0.0}

        score = evaluate_product_mismatch(response_text, ground_truth)

        logger.info(
            f"Scored: {score} | GT: {ground_truth} | "
            f"Model: {str(response_text)[:50]}..."
        )

        return {"mismatch_score": score}

    except Exception as e:
        logger.error(f"Error in product_mismatch_custom_metric: {e}")
        return {"error": str(e)}, 500