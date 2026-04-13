import json
import logging
from typing import List, Dict, Any


def summarize_eval_results(results: List[Dict[str, Any]], config: Dict[str, Any], output_file_path: str = None) -> Dict[str, Any]:
    """
    Summarizes the evaluation results for binary Mismatch classification.
    Positive class: Mismatch
    Negative class: Not_Mismatch (includes Match, Inconclusive, anything else)
    """
    n_total = len(results)
    if n_total == 0:
        logging.warning("No results to summarize.")
        return {}

    total_latency = 0.0
    total_input_tokens = 0
    total_input_tokens_text = 0
    total_input_tokens_image = 0
    total_output_tokens = 0
    telemetry_count = 0

    # Binary categories for Mismatch evaluation
    categories = ["Mismatch", "Not_Mismatch"]
    confusion_matrix_grades = {c: {c_in: 0 for c_in in categories} for c in categories}

    def norm_grade(g):
        if not g:
            return "Not_Mismatch"
        g = str(g).lower().strip()
        if g == "mismatch":
            return "Mismatch"
        return "Not_Mismatch"

    for r in results:
        gt = norm_grade(r.get("ground_truth"))

        # Extract model grade from JSON output
        model_grade = "Not_Mismatch"
        if r.get("is_valid_json"):
            try:
                cleaned_text = r["model_output"].strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned_text)
                model_grade = norm_grade(parsed.get("product_match", ""))
            except Exception:
                pass

        confusion_matrix_grades[gt][model_grade] += 1

        tel = r.get("telemetry")
        if tel:
            total_latency += r.get("latency_sec", 0.0)
            total_input_tokens += tel.get("prompt_token_count", 0)
            total_input_tokens_text += tel.get("prompt_token_count_text", 0)
            total_input_tokens_image += tel.get("prompt_token_count_image", 0)
            total_output_tokens += tel.get("candidates_token_count", 0)
            telemetry_count += 1

    # Calculate accuracy from confusion matrix (TP + TN) / total
    tp = confusion_matrix_grades["Mismatch"]["Mismatch"]
    tn = confusion_matrix_grades["Not_Mismatch"]["Not_Mismatch"]
    fp = confusion_matrix_grades["Not_Mismatch"]["Mismatch"]
    fn = confusion_matrix_grades["Mismatch"]["Not_Mismatch"]
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

    # Precision, Recall, F1 for Mismatch class
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    avg_latency = total_latency / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens = total_input_tokens / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens_text = total_input_tokens_text / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens_image = total_input_tokens_image / telemetry_count if telemetry_count > 0 else 0.0
    avg_output_tokens = total_output_tokens / telemetry_count if telemetry_count > 0 else 0.0

    summary = {
        "config": config,
        "aggregated_metrics": {
            "n_total": n_total,
            "accuracy": accuracy,
            "metrics_mismatch": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "count_gt_mismatch": tp + fn
            },
            "telemetry_avg": {
                "latency": avg_latency,
                "input_tokens": avg_input_tokens,
                "input_tokens_text": avg_input_tokens_text,
                "input_tokens_image": avg_input_tokens_image,
                "output_tokens": avg_output_tokens
            },
            "confusion_matrix_grades": confusion_matrix_grades
        },
        "individual_results": results
    }

    if output_file_path:
        try:
            with open(output_file_path, "w") as f:
                json.dump(summary, f, indent=4)
            logging.info(f"Aggregated metrics and config successfully written to {output_file_path}")
        except Exception as e:
            logging.error(f"Failed to write aggregated metrics output file: {e}")

    return summary