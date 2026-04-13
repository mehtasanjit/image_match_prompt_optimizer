import json
import logging
from typing import List, Dict, Any

def summarize_eval_results(results: List[Dict[str, Any]], config: Dict[str, Any], output_file_path: str = None) -> Dict[str, Any]:
    """
    Summarizes the evaluation results and telemetry, calculating metrics like precision, recall, F1,
    and formatting them into a standardized JSON structure alongside the input config.
    """
    n_total = len(results)
    if n_total == 0:
        logging.warning("No results to summarize.")
        return {}
        
    aligned_count = 0
    total_latency = 0.0
    total_input_tokens = 0
    total_input_tokens_text = 0
    total_input_tokens_image = 0
    total_output_tokens = 0
    telemetry_count = 0
    
    # Matrix Categories for Product Match Binary Evaluation
    categories = ["Match", "Not_Match"]
    confusion_matrix_grades = {c: {c_in: 0 for c_in in categories} for c in categories}
    
    def norm_grade(g):
        if not g:
            return "Not_Match"
        g = str(g).lower().strip()
        if g == "match": return "Match"
        return "Not_Match"
        
    for r in results:
        score = r.get("score", 0.0)
        # Global Accuracy logic (Match==Match, Not_Match==Not_Match)
        if score == 1.0:
            aligned_count += 1
            
        gt = norm_grade(r.get("ground_truth"))
        
        # Extract model grade reliably if valid JSON
        model_grade = "Not_Match"
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
    tp_match = confusion_matrix_grades["Match"]["Match"]
    tn = confusion_matrix_grades["Not_Match"]["Not_Match"]
    alignment_accuracy = (tp_match + tn) / n_total if n_total > 0 else 0.0
    
    # Calculate Precision, Recall, F1 SPECIFICALLY for the 'Match' class
    fp_match = confusion_matrix_grades["Not_Match"]["Match"]
    fn_match = confusion_matrix_grades["Match"]["Not_Match"]
    
    prec_match = tp_match / (tp_match + fp_match) if (tp_match + fp_match) > 0 else 0.0
    rec_match = tp_match / (tp_match + fn_match) if (tp_match + fn_match) > 0 else 0.0
    f1_match = 2 * (prec_match * rec_match) / (prec_match + rec_match) if (prec_match + rec_match) > 0 else 0.0
    
    avg_latency = total_latency / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens = total_input_tokens / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens_text = total_input_tokens_text / telemetry_count if telemetry_count > 0 else 0.0
    avg_input_tokens_image = total_input_tokens_image / telemetry_count if telemetry_count > 0 else 0.0
    avg_output_tokens = total_output_tokens / telemetry_count if telemetry_count > 0 else 0.0
    
    summary = {
        "config": config,
        "aggregated_metrics": {
            "n_total": n_total,
            "accuracy": alignment_accuracy,
            "metrics_match": {
                "precision": prec_match,
                "recall": rec_match,
                "f1": f1_match,
                "count_gt_match": tp_match + fn_match
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
