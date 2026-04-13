"""
Multi-step (chained) product image comparison pipeline.

Runs two prompts in sequence to produce a 3-class output (Match / Mismatch / Inconclusive).

Default flow (--first_step match):
  1. Run match prompt → if "Match" → final = Match, DONE
  2. Otherwise → run mismatch prompt → if "Mismatch" → final = Mismatch
  3. Otherwise → final = Inconclusive

Reversed flow (--first_step mismatch):
  1. Run mismatch prompt → if "Mismatch" → final = Mismatch, DONE
  2. Otherwise → run match prompt → if "Match" → final = Match
  3. Otherwise → final = Inconclusive

Usage:
    .venv/bin/python scripts/pipeline/run_multi_step_pipeline_with_eval.py \\
        --category smartwatch \\
        --data_dir data/sampled/smartwatch_1/test \\
        --system_prompt_match prompts/binary_match_or_not.txt \\
        --system_prompt_mismatch prompts/binary_mismatch_or_not.txt \\
        --first_step match \\
        --eval_cloud_function_name img_match_weighted_guarded \\
        --project my-gcp-project \\
        --location us-central1 \\
        --eval_cloud_function_location us-central1 \\
        --limit 0 \\
        --output_file_path results.json
"""

import os
import json
import argparse
import csv
import time
import requests
import mimetypes
import logging
from typing import Dict, Any, List
import sys
import io
import base64
from PIL import Image
import concurrent.futures

import pandas as pd

from google import genai
from google.genai import types as genai_types

# ─────────────────────────────────────────────────────────
# Image helpers (shared with single-step pipeline)
# ─────────────────────────────────────────────────────────

def convert_image_to_base64_string(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def load_image_as_bytes(image_source: str) -> bytes:
    try:
        if Image:
            with Image.open(image_source) as img:
                b64_str = convert_image_to_base64_string(img)
                return base64.b64decode(b64_str)
        else:
            with open(image_source, 'rb') as f:
                return f.read()
    except Exception as e:
        print(f"Error loading image {image_source}: {e}")
        return b""


def get_mime_type(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "image/jpeg"


# ─────────────────────────────────────────────────────────
# Data loading (CSV-based, matching binary pipeline)
# ─────────────────────────────────────────────────────────

def csv_to_eval_data(csv_path: str) -> List[Dict[str, Any]]:
    """Convert a mapping CSV to evaluation data (list of dicts).

    Expected CSV columns: id, ground_truth, reference_image_filename, image_filename.
    Additional columns are included as metadata.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = ["id", "ground_truth", "reference_image_filename", "image_filename"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}. Columns: {list(df.columns)}")

    items = []
    for _, row in df.iterrows():
        ref_files = [f.strip() for f in str(row["reference_image_filename"]).split(",") if f.strip()]
        img_files = [f.strip() for f in str(row["image_filename"]).split(",") if f.strip()]

        item = {
            "id": row["id"],
            "ground_truth": str(row["ground_truth"]).strip().capitalize(),
            "reference_image_filenames_list": [f"reference_images/{f}" for f in ref_files],
            "image_filenames_list": [f"images/{f}" for f in img_files],
        }

        # Include extra columns as metadata
        skip = set(required)
        for col in df.columns:
            if col not in skip:
                val = row[col]
                item[col] = val if pd.notna(val) else None

        items.append(item)

    return items


# ─────────────────────────────────────────────────────────
# Label extraction
# ─────────────────────────────────────────────────────────

def _extract_product_match_label(model_text: str) -> str:
    """Extract product_match label from model JSON output. Returns lowercase."""
    if not model_text:
        return ""
    cleaned = model_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("product_match", "").strip().lower()
    except (json.JSONDecodeError, TypeError):
        pass
    return cleaned.strip().lower()


# ─────────────────────────────────────────────────────────
# Single model call (shared logic)
# ─────────────────────────────────────────────────────────

def _call_model(
    ref_bytes: bytes,
    ref_mime: str,
    img_bytes: bytes,
    img_mime: str,
    system_prompt: str,
    client: genai.Client,
    category_name: str,
    model_name: str,
    temperature: float,
    top_p: float,
    thinking_budget: int,
    thinking_level: str,
) -> tuple:
    """Call the model with given images and prompt. Returns (model_text, telemetry, latency)."""

    display_category = category_name.capitalize() if category_name else "Not Specified"
    parts = [
        genai_types.Part.from_text(text="**Product Category:**\n"),
        genai_types.Part.from_text(text=display_category),
        genai_types.Part.from_text(text="**Reference Image:**\n"),
        genai_types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
        genai_types.Part.from_text(text="\n**Candidate Image:**\n"),
        genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
    ]

    user_content = genai_types.Content(role="user", parts=parts)

    config_args = {
        "temperature": temperature,
        "top_p": top_p,
        "system_instruction": system_prompt,
    }

    if "gemini-3" in model_name:
        t_level = genai_types.ThinkingLevel.MINIMAL
        if thinking_level:
            try:
                t_level = getattr(genai_types.ThinkingLevel, thinking_level.upper())
            except AttributeError:
                pass
        config_args["thinking_config"] = genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=t_level,
        )
    else:
        config_args["thinking_config"] = genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=thinking_budget,
        )

    config = genai_types.GenerateContentConfig(**config_args)

    start = time.time()
    telemetry = {}
    try:
        response = client.models.generate_content(
            model=model_name, config=config, contents=[user_content]
        )
        model_text = response.text
        usage = response.usage_metadata
        prompt_text_tokens = 0
        prompt_image_tokens = 0
        if usage and usage.prompt_tokens_details:
            for detail in usage.prompt_tokens_details:
                m_str = str(detail.modality)
                if m_str in ("MediaModality.TEXT", "TEXT"):
                    prompt_text_tokens += detail.token_count
                elif m_str in ("MediaModality.IMAGE", "IMAGE"):
                    prompt_image_tokens += detail.token_count
        telemetry = {
            "prompt_token_count": (usage.prompt_token_count if usage else 0) or 0,
            "prompt_token_count_text": prompt_text_tokens,
            "prompt_token_count_image": prompt_image_tokens,
            "candidates_token_count": (usage.candidates_token_count if usage else 0) or 0,
            "thoughts_token_count": (getattr(usage, "thoughts_token_count", 0) if usage else 0) or 0,
            "total_token_count": (usage.total_token_count if usage else 0) or 0,
        }
    except Exception as e:
        model_text = str(e)
    latency = time.time() - start
    return model_text, telemetry, latency


# ─────────────────────────────────────────────────────────
# Multi-step item processor
# ─────────────────────────────────────────────────────────

def process_item_multi_step(
    item: Dict[str, Any],
    data_dir: str,
    match_prompt: str,
    mismatch_prompt: str,
    first_step: str,
    client: genai.Client,
    project: str,
    eval_cloud_function_name: str,
    eval_cloud_function_location: str,
    category_name: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    top_p: float = 0.95,
    thinking_budget: int = 0,
    thinking_level: str = None,
) -> Dict[str, Any]:

    # 1. Load images once
    ref_filenames = item.get("reference_image_filenames_list", [])
    img_filenames = item.get("image_filenames_list", [])
    if not ref_filenames or not img_filenames:
        return {"error": "Missing reference or candidate image in data."}

    ref_path = os.path.join(data_dir, ref_filenames[0])
    img_path = os.path.join(data_dir, img_filenames[0])
    try:
        ref_bytes = load_image_as_bytes(ref_path)
        img_bytes = load_image_as_bytes(img_path)
    except Exception as e:
        return {"error": f"Failed to load images: {e}"}

    ref_mime = get_mime_type(ref_path)
    img_mime = get_mime_type(img_path)

    common_kwargs = dict(
        ref_bytes=ref_bytes, ref_mime=ref_mime,
        img_bytes=img_bytes, img_mime=img_mime,
        client=client, category_name=category_name,
        model_name=model_name, temperature=temperature,
        top_p=top_p, thinking_budget=thinking_budget,
        thinking_level=thinking_level,
    )

    # 2. Determine step order
    if first_step == "match":
        step1_prompt = match_prompt
        step1_positive = "match"
        step2_prompt = mismatch_prompt
        step2_positive = "mismatch"
    else:
        step1_prompt = mismatch_prompt
        step1_positive = "mismatch"
        step2_prompt = match_prompt
        step2_positive = "match"

    # 3. Step 1
    step1_text, step1_tel, step1_latency = _call_model(
        system_prompt=step1_prompt, **common_kwargs
    )
    step1_label = _extract_product_match_label(step1_text)

    total_latency = step1_latency
    total_telemetry = dict(step1_tel)  # copy

    # Step 1 decision
    if step1_label == step1_positive:
        # Positive class confirmed at step 1 → done
        final_label = step1_positive.capitalize()
        steps_used = 1
        step2_text = None
    else:
        # Step 1 was NOT the positive class → proceed to step 2
        step2_text, step2_tel, step2_latency = _call_model(
            system_prompt=step2_prompt, **common_kwargs
        )
        step2_label = _extract_product_match_label(step2_text)
        total_latency += step2_latency

        # Aggregate telemetry
        for k in total_telemetry:
            total_telemetry[k] = total_telemetry.get(k, 0) + step2_tel.get(k, 0)

        if step2_label == step2_positive:
            final_label = step2_positive.capitalize()
        else:
            final_label = "Inconclusive"
        steps_used = 2

    # 4. Build a synthetic model_output JSON so the summarizer can parse it
    synthetic_output = json.dumps({
        "product_in_reference_image": "See step outputs",
        "product_in_candidate_image": "See step outputs",
        "reason": f"Multi-step: step1={step1_label}, "
                  + (f"step2={_extract_product_match_label(step2_text)}" if step2_text else "no step2 needed"),
        "product_match": final_label,
    })

    # 5. Check JSON validity (always valid since we built it)
    is_valid_json = True

    # 6. Call eval CF with 3-class result
    cf_url = f"https://{eval_cloud_function_location}-{project}.cloudfunctions.net/{eval_cloud_function_name}"
    payload = {
        "response": synthetic_output,
        "target": item.get("ground_truth", ""),
    }
    try:
        eval_resp = requests.post(cf_url, json=payload, timeout=30)
        eval_resp.raise_for_status()
        eval_result = eval_resp.json()
    except Exception as e:
        eval_result = {"error": str(e)}

    return {
        "id": item.get("id"),
        "ground_truth": item.get("ground_truth"),
        "model_output": synthetic_output,
        "is_valid_json": is_valid_json,
        "score": eval_result.get("score", 0.0),
        "eval_payload": eval_result,
        "latency_sec": total_latency,
        "telemetry": total_telemetry,
        "multi_step": {
            "first_step": first_step,
            "steps_used": steps_used,
            "step1_label": step1_label,
            "step1_raw": step1_text,
            "step2_raw": step2_text,
            "final_label": final_label,
        },
    }


# ─────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────

def summarize_multi_step_results(results, config, output_file_path=None):
    """Summarize multi-step pipeline results with 3-class confusion matrix."""
    labels = ["Match", "Mismatch", "Inconclusive"]
    confusion = {gt: {pred: 0 for pred in labels} for gt in labels}
    total_latency = 0
    total_items = len(results)
    step1_only = 0
    step2_needed = 0
    total_score = 0
    valid_count = 0

    for r in results:
        if "error" in r:
            continue
        gt = r.get("ground_truth", "")
        ms = r.get("multi_step", {})
        pred = ms.get("final_label", "")
        if gt in confusion and pred in confusion.get(gt, {}):
            confusion[gt][pred] += 1
        total_latency += r.get("latency_sec", 0)
        if ms.get("steps_used") == 1:
            step1_only += 1
        else:
            step2_needed += 1
        total_score += r.get("score", 0)
        valid_count += 1

    avg_score = total_score / valid_count if valid_count else 0
    avg_latency = total_latency / valid_count if valid_count else 0

    # Per-class precision/recall/f1
    metrics_per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics_per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count_gt": sum(confusion[label].values()),
        }

    # Accuracy
    correct = sum(confusion[l][l] for l in labels)
    accuracy = correct / valid_count if valid_count else 0

    summary = {
        "config": config,
        "aggregated_metrics": {
            "n_total": total_items,
            "n_valid": valid_count,
            "accuracy": accuracy,
            "avg_score": avg_score,
            "avg_latency_sec": avg_latency,
            "step1_resolved_count": step1_only,
            "step2_resolved_count": step2_needed,
            "metrics_per_class": metrics_per_class,
            "confusion_matrix": confusion,
        },
        "results": results,
    }

    if output_file_path:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info("Results saved to %s", output_file_path)

    return summary


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Multi-step (chained) product image comparison pipeline with 3-class eval"
    )

    # Data
    parser.add_argument("--category", default=None,
                        help="Product category name (used in prompt and for auto-resolving mapping CSV)")
    parser.add_argument("--data_dir", required=True, help="Base data directory with images and mapping CSV")
    parser.add_argument("--mapping_csv", default=None,
                        help="Explicit path to mapping CSV. If omitted, resolved as <data_dir>/<category>_mapping.csv")
    parser.add_argument("--limit", type=int, default=5, help="Items to evaluate (0 = all)")

    # Prompts
    parser.add_argument("--system_prompt_match", required=True, help="Path to match system prompt")
    parser.add_argument("--system_prompt_mismatch", required=True, help="Path to mismatch system prompt")
    parser.add_argument("--first_step", choices=["match", "mismatch"], default="match",
                        help="Which prompt runs first (default: match)")

    # Eval CF
    parser.add_argument("--eval_cloud_function_name", required=True, help="Cloud Function name for evaluation")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--location", required=True, help="GCP Location for GenAI")
    parser.add_argument("--eval_cloud_function_location", required=True, help="GCP location for Cloud Function")

    # Model
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top P")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Thinking budget (Gemini 2.5)")
    parser.add_argument("--thinking_level", type=str, default=None, help="Thinking level (Gemini 3)")

    # Output
    parser.add_argument("--output_file_path", default=None, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers")

    args = parser.parse_args()

    # Load prompts
    with open(args.system_prompt_match, "r") as f:
        match_prompt_text = f.read()
    with open(args.system_prompt_mismatch, "r") as f:
        mismatch_prompt_text = f.read()

    # Resolve mapping CSV
    if args.mapping_csv:
        csv_path = args.mapping_csv
    elif args.category:
        csv_path = os.path.join(args.data_dir, f"{args.category}_mapping.csv")
    else:
        raise ValueError("Either --category or --mapping_csv must be provided.")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")

    eval_data = csv_to_eval_data(csv_path)

    if args.limit > 0:
        items_to_eval = eval_data[: args.limit]
        logging.info("Loaded %d items from %s. Limiting to %d.", len(eval_data), csv_path, args.limit)
    else:
        items_to_eval = eval_data
        logging.info("Loaded %d items from %s. Evaluating ALL.", len(eval_data), csv_path)

    category_name = args.category or "Not Specified"
    logging.info("Multi-step pipeline: first_step=%s, category=%s", args.first_step, category_name)

    client = genai.Client(project=args.project, location=args.location, vertexai=True)

    results = []
    total_items = len(items_to_eval)
    step1_only_count = 0
    step2_count = 0

    def worker(item, i):
        logging.info("--- Item %d/%d | ID: %s ---", i + 1, total_items, item.get("id"))
        return process_item_multi_step(
            item=item,
            data_dir=args.data_dir,
            match_prompt=match_prompt_text,
            mismatch_prompt=mismatch_prompt_text,
            first_step=args.first_step,
            client=client,
            project=args.project,
            eval_cloud_function_name=args.eval_cloud_function_name,
            eval_cloud_function_location=args.eval_cloud_function_location,
            category_name=category_name,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_budget=args.thinking_budget,
            thinking_level=args.thinking_level,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, item, i) for i, item in enumerate(items_to_eval)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            ms = res.get("multi_step", {})
            if ms.get("steps_used") == 1:
                step1_only_count += 1
            else:
                step2_count += 1

    # Summary stats
    valid_evals = sum(1 for r in results if "score" in r)
    total_score = sum(r.get("score", 0) for r in results if "score" in r)
    avg_score = total_score / valid_evals if valid_evals else 0

    logging.info("[SUMMARY] Evaluated %d items.", valid_evals)
    logging.info("[SUMMARY] Average Score: %.2f", avg_score)
    logging.info("[SUMMARY] Resolved at Step 1: %d | Needed Step 2: %d", step1_only_count, step2_count)

    config = {
        "category": args.category,
        "data_dir": args.data_dir,
        "mapping_csv": csv_path,
        "system_prompt_match": args.system_prompt_match,
        "system_prompt_mismatch": args.system_prompt_mismatch,
        "first_step": args.first_step,
        "limit": args.limit,
        "model": args.model,
        "eval_cloud_function_name": args.eval_cloud_function_name,
        "project": args.project,
        "location": args.location,
        "eval_cloud_function_location": args.eval_cloud_function_location,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "thinking_budget": args.thinking_budget,
        "thinking_level": args.thinking_level,
        "output_file_path": args.output_file_path,
        "pipeline_type": "multi_step",
        "step1_resolved_count": step1_only_count,
        "step2_resolved_count": step2_count,
    }

    summarize_multi_step_results(results, config, args.output_file_path)


if __name__ == "__main__":
    main()