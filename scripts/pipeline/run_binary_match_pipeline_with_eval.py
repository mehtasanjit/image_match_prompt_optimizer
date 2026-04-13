"""Run product image match pipeline with evaluation.

Compares reference images against candidate images using a vision model,
evaluates predictions against ground truth via a Cloud Function scorer,
and produces a summary JSON with per-item results and aggregate metrics.

Args:
    --category: (Optional) Product category name. Used in the model prompt as "Product Category"
                and for auto-resolving the mapping CSV (<data_dir>/<category>_mapping.csv).
                If not provided, defaults to "Not Specified" in the model prompt and
                --mapping_csv must be specified explicitly.

Usage:
    .venv/bin/python scripts/pipeline/run_binary_match_pipeline_with_eval.py \
        --data_dir data/sampled/smartwatch_1/train \
        --system_prompt prompts/binary_match_or_not.txt \
        --eval_cloud_function_name img_match_weighted_moderate \
        --project my-gcp-project \
        --location us-central1 \
        --eval_cloud_function_location us-central1 \
        --model gemini-2.5-flash \
        --output_file_path benchmark_runs/smartwatch_match_1.json \
        --category smartwatch
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

# Add current directory to path to import the summarizer module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from summarize_binary_match_pipeline_eval import summarize_eval_results
except ImportError:
    pass

import pandas as pd

from google import genai
from google.genai import types as genai_types


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


def _extract_product_match(model_text: str) -> str:
    """Extract product_match value from model output JSON."""
    if not model_text:
        return ""
    try:
        cleaned = model_text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("product_match", "").strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _call_model(
    item: Dict[str, Any],
    data_dir: str,
    system_prompt: str,
    client: genai.Client,
    category_name: str,
    model_name: str,
    temperature: float,
    top_p: float,
    thinking_budget: int,
    thinking_level: str = None,
    few_shot_examples: list = None,
) -> tuple:
    """Call the model with images and return (model_text, telemetry, latency)."""
    ref_filenames = item.get("reference_image_filenames_list", [])
    img_filenames = item.get("image_filenames_list", [])

    ref_path = os.path.join(data_dir, ref_filenames[0])
    img_path = os.path.join(data_dir, img_filenames[0])

    ref_bytes = load_image_as_bytes(ref_path)
    img_bytes = load_image_as_bytes(img_path)
    ref_mime = get_mime_type(ref_path)
    img_mime = get_mime_type(img_path)

    parts = []

    if few_shot_examples:
        parts.append(genai_types.Part.from_text(text="**Few Shot Examples**\n\n"))
        for idx, example in enumerate(few_shot_examples):
            parts.append(genai_types.Part.from_text(text=f"Example {idx + 1}:\n**Reference Image:**\n"))
            ex_ref_path = os.path.join(data_dir, example.get("reference_image_filename"))
            try:
                parts.append(genai_types.Part.from_bytes(data=load_image_as_bytes(ex_ref_path), mime_type=get_mime_type(ex_ref_path)))
            except Exception as e:
                logging.warning(f"Failed to load few-shot reference image: {e}")
            parts.append(genai_types.Part.from_text(text="\n**Candidate Image:**\n"))
            ex_img_path = os.path.join(data_dir, example.get("image_filename"))
            try:
                parts.append(genai_types.Part.from_bytes(data=load_image_as_bytes(ex_img_path), mime_type=get_mime_type(ex_img_path)))
            except Exception as e:
                logging.warning(f"Failed to load few-shot candidate image: {e}")
            expected = example.get("expected_output", "Inconclusive")
            parts.append(genai_types.Part.from_text(text=f"\nExpected Output: {expected}\n---\n"))
        parts.append(genai_types.Part.from_text(text="**Input Reference Image and Candidate Image**\n\n"))

    # Always include category in prompt (defaults to "Not Specified" if not provided)
    display_category = category_name.capitalize() if category_name else "Not Specified"
    parts.extend([
        genai_types.Part.from_text(text="**Product Category:**\n"),
        genai_types.Part.from_text(text=display_category),
    ])

    parts.extend([
        genai_types.Part.from_text(text="**Reference Image:**\n"),
        genai_types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
        genai_types.Part.from_text(text="\n**Candidate Image:**\n"),
        genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
    ])

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
        config_args["thinking_config"] = genai_types.ThinkingConfig(include_thoughts=False, thinking_level=t_level)
    else:
        config_args["thinking_config"] = genai_types.ThinkingConfig(include_thoughts=False, thinking_budget=thinking_budget)

    config = genai_types.GenerateContentConfig(**config_args)

    start_time = time.time()
    telemetry = {}
    model_text = ""
    try:
        response = client.models.generate_content(model=model_name, config=config, contents=[user_content])
        model_text = response.text
        usage = response.usage_metadata
        prompt_text_tokens = 0
        prompt_image_tokens = 0
        if usage and usage.prompt_tokens_details:
            for detail in usage.prompt_tokens_details:
                modality_str = str(detail.modality)
                if modality_str in ["MediaModality.TEXT", "TEXT"]:
                    prompt_text_tokens += detail.token_count
                elif modality_str in ["MediaModality.IMAGE", "IMAGE"]:
                    prompt_image_tokens += detail.token_count
        telemetry = {
            "prompt_token_count": (usage.prompt_token_count if usage else 0) or 0,
            "prompt_token_count_text": prompt_text_tokens,
            "prompt_token_count_image": prompt_image_tokens,
            "candidates_token_count": (usage.candidates_token_count if usage else 0) or 0,
            "thoughts_token_count": (getattr(usage, 'thoughts_token_count', 0) if usage else 0) or 0,
            "total_token_count": (usage.total_token_count if usage else 0) or 0,
        }
    except Exception as e:
        model_text = str(e)
        logging.error(f"Model call error: {e}")

    latency = time.time() - start_time
    return model_text, telemetry, latency


def convert_image_to_base64_string(image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def load_image_as_bytes(image_source: str) -> bytes:
    """Loads an image from the given source path and returns its raw bytes."""
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
    """Guess the mime type from the file extension."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "image/jpeg"


def process_item(
    item: Dict[str, Any],
    data_dir: str,
    system_prompt: str,
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
    few_shot_examples: list = None
) -> Dict[str, Any]:

    # 1. Load Images
    ref_filenames = item.get("reference_image_filenames_list", [])
    img_filenames = item.get("image_filenames_list", [])

    if not ref_filenames or not img_filenames:
        return {"error": "Missing reference or candidate image in data."}

    ref_path = os.path.join(data_dir, ref_filenames[0])
    logging.debug(f"Loading Reference Image - {ref_path}")

    img_path = os.path.join(data_dir, img_filenames[0])
    logging.debug(f"Loading Candidate Image - {img_path}")

    try:
        ref_bytes = load_image_as_bytes(ref_path)
        img_bytes = load_image_as_bytes(img_path)
    except Exception as e:
        return {"error": f"Failed to load images: {e}"}

    ref_mime = get_mime_type(ref_path)
    img_mime = get_mime_type(img_path)

    # 2. Construct User Request
    parts = []

    # Inject Few-Shot Examples if provided
    if few_shot_examples:
        parts.append(genai_types.Part.from_text(text="**Few Shot Examples**\n\n"))

        for idx, example in enumerate(few_shot_examples):
            parts.append(genai_types.Part.from_text(text=f"Example {idx + 1}:\n**Reference Image:**\n"))

            ex_ref_path = os.path.join(data_dir, example.get("reference_image_filename"))
            try:
                ex_ref_bytes = load_image_as_bytes(ex_ref_path)
                ex_ref_mime = get_mime_type(ex_ref_path)
                parts.append(genai_types.Part.from_bytes(data=ex_ref_bytes, mime_type=ex_ref_mime))
            except Exception as e:
                logging.warning(f"Failed to load few-shot reference image: {e}")

            parts.append(genai_types.Part.from_text(text="\n**Candidate Image:**\n"))

            ex_img_path = os.path.join(data_dir, example.get("image_filename"))
            try:
                ex_img_bytes = load_image_as_bytes(ex_img_path)
                ex_img_mime = get_mime_type(ex_img_path)
                parts.append(genai_types.Part.from_bytes(data=ex_img_bytes, mime_type=ex_img_mime))
            except Exception as e:
                logging.warning(f"Failed to load few-shot candidate image: {e}")

            expected = example.get("expected_output", "Inconclusive")
            parts.append(genai_types.Part.from_text(text=f"\nExpected Output: {expected}\n---\n"))

        parts.append(genai_types.Part.from_text(text="**Input Reference Image and Candidate Image**\n\n"))

    # Always include category in prompt (defaults to "Not Specified" if not provided)
    display_category = category_name.capitalize() if category_name else "Not Specified"
    parts.extend([
        genai_types.Part.from_text(text="**Product Category:**\n"),
        genai_types.Part.from_text(text=display_category)
    ])

    # Append Target Evaluation Images
    parts.extend([
        genai_types.Part.from_text(text="**Reference Image:**\n"),
        genai_types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
        genai_types.Part.from_text(text="\n**Candidate Image:**\n"),
        genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime)
    ])

    user_content = genai_types.Content(role="user", parts=parts)

    # 3. Call Model (Gemini)
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
                logging.warning(f"Invalid thinking_level '{thinking_level}'. Falling back to MINIMAL.")

        logging.info(f"Model is {model_name}. Setting thinking level to {t_level}.")
        config_args["thinking_config"] = genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=t_level
        )
    else:
        config_args["thinking_config"] = genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=thinking_budget
        )

    config = genai_types.GenerateContentConfig(**config_args)

    start_time = time.time()
    telemetry = {}
    try:
        response = client.models.generate_content(
            model=model_name,
            config=config,
            contents=[user_content]
        )
        model_text = response.text
        logging.info(f"\n[AI RAW OUTPUT | ID: {item.get('id')}]\n{model_text}\n" + "-" * 50)

        usage = response.usage_metadata
        prompt_text_tokens = 0
        prompt_image_tokens = 0

        if usage and usage.prompt_tokens_details:
            for detail in usage.prompt_tokens_details:
                modality_str = str(detail.modality)
                if modality_str in ["MediaModality.TEXT", "TEXT"]:
                    prompt_text_tokens += detail.token_count
                elif modality_str in ["MediaModality.IMAGE", "IMAGE"]:
                    prompt_image_tokens += detail.token_count

        telemetry = {
            "prompt_token_count": (usage.prompt_token_count if usage else 0) or 0,
            "prompt_token_count_text": prompt_text_tokens,
            "prompt_token_count_image": prompt_image_tokens,
            "candidates_token_count": (usage.candidates_token_count if usage else 0) or 0,
            "thoughts_token_count": (getattr(usage, 'thoughts_token_count', 0) if usage else 0) or 0,
            "total_token_count": (usage.total_token_count if usage else 0) or 0,
        }

    except Exception as e:
        model_text = str(e)
        logging.error(f"\n[AI ERROR | ID: {item.get('id')}]\n{model_text}\n" + "-" * 50)

    latency = time.time() - start_time

    # Check JSON validity locally
    is_valid_json = False
    if model_text:
        try:
            cleaned_text = model_text.strip().replace("```json", "").replace("```", "").strip()
            json.loads(cleaned_text)
            is_valid_json = True
        except json.JSONDecodeError:
            pass

    # 4. Call Eval Cloud Function
    cf_url = f"https://{eval_cloud_function_location}-{project}.cloudfunctions.net/{eval_cloud_function_name}"

    payload = {
        "response": model_text,
        "target": item.get("ground_truth", "")
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
        "model_output": model_text,
        "is_valid_json": is_valid_json,
        "score": eval_result.get("match_score", 0.0),
        "eval_payload": eval_result,
        "latency_sec": latency,
        "telemetry": telemetry
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run product image match pipeline with evaluation")
    parser.add_argument("--category", required=False, default=None,
                        help="(Optional) Product category name (e.g. smartwatch). Used in the model "
                             "prompt and for auto-resolving the mapping CSV as <data_dir>/<category>_mapping.csv. "
                             "If omitted, --mapping_csv is required and category defaults to 'Not Specified' in the prompt.")
    parser.add_argument("--data_dir", required=True, help="Base data directory where images and mapping JSON reside")
    parser.add_argument("--system_prompt", required=True, help="Path to the system prompt txt file")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to evaluate (0 for all)")
    parser.add_argument("--eval_cloud_function_name", required=True, help="Cloud Function name for evaluation")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--location", required=True, help="GCP Location for GenAI (e.g. us-central1)")
    parser.add_argument("--eval_cloud_function_location", required=True, help="GCP location for cloud function")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature config")
    parser.add_argument("--top_p", type=float, default=0.95, help="Model top P config")
    parser.add_argument("--thinking_budget", type=int, default=0, help="Thinking budget for supported models")
    parser.add_argument("--thinking_level", type=str, default=None, help="Thinking level for Gemini 3 Preview (e.g., MINIMAL)")
    parser.add_argument("--output_file_path", required=False, default=None,
                        help="Path to save the output summary JSON. Directories are created automatically.")
    parser.add_argument("--few_shot_examples_file", required=False, default=None, help="Path to JSON file containing few-shot examples")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--inconclusive_system_prompt", required=False, default=None,
                        help="Path to inconclusive pre-filter prompt. If provided, items classified as "
                             "Inconclusive skip the match step and are recorded as Inconclusive.")
    parser.add_argument("--mapping_csv", required=False, default=None,
                        help="Path to mapping CSV file. If not provided, defaults to <data_dir>/<category>_mapping.csv. "
                             "Required columns: id, ground_truth, reference_image_filename, image_filename.")

    args = parser.parse_args()

    # Read System Prompt
    with open(args.system_prompt, 'r') as f:
        sys_prompt_text = f.read()

    # Read Inconclusive System Prompt (if provided)
    inconclusive_prompt_text = None
    if args.inconclusive_system_prompt:
        with open(args.inconclusive_system_prompt, 'r') as f:
            inconclusive_prompt_text = f.read()
        logging.info(f"Loaded inconclusive pre-filter prompt from {args.inconclusive_system_prompt}")

    # Read Mapping Data (CSV → eval data)
    if args.mapping_csv:
        csv_path = args.mapping_csv
    elif args.category:
        csv_path = os.path.join(args.data_dir, f"{args.category}_mapping.csv")
    else:
        parser.error("Either --category or --mapping_csv must be provided.")

    logging.info(f"Loading mapping CSV: {csv_path}")
    eval_data = csv_to_eval_data(csv_path)

    # Auto-create output directory if needed
    if args.output_file_path:
        output_dir = os.path.dirname(args.output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Read Few-Shot Examples if provided
    few_shot_examples = None
    if args.few_shot_examples_file:
        try:
            with open(args.few_shot_examples_file, 'r') as f:
                few_shot_data = json.load(f)
                few_shot_examples = few_shot_data.get("examples", [])
                logging.info(f"Loaded {len(few_shot_examples)} few-shot examples from {args.few_shot_examples_file}")
        except Exception as e:
            logging.error(f"Failed to load few-shot examples from {args.few_shot_examples_file}: {e}")

    if args.limit > 0:
        items_to_eval = eval_data[:args.limit]
        logging.info(f"Loaded {len(eval_data)} items from {csv_path}. Limiting to {args.limit}.")
    else:
        items_to_eval = eval_data
        logging.info(f"Loaded {len(eval_data)} items from {csv_path}. Evaluating ALL items.")

    # Initialize Google GenAI SDK Client
    client = genai.Client(project=args.project, location=args.location, vertexai=True)

    results = []
    total_score = 0.0
    valid_evals = 0
    valid_json_count = 0
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    inconclusive_count = 0
    total_items = len(items_to_eval)

    def worker(item, i):
        nonlocal inconclusive_count
        logging.info(f"--- Processing Item {i+1} / {total_items} | ID: {item.get('id')} ---")

        # Step 0: Inconclusive pre-filter (if prompt provided)
        if inconclusive_prompt_text:
            logging.info(f"  Running inconclusive pre-filter for {item.get('id')}...")
            try:
                inc_text, inc_telemetry, inc_latency = _call_model(
                    item=item,
                    data_dir=args.data_dir,
                    system_prompt=inconclusive_prompt_text,
                    client=client,
                    category_name=args.category,
                    model_name=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    thinking_budget=args.thinking_budget,
                    thinking_level=args.thinking_level,
                )
                inc_decision = _extract_product_match(inc_text).lower()
                logging.info(f"  Inconclusive pre-filter result: {inc_decision}")

                if inc_decision == "inconclusive":
                    inconclusive_count += 1
                    logging.info(f"  Item {item.get('id')} marked INCONCLUSIVE — skipping match step.")

                    inconclusive_output = json.dumps({
                        "product_in_reference_image": "N/A - Inconclusive",
                        "product_in_candidate_image": "N/A - Inconclusive",
                        "reason": "Images were classified as inconclusive during pre-filtering",
                        "product_match": "Inconclusive"
                    })

                    cf_url = f"https://{args.eval_cloud_function_location}-{args.project}.cloudfunctions.net/{args.eval_cloud_function_name}"
                    try:
                        eval_resp = requests.post(cf_url, json={
                            "response": inconclusive_output,
                            "target": item.get("ground_truth", "")
                        }, timeout=30)
                        eval_resp.raise_for_status()
                        eval_result = eval_resp.json()
                    except Exception as e:
                        eval_result = {"error": str(e)}

                    return {
                        "id": item.get("id"),
                        "ground_truth": item.get("ground_truth"),
                        "model_output": inconclusive_output,
                        "is_valid_json": True,
                        "score": eval_result.get("match_score", 0.0),
                        "eval_payload": eval_result,
                        "latency_sec": inc_latency,
                        "telemetry": inc_telemetry,
                        "inconclusive_prefilter": True,
                        "inconclusive_raw_output": inc_text,
                    }
            except Exception as e:
                logging.warning(f"  Inconclusive pre-filter failed for {item.get('id')}: {e}. Proceeding to match.")

        # Step 1: Run match pipeline as normal
        res = process_item(
            item=item,
            data_dir=args.data_dir,
            system_prompt=sys_prompt_text,
            client=client,
            project=args.project,
            eval_cloud_function_name=args.eval_cloud_function_name,
            eval_cloud_function_location=args.eval_cloud_function_location,
            category_name=args.category,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_budget=args.thinking_budget,
            thinking_level=args.thinking_level,
            few_shot_examples=few_shot_examples
        )
        res["inconclusive_prefilter"] = False

        logging.info(f"Ground Truth: {res.get('ground_truth')}")
        logging.info(f"Eval Score: {res.get('score')} | Valid JSON: {res.get('is_valid_json')} | CF Response: {res.get('eval_payload')}")
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, item, i) for i, item in enumerate(items_to_eval)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()

            if "score" in res:
                total_score += res["score"]
                valid_evals += 1
                if res.get("is_valid_json"):
                    valid_json_count += 1

            if "telemetry" in res and res["telemetry"]:
                total_prompt_tokens += res["telemetry"].get("prompt_token_count", 0)
                total_candidates_tokens += res["telemetry"].get("candidates_token_count", 0)

            results.append(res)

    # Summary
    if valid_evals > 0:
        avg_score = total_score / valid_evals
        json_rate = valid_json_count / valid_evals

        avg_prompt_tokens = total_prompt_tokens / valid_evals
        avg_candidates_tokens = total_candidates_tokens / valid_evals

        logging.info(f"\n[SUMMARY] Evaluated {valid_evals} items.")
        logging.info(f"[SUMMARY] Valid JSON Rate: {json_rate:.2%}")
        logging.info(f"[SUMMARY] Average Match Score: {avg_score:.2f}")
        if inconclusive_prompt_text:
            logging.info(f"[SUMMARY] Inconclusive pre-filtered: {inconclusive_count} items")
            logging.info(f"[SUMMARY] Sent to match: {valid_evals - inconclusive_count} items")
        logging.info(f"[SUMMARY] Average Prompt Tokens: {avg_prompt_tokens:.1f}")
        logging.info(f"[SUMMARY] Average Completion Tokens: {avg_candidates_tokens:.1f}")

    run_config = {
        "category": args.category,
        "data_dir": args.data_dir,
        "system_prompt": args.system_prompt,
        "inconclusive_system_prompt": args.inconclusive_system_prompt or None,
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
        "output_file_path": args.output_file_path
    }

    if 'summarize_eval_results' in globals():
        summary = summarize_eval_results(results, run_config, args.output_file_path)
    else:
        logging.warning("summarize_eval_results function not found. Skipping detailed JSON summarization.")


if __name__ == "__main__":
    main()