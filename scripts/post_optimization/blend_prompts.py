"""
Standalone prompt blending script.

Takes multiple GEPA grid output JSON files, extracts the final prompt
from each, and blends them into a single unified prompt using a strong model.

Usage:
    ./venv/bin/python scripts/blend_prompts.py \
        --input_files grid_runs/match/sandal_29/run1.json grid_runs/match/sandal_29/run2.json \
        --output_prompt prompts/system_prompt_sandal_blended_v1.txt \
        --project default-project-alpha-1
"""

import argparse
import json
import logging
import os
import sys

from google import genai
from google.genai import types as genai_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default blend template path
_DEFAULT_BLEND_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "blend_prompts_template.txt"
)


def load_prompts_from_grid_files(file_paths: list[str]) -> list[dict]:
    """Extract final prompts and metadata from grid output JSON files.
    
    Each file may contain multiple results (grid cells). Extracts all.
    """
    prompts = []
    for fpath in file_paths:
        logger.info("Loading: %s", fpath)
        with open(fpath, "r") as f:
            data = json.load(f)

        for r in data.get("results", []):
            if "error" in r:
                continue
            gepa = r.get("gepa", {})
            prompt_text = gepa.get("final_prompt")
            if not prompt_text:
                logger.warning("No final_prompt in result from %s, skipping.", fpath)
                continue

            # Collect eval metrics for context
            val_metrics = r.get("eval_validation", {}).get("aggregated_metrics", {}).get("metrics_match", {})
            test_metrics = r.get("eval_test", {}).get("aggregated_metrics", {}).get("metrics_match", {})
            train_metrics = r.get("eval_train", {}).get("aggregated_metrics", {}).get("metrics_match", {})

            prompts.append({
                "prompt": prompt_text,
                "source_file": os.path.basename(fpath),
                "gepa_final_score": gepa.get("final_eval_score"),
                "train_precision": train_metrics.get("precision"),
                "train_recall": train_metrics.get("recall"),
                "val_precision": val_metrics.get("precision"),
                "val_recall": val_metrics.get("recall"),
                "test_precision": test_metrics.get("precision"),
                "test_recall": test_metrics.get("recall"),
            })

    return prompts


def blend(
    prompts_data: list[dict],
    project: str,
    blend_template_path: str = None,
) -> str:
    """Blend prompts using gemini-3.1-pro-preview."""

    # Load blend template
    template_path = blend_template_path or _DEFAULT_BLEND_TEMPLATE
    with open(template_path, "r") as f:
        blend_template = f.read()

    # Build prompts section
    prompts_section = ""
    for i, pd in enumerate(prompts_data):
        header = f"#### Prompt {i+1} (source: {pd['source_file']}"
        if pd.get("val_precision") is not None:
            header += f", val_P={pd['val_precision']:.3f}"
        if pd.get("val_recall") is not None:
            header += f", val_R={pd['val_recall']:.3f}"
        header += ")"
        prompts_section += f"{header}\n```\n{pd['prompt']}\n```\n\n"

    # Build scores section
    scores = [{
        "prompt": i + 1,
        "source": pd["source_file"],
        "gepa_final_score": pd.get("gepa_final_score"),
        "train_P": pd.get("train_precision"),
        "train_R": pd.get("train_recall"),
        "val_P": pd.get("val_precision"),
        "val_R": pd.get("val_recall"),
        "test_P": pd.get("test_precision"),
        "test_R": pd.get("test_recall"),
    } for i, pd in enumerate(prompts_data)]

    blend_instruction = blend_template.format(
        num_prompts=len(prompts_data),
        step_scores=json.dumps(scores, indent=2),
        prompts_section=prompts_section,
    )

    logger.info("Calling gemini-3.1-pro-preview (global, thinking=HIGH) to blend %d prompts ...",
                len(prompts_data))

    client = genai.Client(project=project, location="global", vertexai=True)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config=genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                thinking_level="HIGH",
            ),
        ),
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=blend_instruction)],
        )],
    )

    blended = response.text.strip()
    # Strip markdown code fences if wrapped
    if blended.startswith("```"):
        blended = blended.split("\n", 1)[1] if "\n" in blended else blended[3:]
    if blended.endswith("```"):
        blended = blended[:-3].rstrip()

    logger.info("Blended prompt length: %d chars", len(blended))
    return blended


def main():
    parser = argparse.ArgumentParser(
        description="Blend prompts from multiple GEPA grid output files into a single unified prompt."
    )
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Paths to GEPA grid output JSON files.")
    parser.add_argument("--output_prompt", required=True,
                        help="Path to write the blended prompt text file.")
    parser.add_argument("--project", required=True,
                        help="GCP project ID for Vertex AI.")
    parser.add_argument("--blend_template", default=None,
                        help="Path to custom blend prompt template (default: prompts/blend_prompts_template.txt).")

    args = parser.parse_args()

    # Extract prompts
    prompts_data = load_prompts_from_grid_files(args.input_files)
    if not prompts_data:
        logger.error("No prompts extracted from input files.")
        return

    logger.info("Extracted %d prompts from %d files.", len(prompts_data), len(args.input_files))
    for i, pd in enumerate(prompts_data):
        logger.info("  Prompt %d: %s | val_P=%s val_R=%s",
                     i + 1, pd["source_file"],
                     f"{pd['val_precision']:.3f}" if pd.get("val_precision") is not None else "n/a",
                     f"{pd['val_recall']:.3f}" if pd.get("val_recall") is not None else "n/a")

    # Blend
    blended = blend(
        prompts_data=prompts_data,
        project=args.project,
        blend_template_path=args.blend_template,
    )

    # Write output
    os.makedirs(os.path.dirname(args.output_prompt), exist_ok=True)
    with open(args.output_prompt, "w") as f:
        f.write(blended)

    logger.info("Blended prompt written to: %s", args.output_prompt)
    print(f"\nBlended prompt saved to: {args.output_prompt}")
    print(f"Blended from {len(prompts_data)} prompts.")


if __name__ == "__main__":
    main()