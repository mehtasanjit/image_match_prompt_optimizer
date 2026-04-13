"""
Predict function for MLflow GEPA prompt optimization.

GEPA calls ``predict_fn(inputs)`` for every training row.
This module loads the current prompt version from the MLflow
prompt registry, builds a vision request with reference + candidate
images, and returns the full model JSON output so the GEPA
reflection model can analyse reasoning patterns across failures.
"""

import io
import json
import base64
import logging
import mimetypes

from PIL import Image
from google import genai
from google.genai import types as genai_types

import mlflow

from mlflow_gepa.config import GEPAConfig

logger = logging.getLogger(__name__)

# ── Module-level state (initialised by ``init``) ────────────
_client: genai.Client | None = None
_config: GEPAConfig | None = None


# ── Helpers ─────────────────────────────────────────────────
def _convert_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_image_as_bytes(path: str) -> bytes:
    """Load an image from *path* and return raw bytes."""
    try:
        with Image.open(path) as img:
            return base64.b64decode(_convert_to_base64(img))
    except Exception as e:
        logger.error("Failed to load image %s: %s", path, e)
        return b""


def get_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"


# ── Initialisation ──────────────────────────────────────────
def init(config: GEPAConfig) -> None:
    """
    Initialise the module-level GenAI client.

    Must be called once before any ``predict_fn`` invocations.
    """
    global _client, _config
    _config = config

    # Disable MLflow's auto-tracing for google.genai SDK calls.
    # Without this, MLflow captures the full inputs to generate_content()
    # — including raw image bytes — into trace spans.  GEPA's
    # make_reflective_dataset() then serialises those spans into the
    # reflection prompt sent to the critic model, causing
    # "text fields too large" errors.
    mlflow.gemini.autolog(disable=True)
    logger.info("Disabled mlflow.gemini autolog to prevent image bytes in traces")

    _client = genai.Client(
        project=config.project,
        location=config.location,
        vertexai=True,
    )
    logger.info(
        "predict module initialised (model=%s, project=%s)",
        config.target_model,
        config.project,
    )


# ── Predict function (passed to optimize_prompts) ──────────
def predict_fn(reference_image_path: str, image_path: str) -> str:
    """
    Predict function called by ``mlflow.genai.optimize_prompts``
    for each training row.

    Returns:
        The **full raw JSON string** from the model (including
        ``reason`` and ``product_match``).  Returning the complete
        output lets the GEPA reflection model inspect reasoning
        when diagnosing failures.

        Falls back to a minimal JSON with ``"Inconclusive"`` on
        any error.
    """
    if _client is None or _config is None:
        raise RuntimeError(
            "predict module not initialised. Call predict.init(config) first."
        )

    # 1. Load current prompt version from MLflow registry
    prompt_version = mlflow.load_prompt(_config.prompt_name)
    system_prompt = prompt_version.format()

    # 2. Load images
    ref_bytes = load_image_as_bytes(reference_image_path)
    ref_mime = get_mime_type(reference_image_path)
    img_bytes = load_image_as_bytes(image_path)
    img_mime = get_mime_type(image_path)

    if not ref_bytes or not img_bytes:
        logger.warning("Empty image bytes for reference=%s candidate=%s", reference_image_path, image_path)
        return json.dumps({"product_match": "Inconclusive", "reason": "Image load failure"})

    # 3. Build multimodal content
    category_display = _config.category.capitalize() if _config.category else "Not Specified"
    parts = [
        genai_types.Part.from_text(text="**Product Category:**\n"),
        genai_types.Part.from_text(text=category_display),
        genai_types.Part.from_text(text="\n**Reference Image:**\n"),
        genai_types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
        genai_types.Part.from_text(text="\n**Candidate Image:**\n"),
        genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
    ]
    user_content = genai_types.Content(role="user", parts=parts)

    gen_config = genai_types.GenerateContentConfig(
        temperature=_config.temperature,
        top_p=_config.top_p,
        system_instruction=system_prompt,
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=_config.thinking_budget,
        ),
    )

    # 4. Call the target model
    try:
        response = _client.models.generate_content(
            model=_config.target_model,
            config=gen_config,
            contents=[user_content],
        )
        output = response.text
        if not output:
            logger.warning("Model returned empty output")
            return json.dumps({"product_match": "Inconclusive", "reason": "Empty model output"})

        # Strip markdown code fences if present, return raw JSON string
        cleaned = output.strip().removeprefix("```json").removesuffix("```").strip()

        # Validate it's parseable JSON before returning
        json.loads(cleaned)
        return cleaned

    except json.JSONDecodeError:
        # Model returned non-JSON — wrap it so downstream stays consistent
        logger.warning("Model output was not valid JSON, wrapping raw text")
        return json.dumps({"product_match": "Inconclusive", "reason": output.strip()})

    except Exception as e:
        logger.error("predict_fn failed: %s", e)
        return json.dumps({"product_match": "Inconclusive", "reason": str(e)})