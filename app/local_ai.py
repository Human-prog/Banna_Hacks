"""Optional local-only Ollama integration for VisionForge.

Uses Python's standard library rather than an external API client. All model calls
are made to the local Ollama server, never a hosted inference provider.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

DEFAULT_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
VISION_MODEL_PREFIXES = ("qwen2.5vl", "qwen2.5-vl")
REASONING_MODEL_PREFIXES = ("phi4-mini", "phi-4-mini", "phi4")


class LocalAIError(RuntimeError):
    """A local AI service or model is missing, unavailable, or returned an error."""


def list_local_models(base_url: str | None = None, timeout: float = 2.0) -> list[str]:
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LocalAIError(
            "Ollama is not responding at 127.0.0.1:11434. Open Ollama or start its local service. "
            "No token or cloud connection is needed."
        ) from exc
    models = payload.get("models", [])
    return sorted({str(item.get("name") or item.get("model") or "") for item in models if item.get("name") or item.get("model")})


def _find_model(models: list[str], prefixes: tuple[str, ...]) -> str | None:
    for model in models:
        lowered = model.casefold()
        if any(lowered == prefix or lowered.startswith(prefix + ":") for prefix in prefixes):
            return model
    return None


def local_runtime_status(base_url: str | None = None) -> str:
    try:
        models = list_local_models(base_url=base_url, timeout=1.5)
    except LocalAIError as exc:
        return (
            "## Local AI status\n"
            "- **Ollama:** unavailable\n"
            "- **Offline engineering checker:** ready; it does not need Ollama\n\n"
            f"{exc}\n\n"
            "After installing Ollama, optional model commands are `ollama pull qwen2.5vl:3b` and `ollama pull phi4-mini`."
        )
    vision = _find_model(models, VISION_MODEL_PREFIXES)
    reasoning = _find_model(models, REASONING_MODEL_PREFIXES)
    installed = ", ".join(f"`{name}`" for name in models) if models else "No models installed"
    return (
        "## Local AI status\n"
        "- **Ollama API:** available at `127.0.0.1:11434`\n"
        f"- **Vision model (Qwen2.5-VL):** {'ready — `' + vision + '`' if vision else 'not installed'}\n"
        f"- **Reasoning model (Phi-4 Mini):** {'ready — `' + reasoning + '`' if reasoning else 'not installed'}\n\n"
        f"**Installed models:** {installed}\n\n"
        "Calls stay on the local Ollama service. This integration does not require a Hugging Face token or make hosted inference requests. Models need to be downloaded separately."
    )


def _post_chat(
    model: str,
    messages: list[dict[str, Any]],
    base_url: str | None = None,
    timeout: float = 180.0,
) -> str:
    endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0.2},
    }
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(1024).decode("utf-8", errors="replace")
        except OSError:
            detail = ""
        if "not found" in detail.casefold() or exc.code == 404:
            raise LocalAIError(f"Ollama model `{model}` is unavailable. Confirm it is installed with `ollama list`.") from exc
        raise LocalAIError(f"Ollama returned HTTP {exc.code}. Check the local service and model status.") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LocalAIError("The local Ollama request failed. Confirm Ollama is running and the selected model is installed.") from exc
    content = ((result.get("message") or {}).get("content") or "").strip()
    if not content:
        raise LocalAIError("Ollama returned an empty response. Try again or choose a smaller local model.")
    return content


def _select_model(prefixes: tuple[str, ...], requested: str | None = None) -> str:
    models = list_local_models()
    if requested:
        requested_lower = requested.casefold()
        match = next((name for name in models if name.casefold() == requested_lower), None)
        if match:
            return match
    match = _find_model(models, prefixes)
    if match:
        return match
    if prefixes == VISION_MODEL_PREFIXES:
        raise LocalAIError(
            "Qwen2.5-VL is not installed. In PowerShell, run `ollama pull qwen2.5vl:3b`, "
            "then refresh Local AI status. This download is optional and does not use a cloud token."
        )
    raise LocalAIError(
        "Phi-4 Mini is not installed. In PowerShell, run `ollama pull phi4-mini`, "
        "then refresh Local AI status. You can still use the offline engineering checker without it."
    )


def _image_as_base64(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise LocalAIError("The image file is unavailable. Upload or select it again.")
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=86, optimize=True)
    except (OSError, ValueError) as exc:
        raise LocalAIError("The uploaded file could not be prepared as an image.") from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def inspect_sketch_with_local_vision(image_path: str | Path, prompt_context: str = "") -> str:
    model = _select_model(VISION_MODEL_PREFIXES)
    encoded_image = _image_as_base64(image_path)
    request_text = (
        "Inspect this product or engineering sketch. Return these headings: "
        "1. Visible observations, 2. Candidate components and confidence, "
        "3. Approximate spatial relationships, 4. Unclear or occluded details, "
        "5. Questions a designer should resolve. Distinguish direct visual evidence from guesses. "
        "Do not infer exact dimensions, electrical correctness, safety, or manufacturability. "
        "Be conservative; say 'unclear' instead of inventing parts."
    )
    if prompt_context.strip():
        request_text += "\nUser brief for context (not proof of what is visible):\n" + prompt_context.strip()[:3000]
    return _post_chat(model, [{"role": "user", "content": request_text, "images": [encoded_image]}])


def critique_concept_with_local_vision(image_path: str | Path, requirements: str) -> str:
    model = _select_model(VISION_MODEL_PREFIXES)
    encoded_image = _image_as_base64(image_path)
    prompt = (
        "Review this generated product-concept image against the user's requirements. "
        "Use these headings: Observed evidence; Requirements that seem represented; "
        "Potential mismatches or missing features; Uncertain points; Three revision priorities. "
        "Only comment on visible evidence. Do not claim dimension verification, functional correctness, "
        "electrical safety, or true image-to-sketch similarity. If a requirement cannot be observed, mark it unknown.\n\n"
        f"Requirements:\n{requirements.strip()[:4000] or 'No explicit requirements provided.'}"
    )
    return _post_chat(model, [{"role": "user", "content": prompt, "images": [encoded_image]}])


def refine_prompt_with_local_reasoner(
    product_description: str,
    base_prompt: str,
    design_constraints: str = "",
    visual_inventory: str = "",
) -> str:
    model = _select_model(REASONING_MODEL_PREFIXES)
    system = (
        "You are the prompt-planning stage of a hardware concept design studio. Improve an image-generation prompt "
        "without introducing unsupported product requirements. Preserve explicitly requested parts and constraints. "
        "Use cautious language for dimensions and feasibility. Do not claim engineering validation. Return only the revised prompt."
    )
    user = (
        f"Product description:\n{product_description.strip()[:3000]}\n\n"
        f"Constraints and requirements:\n{design_constraints.strip()[:3000] or 'None provided.'}\n\n"
        f"Candidate visible-component notes (may contain uncertainty):\n{visual_inventory.strip()[:3000] or 'No visual inventory provided.'}\n\n"
        f"Current prompt draft:\n{base_prompt.strip()[:5000]}\n\n"
        "Make the composition and visible controls unambiguous. Don't add details that conflict with the requirements."
    )
    return _post_chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def draft_revision_prompt_with_local_reasoner(
    product_description: str,
    current_prompt: str,
    visual_critique: str,
    requirements: str = "",
) -> str:
    model = _select_model(REASONING_MODEL_PREFIXES)
    system = (
        "You write the next revision prompt for an iterative product-image concept workflow. "
        "Use the visual critique as uncertain model feedback, not verified ground truth. "
        "Preserve requirements not identified as problems; correct only prioritized issues. "
        "Return only the revised image-generation prompt and do not claim that a new image has been generated."
    )
    user = (
        f"Product purpose:\n{product_description.strip()[:2500]}\n\n"
        f"Required features and constraints:\n{requirements.strip()[:3500] or 'None provided.'}\n\n"
        f"Current prompt:\n{current_prompt.strip()[:5000]}\n\n"
        f"Local vision-model critique:\n{visual_critique.strip()[:5000]}\n\n"
        "Produce a targeted revision prompt that addresses the highest-priority visible mismatches."
    )
    return _post_chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}])
