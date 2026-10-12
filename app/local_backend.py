"""Local-only VisionForge backend: Ollama reasoning + a localhost SD worker.

No hosted inference API is called by this module. The public SD model weights are
fetched once by Diffusers on first use; subsequent inference runs locally.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
SD_WORKER = BASE_DIR / "app" / "local_sd_server.py"
SD_URL = os.getenv("VISIONFORGE_SD_URL", "http://127.0.0.1:7861").rstrip("/")
OLLAMA_URL = os.getenv("VISIONFORGE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("VISIONFORGE_OLLAMA_MODEL", "phi4-mini:latest")
DEFAULT_SD_PYTHON = Path(os.getenv("VISIONFORGE_SD_PYTHON", r"C:\visionforge-sd-venv\Scripts\python.exe"))


def _opener() -> urllib.request.OpenerDirector:
    # This service is local; avoid sending localhost requests through any proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
    try:
        with _opener().open(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        try:
            error_data = json.loads(exc.read().decode("utf-8", errors="replace"))
            detail = error_data.get("error", f"Local service returned HTTP {exc.code}.")
        except Exception:
            detail = f"Local service returned HTTP {exc.code}."
        raise RuntimeError(str(detail)) from None
    result = json.loads(data.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Local service returned an unexpected response.")
    return result


def refine_prompt_with_ollama(prompt: str) -> tuple[str, str]:
    """Try Phi-4 Mini locally; return the original prompt if Ollama is unavailable."""
    prompt = (prompt or "").strip()
    if not prompt:
        return prompt, "Prompt refinement skipped: the prompt was empty."

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        # Unload Phi-4 Mini after this request so it is less likely to compete
        # with Stable Diffusion for the laptop's shared system/GPU memory.
        "keep_alive": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You refine prompts for an industrial-design image generator. "
                    "Preserve every stated component, constraint, and uncertainty. "
                    "Do not invent exact dimensions, internal electronics, certifications, "
                    "or features not supported by the brief. Improve composition, materials, "
                    "and lighting. Return only a concise image-generation prompt."
                ),
            },
            {"role": "user", "content": "Refine this prompt without changing its requirements:\n\n" + prompt},
        ],
        "options": {"temperature": 0.2, "num_predict": 300},
    }
    try:
        response = _json_request(f"{OLLAMA_URL}/api/chat", payload, timeout=90)
        message = response.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:8000], f"Phi-4 Mini refinement used locally ({OLLAMA_MODEL})."
        return prompt, "Phi-4 Mini returned no text; the original prompt was retained."
    except Exception:
        return prompt, "Ollama was unavailable; the structured prompt was used unchanged."


def _worker_python() -> Path:
    return DEFAULT_SD_PYTHON


def _health(timeout: float = 2) -> dict[str, Any] | None:
    try:
        return _json_request(f"{SD_URL}/health", timeout=timeout)
    except Exception:
        return None


def ensure_sd_worker(wait_seconds: int = 90) -> dict[str, Any]:
    """Start the dedicated Python 3.11 DirectML worker when needed."""
    health = _health()
    if health and health.get("ok"):
        return health

    python_exe = _worker_python()
    if not python_exe.is_file():
        raise RuntimeError(
            "The local Stable Diffusion worker environment is not installed. "
            "Run `powershell -ExecutionPolicy Bypass -File .\\scripts\\setup_local_sd.ps1` "
            "from the VisionForge project folder. No hosted API or token is used."
        )
    if not SD_WORKER.is_file():
        raise RuntimeError(f"Local worker file is missing: {SD_WORKER}")

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "local_sd_worker.log"
    env = os.environ.copy()
    env["HF_HOME"] = os.getenv("VISIONFORGE_HF_HOME", r"C:\visionforge-sd-model-cache")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env["HF_HUB_OFFLINE"] = "1"
    env.setdefault("VISIONFORGE_SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    env.setdefault("VISIONFORGE_SD_PORT", "7861")

    # Reuse an already running worker if it became ready between the health
    # check and process start. The worker binds to loopback only.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [str(python_exe), str(SD_WORKER)],
            cwd=str(BASE_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )

    deadline = time.monotonic() + max(5, wait_seconds)
    while time.monotonic() < deadline:
        health = _health(timeout=2)
        if health and health.get("ok"):
            return health
        time.sleep(1)

    tail = ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-12:])
    except OSError:
        pass
    message = "Local Stable Diffusion worker did not start."
    if tail:
        message += " Recent worker log:\n" + tail[-1800:]
    raise RuntimeError(message)


def generate_local_image(
    prompt: str,
    mode: str,
    reference_image: Image.Image | None = None,
    strength: float = 0.62,
    seed: int | None = None,
    steps: int = 16,
) -> tuple[Image.Image, dict[str, Any]]:
    """Request a local SD 1.5 generation from the loopback worker."""
    if mode not in {"txt2img", "img2img"}:
        raise ValueError("Unsupported local generation mode.")
    if mode == "img2img" and reference_image is None:
        raise ValueError("Sketch-guided mode requires an uploaded reference image.")
    health = ensure_sd_worker()

    image_b64 = None
    if reference_image is not None and mode == "img2img":
        buffer = io.BytesIO()
        reference_image.convert("RGB").save(buffer, format="PNG", optimize=True)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    payload = {
        "prompt": (prompt or "").strip()[:8000],
        "mode": mode,
        "image_base64": image_b64,
        "strength": float(min(0.9, max(0.25, strength))),
        "seed": seed,
        "steps": int(min(24, max(8, steps))),
        "width": 512,
        "height": 512,
    }
    if not payload["prompt"]:
        raise ValueError("A prompt is required for local generation.")

    response = _json_request(f"{SD_URL}/generate", payload, timeout=1800)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "The local image worker failed."))
    encoded = response.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("The local worker did not return an image.")

    from base64 import b64decode
    try:
        with Image.open(io.BytesIO(b64decode(encoded))) as image:
            result = image.convert("RGB")
    except Exception as exc:
        raise RuntimeError("The local worker returned invalid image data.") from exc

    details = {
        "model": response.get("model", health.get("model", "Stable Diffusion 1.5")),
        "provider": f"Local Stable Diffusion / {response.get('device', 'local')}",
        "device": response.get("device", health.get("device", "DirectML")),
        "generation_seconds": float(response.get("generation_seconds", 0)),
        "seed": response.get("seed", seed),
        "steps": response.get("steps", payload["steps"]),
        "mode": mode,
        "ollama_endpoint": OLLAMA_URL,
    }
    return result, details


def local_backend_status() -> str:
    """Return a safe status summary without revealing any credential values."""
    python_exe = _worker_python()
    sd_health = _health()
    if sd_health and sd_health.get("ok"):
        sd_line = (
            f"- **Image worker:** Ready · `{sd_health.get('model', 'Stable Diffusion 1.5')}`\n"
            f"- **Worker device:** `{sd_health.get('device', 'DirectML')}` · weights load on first generation"
        )
    elif python_exe.is_file():
        sd_line = f"- **Image worker:** Not running yet; it starts automatically when generation is requested.\n- **Worker interpreter:** `{python_exe}`"
    else:
        sd_line = "- **Image worker:** Not installed. Run `scripts/setup_local_sd.ps1`."

    ollama_line = "- **Local reasoning:** Ollama status unknown."
    try:
        tags = _json_request(f"{OLLAMA_URL}/api/tags", timeout=3).get("models", [])
        names = [str(item.get("name", "")) for item in tags if isinstance(item, dict)]
        found = next((n for n in names if n.split(":")[0] == OLLAMA_MODEL.split(":")[0]), None)
        ollama_line = (
            f"- **Local reasoning:** `{found}` installed · Phi-4 Mini refinement runs on this computer."
            if found else
            f"- **Local reasoning:** `{OLLAMA_MODEL}` was not found. Image generation still works; prompt refinement will fall back to the structured prompt."
        )
    except Exception:
        ollama_line = "- **Local reasoning:** Ollama is not reachable at `127.0.0.1:11434`; structured prompts will be used unchanged."

    return (
        "### Local-only runtime\n"
        f"{ollama_line}\n{sd_line}\n\n"
        "- **Hosted inference:** Disabled in the generation path.\n"
        "- **API token:** Not required.\n"
        "- **Network:** The public Stable Diffusion model weights download once to the local cache; image inference runs locally.\n\n"
        "The sketch-guided path uses img2img initialization, not ControlNet. It may change geometry and does not validate engineering correctness."
    )
