"""Local Stable Diffusion 1.5 worker for Windows + torch-directml.

Only binds to 127.0.0.1. The public model is downloaded to a local cache on
first inference; generation requests themselves never go to a hosted API.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import time
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

os.environ.setdefault("HF_HOME", r"C:\visionforge-sd-model-cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL_ID = os.getenv("VISIONFORGE_SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
HOST = "127.0.0.1"
PORT = int(os.getenv("VISIONFORGE_SD_PORT", "7861"))

_PIPELINE: Any = None
_PIPELINE_MODE: str | None = None
_DEVICE_NAME = "DirectML"
_INFERENCE_LOCK = threading.Lock()


def _load_pipeline(mode: str):
    global _PIPELINE, _PIPELINE_MODE, _DEVICE_NAME
    if _PIPELINE is not None and _PIPELINE_MODE == mode:
        return _PIPELINE

    # Only one pipeline mode is held at a time to reduce memory pressure on
    # integrated graphics with shared system RAM.
    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None
        _PIPELINE_MODE = None
        import gc
        gc.collect()

    import torch
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
    )

    requested_device = os.getenv("VISIONFORGE_SD_DEVICE", "cpu").strip().lower()
    if requested_device == "cpu":
        device = torch.device("cpu")
        _DEVICE_NAME = "CPU"
        # Limit thread count to leave resources for the desktop and Gradio.
        torch.set_num_threads(min(6, os.cpu_count() or 4))
    else:
        import torch_directml
        device = torch_directml.device()
        _DEVICE_NAME = f"DirectML ({device})"
    pipeline_class = StableDiffusionImg2ImgPipeline if mode == "img2img" else StableDiffusionPipeline
    print(f"Loading {MODEL_ID} in {mode} mode on {device} (float32)...", flush=True)

    pipeline = pipeline_class.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    try:
        pipeline.enable_attention_slicing("auto")
    except Exception:
        pass
    try:
        pipeline.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass

    _PIPELINE = pipeline
    _PIPELINE_MODE = mode
    print(f"Pipeline ready: {mode} on {device}", flush=True)
    return _PIPELINE


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageOps

    prompt = str(payload.get("prompt", "")).strip()
    mode = str(payload.get("mode", "txt2img"))
    if not prompt:
        raise ValueError("Prompt is empty.")
    if mode not in {"txt2img", "img2img"}:
        raise ValueError("Unknown generation mode.")

    size = int(os.getenv("VISIONFORGE_SD_SIZE", "384"))
    size = min(512, max(256, size))
    size = max(256, (size // 8) * 8)
    width = size
    height = size
    max_steps = int(os.getenv("VISIONFORGE_SD_MAX_STEPS", "6"))
    steps = min(max_steps, max(4, int(payload.get("steps", 6))))
    strength = min(0.9, max(0.25, float(payload.get("strength", 0.62))))
    seed_value = payload.get("seed")
    seed = int(seed_value) if seed_value is not None else random.randint(0, 2**31 - 1)

    init_image = None
    if mode == "img2img":
        encoded = payload.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Sketch-guided mode needs a reference image.")
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            init_image = ImageOps.pad(
                opened.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                color="white",
            )

    import torch
    pipe = _load_pipeline(mode)
    torch.manual_seed(seed)
    started = time.perf_counter()
    if mode == "img2img":
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=6.5,
        ).images[0]
    else:
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=6.5,
        ).images[0]
    elapsed = round(time.perf_counter() - started, 2)
    buffer = io.BytesIO()
    result.convert("RGB").save(buffer, format="PNG", optimize=True)
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": _DEVICE_NAME,
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "generation_seconds": elapsed,
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "VisionForgeLocalSD/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[VisionForge SD] " + fmt % args, flush=True)

    def _send(self, data: dict[str, Any], code: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._send({"ok": False, "error": "Not found"}, 404)
            return
        self._send({
            "ok": True,
            "model": MODEL_ID,
            "device": _DEVICE_NAME,
            "loaded": _PIPELINE is not None,
            "mode": _PIPELINE_MODE,
            "inference": "local-only",
        })

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/generate":
            self._send({"ok": False, "error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 30 * 1024 * 1024:
                raise ValueError("Request size is invalid or exceeds 30 MB.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            with _INFERENCE_LOCK:
                response = _generate(payload)
            self._send(response)
        except Exception as exc:
            traceback.print_exc()
            self._send({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}. Check outputs/local_sd_worker.log. No hosted inference was attempted.",
            }, 500)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"VisionForge local image worker at http://{HOST}:{PORT}", flush=True)
    print(f"Model: {MODEL_ID}; weights cached under {os.environ['HF_HOME']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
