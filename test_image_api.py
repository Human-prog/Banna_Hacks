import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN", "").strip()
model = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")

if not token:
    raise SystemExit("Add your HF_TOKEN to .env first.")

client = InferenceClient(
    provider="fal-ai",
    api_key=token,
    timeout=180,
)

prompt = (
    "Professional industrial design concept render of a compact "
    "pocket-sized portable music player, USB-C charging port, "
    "physical playback buttons, replaceable microSD card slot, "
    "realistic materials, three-quarter view, neutral studio "
    "background, no text, no watermark."
)

print(f"Model: {model}")
print("Generating image...")

try:
    image = client.text_to_image(prompt, model=model)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (
        "visionforge_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    )
    image.save(output_path)
    print(f"SUCCESS: {output_path.resolve()}")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
