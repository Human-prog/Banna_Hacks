# VisionForge AI

**Sketch → Process → Concept** — an AI-assisted product-concept studio for rough hardware and electronics sketches.

VisionForge prepares and measures the uploaded sketch, combines it with a structured design brief, generates a visual concept, and exports a review package. Its goal is to help communicate and iterate on early ideas, not to replace engineering validation.

## Current features

- **Guided multi-tab workflow:** separate Sketch, Design brief, Configure & generate, Results & export, Project library, and System & limits tabs.
- **Categorized design priorities:** form factor, usability, reliability/lifecycle, and engineering goals are grouped instead of presented as one long checkbox list.
- **Prompt preview:** inspect the assembled generation prompt before sending a model request.
- **Live result feedback:** the generation status and a quick image preview stay visible on the configuration tab, with full exports in Results.
- **Sketch preprocessing:** orientation correction, file-size and dimension checks, 1024 px maximum edge, line cleanup, and a high-contrast profile.
- **Image diagnostics:** original/prepared dimensions, file size, grayscale contrast, and estimated edge density.
- **Sketch-guided generation:** uses an image-to-image model with the prepared sketch as the reference image.
- **Prompt-only fallback:** generates from the written brief when you do not have a sketch or the edit model is unavailable.
- **Prompt builder:** includes the product purpose, design priorities, constraints, and render style.
- **Exportable records:** generated PNG, processed reference image, edge map, Markdown brief, JSON metadata, and a ZIP bundle.
- **Local history:** recent concepts are saved in `outputs/runs/` and shown in the gallery.
- **Offline tests for preprocessing:** test the sketch processor without making a paid/remote model request.

## Run locally on Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python app/main.py
```

Add your own Hugging Face token to `.env`; never commit `.env` or paste the token into source code. The token needs permission to use Inference Providers. The default `fal-ai` provider and hosted models may have quotas or charges depending on your account and current provider terms.

The application opens at `http://127.0.0.1:7860` by default. The browser UI stays local. Do not set `GRADIO_SERVER_NAME=0.0.0.0` unless you intentionally need network access and understand the exposure.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token with Inference Providers permission |
| `HF_PROVIDER` | `fal-ai` | Inference provider |
| `HF_EDIT_MODEL` | `black-forest-labs/FLUX.2-klein-9B` | Sketch-guided image-to-image model |
| `HF_IMAGE_MODEL` | `black-forest-labs/FLUX.1-dev` | Prompt-only text-to-image model |
| `PORT` | `7860` | Local application port |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address; keep local by default |

## Test processing without calling the AI provider

```powershell
python -m unittest discover -s tests -v
```

These tests only exercise local image preprocessing. They do not verify hosted model availability, cost, or generation quality.

## How sketch processing works

1. Validate the file and reject empty, unreadable, oversized, or extremely small uploads.
2. Respect EXIF orientation, convert to RGB, and downscale large inputs to a 1024 px longest edge.
3. Apply one of three Pillow-based profiles: `Original`, `Clean sketch`, or `High-contrast linework`.
4. Calculate a simple edge map and image-statistics report.
5. For sketch-guided mode, send the processed sketch plus the prompt to the image-to-image model. Prompt-only fallback does not use the sketch to condition the generation.
6. Save a ZIP bundle that contains the result and enough metadata to review how it was made.

The edge map is for inspection; it is not a CAD trace or a substitute for geometry extraction. The image model may change dimensions, invent controls or ports, or misunderstand the drawing.

## Engineering and safety limitations

AI outputs are **visual concepts only**. They are not validated CAD, an electrical schematic, verified dimensions, a bill of materials, a safety assessment, or manufacturing-ready documentation. Review the actual design independently before making technical decisions.

## Suggested next improvements

- Multi-variant generation with consistent prompt/settings records.
- Side-by-side original / processed / output comparison with synchronized zoom.
- User-adjustable target aspect ratio and image-to-image strength, once supported reliably by the selected provider.
- Optional structured design report with a human verification checklist.
- A deployed demo with a carefully managed secret and request limits.
