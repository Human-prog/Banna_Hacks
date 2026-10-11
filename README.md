# VisionForge AI

**Sketch → Process → Explore → Compare** — an AI-assisted product-concept studio for rough hardware and electronics sketches.

VisionForge helps turn an early sketch and a structured written brief into several visual concept directions. It prepares and measures the reference image, sends the selected reference mode and prompt to a hosted image model, and saves a review package for each generated concept. It is intended for early visual exploration, not engineering verification.

## Features

- **Guided multi-tab workflow:** Sketch, Design brief, Configure & generate, Results & export, Project library, and System & limits.
- **Local sketch processing:** EXIF orientation correction, minimum/maximum dimension checks, 1024 px maximum edge, cleanup profiles, and edge-map preview.
- **Image diagnostics:** input/output dimensions, source format, file size, grayscale contrast, and approximate edge density.
- **Categorized design goals:** form factor, usability, lifecycle/reliability, and practical engineering priorities.
- **Separated engineering constraints:** must-have features, approximate dimensions/interfaces, and things to avoid.
- **Prompt preview:** review the common prompt and distinct variant directions before calling the hosted model.
- **Sketch-guided or prompt-only generation:** sketch-guided sends the prepared sketch to an image-to-image model; prompt-only uses the text brief only.
- **Generate 1–3 design variants:** variants share the brief but explore a faithful refinement, alternative form, and material/interaction study. Each variant makes a separate hosted inference request.
- **Side-by-side comparison sheet:** automatically generated contact sheet for reviewing the concepts together.
- **Consolidated ZIP export:** includes generated concept images, per-variant briefs and metadata, sketch diagnostics when available, comparison sheet, and a manifest.
- **Local project library:** previously saved runs can be refreshed and browsed from the app.
- **Offline tests:** sketch processing, variant utilities, comparison-sheet creation, ZIP exports, and the UI output contract are tested without making a model request.

## Run on Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python app/main.py
```

If you already have a working `.env`, **do not replace it**. Add or update settings manually and preserve your private `HF_TOKEN`.

The app normally opens at `http://127.0.0.1:7860`. Keep it bound to localhost unless you intentionally understand and configure the network exposure. Do not commit `.env` or place your token in source code.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token with Inference Providers permission |
| `HF_PROVIDER` | `fal-ai` | Hosted inference provider |
| `HF_EDIT_MODEL` | `black-forest-labs/FLUX.2-klein-9B` | Sketch-guided image-to-image model |
| `HF_IMAGE_MODEL` | `black-forest-labs/FLUX.1-dev` | Prompt-only text-to-image model |
| `PORT` | `7860` | Local app port |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address; keep local by default |

Provider access, model support, rate limits, quotas, and potential charges depend on the current account and provider settings. Choose one variant when conserving requests. Generating three variants usually requires three separate inference calls.

## Run tests without making AI requests

```powershell
python -m unittest discover -s tests -v
```

All tests in this suite are intended to run locally without the hosted model. They cannot guarantee that an external model or provider will be available at runtime.

## How variants are generated

1. VisionForge validates and prepares the input sketch if one is provided.
2. It builds a shared prompt from the project name, purpose, priorities, visualization style, and constraints.
3. For each requested variant, it appends one exploration direction: **Reference-faithful refinement**, **Alternative form exploration**, or **Material and interaction study**.
4. In sketch-guided mode the processed sketch is included in each image-to-image request. Prompt-only mode does not condition generation on the sketch.
5. Successful variants are saved as their own run directories under `outputs/runs/`.
6. A side-by-side sheet and combined ZIP are saved under `outputs/exports/`.

The directions are prompt guidance, not hard guarantees. A model can omit or invent features, change proportions, or fail to follow constraints. Compare each render manually with the sketch and written brief.

## Engineering limitations

VisionForge creates **visual concepts only**. Generated images are not validated CAD, circuit schematics, verified dimensions, thermal or electrical analysis, a bill of materials, a safety assessment, or manufacturing instructions. The edge map is a visual aid, not an extracted engineering drawing. Use proper CAD/EDA workflows and independent checks before making technical decisions.
