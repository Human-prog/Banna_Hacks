param(
    [string]$ProjectPath = "",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
if (-not $ProjectPath) { $ProjectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$ModelId = "stable-diffusion-v1-5/stable-diffusion-v1-5"
$VenvPath = "C:\visionforge-sd-venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$ModelCache = "C:\visionforge-sd-model-cache"

Write-Host "`n=== VisionForge local image backend setup ===" -ForegroundColor Cyan
Write-Host "Image generation will run locally via Stable Diffusion 1.5 + DirectML."
Write-Host "Phi-4 Mini remains the local text-reasoning model through Ollama."
Write-Host "No Hugging Face Inference API calls or paid inference endpoints are configured."
Write-Host "The model weights are public downloads; first setup uses several GB of internet and disk space."

if (-not (Test-Path (Join-Path $ProjectPath "app\local_sd_server.py"))) {
    throw "Local worker source not found. First run the VisionForge local backend installer."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher `py` not found. Install Python 3.11, reopen PowerShell, and rerun this script."
}

$Drive = Get-PSDrive -Name C
if ($Drive.Free -lt 12GB) {
    throw "Drive C: has less than 12 GB free. Free space before downloading the model and dependencies."
}

# The existing VisionForge app uses Python 3.13. torch-directml currently
# publishes Windows wheels for Python 3.10–3.12; isolate the worker on 3.11.
$Python311 = $null
try {
    $Probe = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $Probe) { $Python311 = ($Probe | Select-Object -Last 1).Trim() }
} catch { }

if (-not $Python311 -or -not (Test-Path $Python311)) {
    Write-Host "Python 3.11 was not found. Installing it for the current user via WinGet..." -ForegroundColor Yellow
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python 3.11 is required for the separate DirectML worker, and WinGet is unavailable. Install Python 3.11, then run this script again."
    }
    & winget install --id Python.Python.3.11 --exact --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 installation failed. Install Python 3.11, reopen PowerShell, and rerun this script."
    }
    $CandidatePaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe")
    )
    $Python311 = $CandidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Python311) {
        Write-Host "Python was installed. Close and reopen PowerShell, then rerun this setup script." -ForegroundColor Yellow
        return
    }
}

Write-Host "Using Python: $Python311" -ForegroundColor Green
if (-not (Test-Path $PythonPath)) {
    & $Python311 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the dedicated worker virtual environment." }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $PythonPath -m pip install -r (Join-Path $ProjectPath "scripts\requirements-local-sd.txt")
if ($LASTEXITCODE -ne 0) { throw "Local image backend dependencies failed to install." }

New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null
$env:HF_HOME = $ModelCache
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:VISIONFORGE_SD_PYTHON = $PythonPath
$env:VISIONFORGE_HF_HOME = $ModelCache
[Environment]::SetEnvironmentVariable("VISIONFORGE_SD_PYTHON", $PythonPath, "User")
[Environment]::SetEnvironmentVariable("VISIONFORGE_HF_HOME", $ModelCache, "User")

Write-Host "`n=== Checking DirectML ===" -ForegroundColor Cyan
$CheckCode = @'
import torch
import torch_directml
print("PyTorch:", torch.__version__)
device = torch_directml.device()
print("DirectML device:", device)
a = torch.tensor([1.0]).to(device)
b = torch.tensor([2.0]).to(device)
print("DirectML tensor check:", (a + b).item())
'@
$CheckCode | & $PythonPath -
if ($LASTEXITCODE -ne 0) {
    throw "DirectML did not pass the hardware check. Review GPU drivers and the error above; the Gradio app's environment remains separate."
}

if (-not $SkipModelDownload) {
    Write-Host "`n=== Downloading public Stable Diffusion 1.5 model weights ===" -ForegroundColor Cyan
    Write-Host "This is a model-weight download only, not a paid hosted inference request. It may take a while."
    $DownloadCode = @'
from huggingface_hub import snapshot_download
repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# Download only the Diffusers components required for the standard pipeline.
# The repository also contains many .bin/.ckpt/fp16 variants totalling tens of GB;
# avoid fetching those duplicates.
allow = [
    "model_index.json",
    "scheduler/*.json",
    "feature_extractor/*.json",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
path = snapshot_download(repo_id=repo, allow_patterns=allow,
                         ignore_patterns=["*.bin", "*.ckpt", "*.fp16.*", "*.fp16.safetensors"])
print("Required model components cached at:", path)
'@
    $DownloadCode | & $PythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw "Model weight download failed. Verify internet access and the model repository page, then rerun setup. No paid API was called."
    }
}

Write-Host "`nSetup finished." -ForegroundColor Green
Write-Host "Worker interpreter: $PythonPath"
Write-Host "Model cache: $ModelCache"
Write-Host "Next: return to the main project venv and run: python .\app\main.py"
Write-Host "The worker launches automatically on the first image-generation request."
