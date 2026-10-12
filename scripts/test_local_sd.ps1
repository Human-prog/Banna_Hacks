$ErrorActionPreference = "Stop"
$ProjectPath = "C:\Users\Arega\Desktop\Projects\Hackthon\Banna Hack"
$PythonPath = "C:\visionforge-sd-venv\Scripts\python.exe"
Set-Location $ProjectPath

Write-Host "=== Project syntax and offline tests ===" -ForegroundColor Cyan
python -m py_compile app\main.py app\local_backend.py app\local_sd_server.py
if ($LASTEXITCODE -ne 0) { throw "Main application syntax check failed." }
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "App unit tests failed." }

if (-not (Test-Path $PythonPath)) { throw "Local SD worker venv is missing. Run scripts\setup_local_sd.ps1." }
Write-Host "`n=== Worker environment ===" -ForegroundColor Cyan
& $PythonPath -c "import torch, torch_directml, diffusers; print('torch:', torch.__version__); print('diffusers:', diffusers.__version__); print('device:', torch_directml.device())"
if ($LASTEXITCODE -ne 0) { throw "Worker environment verification failed." }
Write-Host "All static and environment checks passed. Start the app and test one image." -ForegroundColor Green
