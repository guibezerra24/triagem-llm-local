$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force work\tmp | Out-Null
$env:TEMP = (Resolve-Path work\tmp).Path
$env:TMP = $env:TEMP
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar ambiente Python.' }
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar PyTorch CUDA.' }
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha nas dependencias.' }
& .\.venv\Scripts\python.exe -m src.gpu_check
if ($LASTEXITCODE -ne 0) { throw 'CUDA indisponivel ou teste GPU falhou.' }
Write-Host 'Ambiente pronto. Execute .\.venv\Scripts\python.exe -m src.run_experiments'
