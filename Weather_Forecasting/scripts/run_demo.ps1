# Execute from the windows_hpc directory using VS Code PowerShell:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
# CPU-only: append -CpuOnly
param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Missing $py. Create the virtual environment and install requirements.txt." }
Write-Host '[1/6] Checking setup and REAL CSV paths'
& $py scripts\check_setup.py
Write-Host '[2/6] Compiling C/OpenMP/CUDA'
& .\scripts\build.cmd
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
Write-Host '[3/6] Preparing original real CSV files'
& $py scripts\prepare_data.py --dataset delhi
if ($LASTEXITCODE -ne 0) { throw 'Missing CSV file or column mapping error. See README.md.' }
Write-Host '[4/6] Running correctness tests'
& $py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
Write-Host '[5/6] Benchmarking real CSV rows'
$cudaMode = if ($CpuOnly) { 'off' } else { 'on' }
& $py scripts\benchmark.py --datasets delhi --rows 1000 10000 50000 --passes 20 --threads 1 2 4 8 --repeats 3 --cuda $cudaMode
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
Write-Host '[6/6] Launching Streamlit at http://localhost:8501'
& $py -m streamlit run dashboard.py
