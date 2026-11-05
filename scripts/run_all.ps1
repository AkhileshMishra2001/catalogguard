$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Create .venv and install requirements first. See README.' }
function Run-Python { & $py @args; if ($LASTEXITCODE -ne 0) { throw "Python step failed: $args" } }
Run-Python scripts/setup_env.py
docker compose up -d --wait mongo
if ($LASTEXITCODE -ne 0) { throw 'MongoDB startup failed. Start Docker Desktop and inspect docker compose logs mongo.' }
Run-Python -m pytest -q
foreach ($step in @('download','prepare','analyze','sync','embed','train','evaluate','retrieval','latency','sync','abo-scale','export-evidence')) {
    Write-Host "`n=== CatalogGuard: $step ===" -ForegroundColor Cyan
    Run-Python -m catalogguard.cli $step
}
Write-Host 'Pipeline complete. Start the UI or Docker serving stack using README.' -ForegroundColor Green
