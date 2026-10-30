<#
.SYNOPSIS
  One-shot reproducible deployment for the chords-analysis skill (Windows).

.DESCRIPTION
  Creates a self-contained Python 3.9 environment next to this script and installs
  every dependency, without conda and without touching the system Python.

  Layout created under -Root (default: the repo root that contains this script):
    .venv39\              Python 3.9 + all packages
    .cache\pip            pip wheel cache (kept off C:)
    .cache\torch          demucs / harmonix checkpoints
    .cache\huggingface    HF downloads
    .cache\models         BTC + manifest-declared weights

  Nothing large is written to the system drive. Safe to delete -Root\.cache at any
  time; weights re-download on demand.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy.ps1
  powershell -ExecutionPolicy Bypass -File deploy.ps1 -Mirror none   # official HF
#>
param(
  [string]$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [string]$Mirror = "https://hf-mirror.com",
  [string]$IndexUrl = "",
  [switch]$SkipWeights
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
$SkillDir = Join-Path $Root "skills\chords-analysis"
$Venv = Join-Path $Root ".venv39"
$Py = Join-Path $Venv "Scripts\python.exe"
$Cache = Join-Path $Root ".cache"

Write-Host "== chords-analysis deploy ==" -ForegroundColor Cyan
Write-Host "root   : $Root"
Write-Host "python : $Py"
Write-Host "cache  : $Cache"

# --- 0. cache redirection (keeps C: free) ---
New-Item -ItemType Directory -Force $Cache | Out-Null
$env:PIP_CACHE_DIR = Join-Path $Cache "pip"
$env:TORCH_HOME = Join-Path $Cache "torch"
$env:HF_HOME = Join-Path $Cache "huggingface"
$env:CHORDS_MODEL_DIR = Join-Path $Cache "models"
$env:CHORDS_HF_MIRROR = $Mirror
if ($Mirror -ne "none") { $env:HF_ENDPOINT = $Mirror }

# --- 1. uv ---
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
  Write-Host "[1/6] installing uv ..."
  $base = Get-Command python -ErrorAction SilentlyContinue
  if (-not $base) { throw "No python found to bootstrap uv. Install Python 3.8+ or uv manually." }
  & python -m pip install --quiet --user uv
  $uv = Get-Command uv -ErrorAction SilentlyContinue
  if (-not $uv) { throw "uv install failed. Try: pip install uv" }
} else {
  Write-Host "[1/6] uv present: $($uv.Source)"
}

# --- 2. Python 3.9 ---
Write-Host "[2/6] ensuring Python 3.9 ..."
& uv python install 3.9
if ($LASTEXITCODE -ne 0) { throw "uv python install 3.9 failed" }

# --- 3. venv ---
if (-not (Test-Path $Py)) {
  Write-Host "[3/6] creating venv at $Venv ..."
  & uv venv --python 3.9 --seed $Venv
  if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
} else {
  Write-Host "[3/6] venv exists"
}

# --- 4. packages (uv pip: parallel downloads, much faster than plain pip) ---
Write-Host "[4/6] installing packages ..."
$uvIndex = @()
if ($IndexUrl -ne "") { $uvIndex = @("--index-url", $IndexUrl) }
# madmom 0.16.1 needs numpy<1.24, setuptools<81 and a no-isolation build
& uv pip install --python $Py @uvIndex "setuptools<81" wheel cython "numpy<1.24"
& uv pip install --python $Py @uvIndex --no-build-isolation "madmom==0.16.1"
if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] madmom failed; ensemble runs BTC-only" -ForegroundColor Yellow }
& uv pip install --python $Py @uvIndex -r (Join-Path $SkillDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] some optional packages failed; layers degrade" -ForegroundColor Yellow }

# --- 5. weights ---
if (-not $SkipWeights) {
  Write-Host "[5/6] fetching model weights (mirror: $Mirror) ..."
  & $Py -W ignore (Join-Path $SkillDir "scripts\models.py") --fetch-all
} else {
  Write-Host "[5/6] weights skipped (-SkipWeights)"
}

# --- 6. smoke test ---
Write-Host "[6/6] smoke test ..."
& $Py -W ignore -c "import madmom, librosa, transformers, mir_eval; print('core deps OK')"
& $Py -W ignore -c "import basic_pitch; print('basic-pitch OK')" 2>$null
& $Py -W ignore -c "import allin1_infer; print('structure model OK')" 2>$null

Write-Host ""
Write-Host "Done. Use this interpreter:" -ForegroundColor Green
Write-Host "  $Py"
Write-Host ""
Write-Host "One-shot analysis:"
Write-Host "  `$env:CHORDS_MODEL_DIR='$env:CHORDS_MODEL_DIR'; `$env:HF_ENDPOINT='$env:HF_ENDPOINT'"
Write-Host "  & '$Py' -W ignore '$SkillDir\scripts\run_all.py' audio\<file> --outdir out"
Write-Host ""
Write-Host "To make the cache/env settings permanent for new shells, run once:"
Write-Host "  [Environment]::SetEnvironmentVariable('TORCH_HOME','$env:TORCH_HOME','User')"
Write-Host "  [Environment]::SetEnvironmentVariable('HF_HOME','$env:HF_HOME','User')"
Write-Host "  [Environment]::SetEnvironmentVariable('CHORDS_MODEL_DIR','$env:CHORDS_MODEL_DIR','User')"
