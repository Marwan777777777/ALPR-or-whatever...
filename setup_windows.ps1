# ==============================================================================
# Egyptian Vehicle Intelligence System: Windows 11 Automated Installation Script
# ==============================================================================
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#
# Optional Parameters:
#   .\setup_windows.ps1 -Mode GPU      # Force NVIDIA GPU (CUDA 12.4) PyTorch installation
#   .\setup_windows.ps1 -Mode CPU      # Force CPU-only fallback PyTorch installation
#   .\setup_windows.ps1 -SkipTests     # Skip post-installation smoke tests
# ==============================================================================

[CmdletBinding()]
param(
    [ValidateSet("Auto", "GPU", "CPU")]
    [string]$Mode = "Auto",

    [string]$PythonExe = "python",
    [string]$VenvPath = ".venv",
    [switch]$SkipTests,
    [switch]$SkipVerification
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "EGYPTIAN VEHICLE INTELLIGENCE: REPRODUCIBLE SETUP (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Verify Python Availability & Version
Write-Host "`n[1/7] Auditing host Python installation..." -ForegroundColor Yellow
try {
    $pythonVersionOutput = & $PythonExe --version 2>&1
    Write-Host "  Found: $pythonVersionOutput" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python executable '$PythonExe' not found in PATH." -ForegroundColor Red
    Write-Host "  Please install Python 3.11 (64-bit) from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check version numbers
$pyVerMatch = [regex]::Match($pythonVersionOutput, "Python (\d+)\.(\d+)\.(\d+)")
if ($pyVerMatch.Success) {
    $major = [int]$pyVerMatch.Groups[1].Value
    $minor = [int]$pyVerMatch.Groups[2].Value
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "  [ERROR] Python 3.10 or higher is required. Found Python $major.$minor" -ForegroundColor Red
        exit 1
    }
}

# 2. Detect Hardware Acceleration Mode
Write-Host "`n[2/7] Detecting hardware compute acceleration..." -ForegroundColor Yellow
$SelectedMode = $Mode

if ($Mode -eq "Auto") {
    $hasNvidiaGpu = $false
    try {
        $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $smiOut = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0 -and $smiOut) {
                $hasNvidiaGpu = $true
                Write-Host "  Detected NVIDIA GPU: $smiOut" -ForegroundColor Green
            }
        }
    } catch {
        $hasNvidiaGpu = $false
    }

    if ($hasNvidiaGpu) {
        $SelectedMode = "GPU"
        Write-Host "  Auto-selected Mode: GPU (CUDA 12.4 acceleration)" -ForegroundColor Cyan
    } else {
        $SelectedMode = "CPU"
        Write-Host "  Auto-selected Mode: CPU (No NVIDIA GPU detected; using CPU fallback)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Explicit user mode specified: $SelectedMode" -ForegroundColor Cyan
}

# 3. Create or Inspect Virtual Environment
Write-Host "`n[3/7] Setting up isolated Python virtual environment at '$VenvPath'..." -ForegroundColor Yellow
if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    Write-Host "  Creating virtual environment with: $PythonExe -m venv $VenvPath" -ForegroundColor Gray
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "  Existing virtual environment detected at '$VenvPath'." -ForegroundColor Green
}

$VenvPython = Join-Path (Get-Location) "$VenvPath\Scripts\python.exe"
$VenvPip = Join-Path (Get-Location) "$VenvPath\Scripts\pip.exe"

# 4. Upgrade Packaging Toolchain
Write-Host "`n[4/7] Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
& $VenvPip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARNING] Toolchain upgrade returned non-zero exit code. Continuing..." -ForegroundColor Yellow
}

# 5. Install PyTorch according to Hardware Mode
Write-Host "`n[5/7] Installing PyTorch runtime ($SelectedMode profile)..." -ForegroundColor Yellow
if ($SelectedMode -eq "GPU") {
    Write-Host "  Installing CUDA 12.4 wheels from download.pytorch.org..." -ForegroundColor Gray
    & $VenvPip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "  Installing CPU wheels from download.pytorch.org..." -ForegroundColor Gray
    & $VenvPip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] PyTorch installation failed." -ForegroundColor Red
    exit 1
}

# 6. Install Core Dependencies & Editable Package
Write-Host "`n[6/7] Installing core dependencies and package..." -ForegroundColor Yellow
& $VenvPip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Dependency installation from requirements.txt failed." -ForegroundColor Red
    exit 1
}

Write-Host "  Installing vehicle_intelligence in editable mode (-e .)..." -ForegroundColor Gray
& $VenvPip install -e . --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Editable package installation failed." -ForegroundColor Red
    exit 1
}

# Initialize environment configuration and directories
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  Created local '.env' from '.env.example'." -ForegroundColor Green
}

New-Item -ItemType Directory -Force -Path "weights" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

# 7. Run Verification and Smoke Tests
Write-Host "`n[7/7] Executing portability verification & smoke tests..." -ForegroundColor Yellow
if (-not $SkipVerification) {
    & $VenvPython scripts/verify_portability.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARNING] Portability verification returned warnings. Review checklist above." -ForegroundColor Yellow
    }
}

if (-not $SkipTests) {
    Write-Host "`nRunning test suite smoke tests..." -ForegroundColor Gray
    & $VenvPython -m pytest tests/test_config.py tests/test_domain.py tests/test_golden_plates.py -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Smoke tests PASSED cleanly." -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Some smoke tests failed. Verify checkpoint presence in artifacts/." -ForegroundColor Yellow
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "To activate your new environment in PowerShell, run:" -ForegroundColor White
Write-Host "    .\$VenvPath\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "`nTo run end-to-end license plate inference on a sample image:" -ForegroundColor White
Write-Host "    python scripts/alpr_cli.py --input tests/golden/images/0010.jpg" -ForegroundColor Yellow
Write-Host "`nTo verify checkpoints and GPU acceleration:" -ForegroundColor White
Write-Host "    python scripts/verify_portability.py" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
