# scripts/reproduce_all.ps1
# End-to-end automated reproduction script for Egyptian ALPR pipeline
param (
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$RawDatasetPath = "C:\Users\MARWAN\plate_project\images_raw\EALPR-master",
    [string]$DetectorWeights = "artifacts\runs\detect\ealpr_cleaned_yolov8n\weights\best.pt",
    [string]$CharWeights = "artifacts\runs\detect_chars\ealpr_characters_yolov8n\weights\best.pt"
)

$ErrorActionPreference = "Stop"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Egyptian ALPR End-to-End Reproduction & Verification Pipeline" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Manifest Integrity
Write-Host "`n[Step 1/6] Verifying Split Manifest Byte-Identical Integrity..." -ForegroundColor Yellow
$manifestPath = "artifacts\splits\cleaned_split_manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Cleaned split manifest not found at $manifestPath"
}
$expectedHash = "532ce0b498da092d06957a3edf406b0d194d1276112b2e6a345884bc3248bf78"
$actualHash = (Get-FileHash -Path $manifestPath -Algorithm SHA256).Hash.ToLower()

if ($actualHash -ne $expectedHash) {
    throw "Manifest integrity check FAILED! Expected: $expectedHash, Found: $actualHash"
}
Write-Host "  [OK] Manifest SHA256 verified: $actualHash" -ForegroundColor Green

# 2. Check Weights
Write-Host "`n[Step 2/6] Verifying Checkpoints..." -ForegroundColor Yellow
if (-not (Test-Path $DetectorWeights)) {
    throw "Detector weights not found at $DetectorWeights"
}
if (-not (Test-Path $CharWeights)) {
    throw "Character weights not found at $CharWeights"
}
Write-Host "  [OK] Found Detector Weights: $DetectorWeights" -ForegroundColor Green
Write-Host "  [OK] Found Character Weights: $CharWeights" -ForegroundColor Green

# 3. Run Pytest Suite
Write-Host "`n[Step 3/6] Running Complete Pytest Suite (Unit, Domain, Splitter, Golden)..." -ForegroundColor Yellow
& $PythonExe -m pytest -v
if ($LASTEXITCODE -ne 0) {
    throw "Test suite failed with exit code $LASTEXITCODE"
}
Write-Host "  [OK] All test suites passed." -ForegroundColor Green

# 4. Evaluate Plate Detector
Write-Host "`n[Step 4/6] Evaluating Plate Detector on Cleaned Test Split..." -ForegroundColor Yellow
& $PythonExe scripts/evaluate_plate_detector.py `
    --weights $DetectorWeights `
    --dataset-dir "artifacts\ealpr_yolo_dataset_cleaned" `
    --gt-dir "artifacts\splits\cleaned_ground_truth" `
    --output-dir "artifacts\evaluation_cleaned_verified" `
    --conf 0.25
if ($LASTEXITCODE -ne 0) {
    throw "Plate detector evaluation failed."
}
Write-Host "  [OK] Plate detector evaluation completed." -ForegroundColor Green

# 5. Evaluate End-to-End ALPR and OCR
Write-Host "`n[Step 5/6] Evaluating End-to-End ALPR and Character OCR..." -ForegroundColor Yellow
& $PythonExe scripts/audit_end_to_end_and_ocr.py `
    --detector-weights $DetectorWeights `
    --char-weights $CharWeights `
    --ealpr-root $RawDatasetPath
if ($LASTEXITCODE -ne 0) {
    throw "End-to-End evaluation failed."
}
Write-Host "  [OK] End-to-End ALPR evaluation completed." -ForegroundColor Green

# 6. Run Confusion Matrix
Write-Host "`n[Step 6/6] Computing 27-Class Confusion Matrix..." -ForegroundColor Yellow
& $PythonExe scripts/generate_character_confusion_matrix.py
if ($LASTEXITCODE -ne 0) {
    throw "Confusion matrix generation failed."
}
Write-Host "  [OK] Confusion matrix computed and saved." -ForegroundColor Green

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "  [SUCCESS] All headline metrics reproduced successfully from artifacts!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
