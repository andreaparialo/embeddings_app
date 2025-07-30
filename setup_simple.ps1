# FAISS GPU Product Search System - Windows Setup Script (Simplified)
# PowerShell script for automated setup

# Set error action preference
$ErrorActionPreference = "Stop"

# Configuration
$CONDA_ENV_NAME = "faiss_env"
$PYTHON_VERSION = "3.10"
$PROJECT_ROOT = $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   FAISS GPU Product Search System - Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check for conda
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
try {
    $condaVersion = & conda --version 2>$null
    Write-Host "✓ Conda found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Conda/Miniconda not found. Please install from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Red
    exit 1
}

# Check for NVIDIA GPU
try {
    $gpuInfo = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    Write-Host "✓ NVIDIA GPU found: $gpuInfo" -ForegroundColor Green
} catch {
    Write-Host "⚠ NVIDIA GPU not detected - the system will work but may run slowly" -ForegroundColor Yellow
}

Write-Host ""

# Create conda environment
Write-Host "Step 1: Setting up Conda environment..." -ForegroundColor Cyan
$envList = & conda env list 2>$null
if ($envList -match $CONDA_ENV_NAME) {
    Write-Host "Environment '$CONDA_ENV_NAME' already exists." -ForegroundColor Yellow
    $recreate = Read-Host "Do you want to recreate it? (y/N)"
    if ($recreate -eq 'y') {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        & conda env remove -n $CONDA_ENV_NAME -y
    }
}

if (!($envList -match $CONDA_ENV_NAME) -or $recreate -eq 'y') {
    Write-Host "Creating new conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..." -ForegroundColor Yellow
    & conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create conda environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Conda environment created successfully" -ForegroundColor Green
}

Write-Host ""

# Install dependencies
Write-Host "Step 2: Installing dependencies..." -ForegroundColor Cyan
Write-Host "This will install FAISS GPU, PyTorch, and other requirements..." -ForegroundColor Yellow

# Create batch file for installation
$tempBatch = Join-Path $env:TEMP "install_deps.bat"
$batchContent = @"
@echo off
call conda activate $CONDA_ENV_NAME
echo Installing FAISS GPU (this is critical - using conda, not pip)...
call conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
if errorlevel 1 (
    echo Failed to install FAISS GPU
    exit /b 1
)
echo Installing PyTorch with CUDA support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo Failed to install PyTorch
    exit /b 1
)
echo Installing Python requirements...
cd /d "$PROJECT_ROOT"
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements
        exit /b 1
    )
    echo Downgrading transformers for GME compatibility...
    pip install transformers==4.51.3
)
echo Dependencies installed successfully!
"@

Set-Content -Path $tempBatch -Value $batchContent

# Run the batch file
$process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $tempBatch -Wait -PassThru -NoNewWindow

# Clean up
Remove-Item $tempBatch -Force

if ($process.ExitCode -ne 0) {
    Write-Host "Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✓ All dependencies installed successfully" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Step 3: Verifying installation..." -ForegroundColor Cyan

$tempScript = Join-Path $env:TEMP "verify_install.py"
$pythonScript = @'
import sys
print("Python:", sys.version)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import faiss
    print(f"✓ FAISS: {faiss.__version__}")
    gpu_available = hasattr(faiss, 'StandardGpuResources')
    print(f"  GPU support: {gpu_available}")
except ImportError as e:
    print(f"✗ FAISS import failed: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")
'@

Set-Content -Path $tempScript -Value $pythonScript

# Create verification batch file
$tempBatch = Join-Path $env:TEMP "verify.bat"
$batchContent = @"
@echo off
call conda activate $CONDA_ENV_NAME
cd /d "$PROJECT_ROOT"
python "$tempScript"
"@

Set-Content -Path $tempBatch -Value $batchContent

# Run verification
& cmd /c $tempBatch

# Clean up
Remove-Item $tempScript -Force
Remove-Item $tempBatch -Force

Write-Host ""

# Check for model files
Write-Host "Step 4: Checking for required model files..." -ForegroundColor Cyan
$requiredFiles = @(
    @{Name="Model"; Path="gme-Qwen2-VL-7B-Instruct"},
    @{Name="LoRA"; Path="loras\v11-20250620-105815\checkpoint-1095"},
    @{Name="FAISS Index"; Path="faiss_indexes\v11_complete_merged_20250625_115302.faiss"},
    @{Name="Database"; Path="database_results\DB_ACTIVE.csv"}
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $PROJECT_ROOT $file.Path
    if (Test-Path $fullPath) {
        Write-Host "✓ $($file.Name): Found" -ForegroundColor Green
    } else {
        Write-Host "✗ $($file.Name): Not found" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing files detected. Please download these manually:" -ForegroundColor Yellow
    foreach ($file in $missingFiles) {
        Write-Host "  - $($file.Name) to: $(Join-Path $PROJECT_ROOT $file.Path)" -ForegroundColor Yellow
    }
}

Write-Host ""

# Create shortcuts
Write-Host "Step 5: Creating shortcuts..." -ForegroundColor Cyan
$shortcuts = @{
    "FAISS Search (GPU)" = "start_gpu.bat"
    "FAISS Search (CPU)" = "start_cpu.bat"
    "FAISS Search (OpenCLIP)" = "start_openclip_gpu.bat"
}

foreach ($name in $shortcuts.Keys) {
    $targetPath = Join-Path $PROJECT_ROOT $shortcuts[$name]
    if (Test-Path $targetPath) {
        $shortcutPath = Join-Path $PROJECT_ROOT "$name.lnk"
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($shortcutPath)
        $Shortcut.TargetPath = $targetPath
        $Shortcut.WorkingDirectory = $PROJECT_ROOT
        $Shortcut.Save()
        Write-Host "✓ Created shortcut: $name" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   Setup completed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate the conda environment: conda activate $CONDA_ENV_NAME" -ForegroundColor White
Write-Host "2. Run the application: .\start_gpu.bat" -ForegroundColor White
Write-Host "3. Access the web interface at: http://127.0.0.1:8080" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")