# FAISS GPU Product Search System - Windows Setup Script
# PowerShell script for automated setup with interactive model downloads

# Ensure script runs with proper execution policy
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges for some operations." -ForegroundColor Yellow
    Write-Host "Some features may not work without admin rights." -ForegroundColor Yellow
    Write-Host ""
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Configuration
$CONDA_ENV_NAME = "faiss_env"
$PYTHON_VERSION = "3.10"
$PROJECT_ROOT = $PSScriptRoot
$MIN_DISK_SPACE_GB = 100

# Model download configurations (update these URLs)
$MODEL_CONFIGS = @{
    "GME-Qwen2-VL-7B" = @{
        "url" = "https://huggingface.co/YOUR_MODEL_URL_HERE"  # Update this
        "path" = "gme-Qwen2-VL-7B-Instruct"
        "size" = "15GB"
        "required" = $true
    }
    "LoRA-Checkpoint" = @{
        "url" = "https://your-storage.com/lora-checkpoint.zip"  # Update this
        "path" = "loras\v11-20250620-105815\checkpoint-1095"
        "size" = "500MB"
        "required" = $true
    }
    "FAISS-Index" = @{
        "url" = "https://your-storage.com/faiss-indexes.zip"  # Update this
        "path" = "faiss_indexes"
        "size" = "5GB"
        "required" = $true
    }
    "Sample-Images" = @{
        "url" = "https://your-storage.com/sample-images.zip"  # Update this
        "path" = "pictures"
        "size" = "1GB"
        "required" = $false
    }
}

# Color functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Banner
function Show-Banner {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "   FAISS GPU Product Search System - Windows Setup" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

# Check prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow
    $issues = @()
    
    # Check disk space
    $drive = (Get-Item $PROJECT_ROOT).PSDrive.Name
    $disk = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='${drive}:'"
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    
    if ($freeSpaceGB -lt $MIN_DISK_SPACE_GB) {
        $issues += "Insufficient disk space. Required: ${MIN_DISK_SPACE_GB}GB, Available: ${freeSpaceGB}GB"
    } else {
        Write-Host "✓ Disk space: ${freeSpaceGB}GB available" -ForegroundColor Green
    }
    
    # Check for conda
    try {
        $condaVersion = & conda --version 2>$null
        Write-Host "✓ Conda found: $condaVersion" -ForegroundColor Green
    } catch {
        $issues += "Conda/Miniconda not found. Please install from: https://docs.conda.io/en/latest/miniconda.html"
    }
    
    # Check for NVIDIA GPU
    try {
        $gpuInfo = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        Write-Host "✓ NVIDIA GPU found: $gpuInfo" -ForegroundColor Green
    } catch {
        Write-Host "⚠ NVIDIA GPU not detected or nvidia-smi not found" -ForegroundColor Yellow
        Write-Host "  The system will work but may run slowly without GPU acceleration" -ForegroundColor Yellow
    }
    
    # Check CUDA
    $cudaPath = $env:CUDA_PATH
    if ($cudaPath) {
        Write-Host "✓ CUDA found at: $cudaPath" -ForegroundColor Green
    } else {
        Write-Host "⚠ CUDA_PATH not set. CUDA might not be properly installed" -ForegroundColor Yellow
    }
    
    # Check Visual Studio Build Tools
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        Write-Host "✓ Visual Studio Build Tools found" -ForegroundColor Green
    } else {
        Write-Host "⚠ Visual Studio Build Tools not found. Some packages may fail to install" -ForegroundColor Yellow
        Write-Host "  Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Yellow
    }
    
    if ($issues.Count -gt 0) {
        Write-Host "`nIssues found:" -ForegroundColor Red
        $issues | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
        
        $continue = Read-Host "`nDo you want to continue anyway? (y/N)"
        if ($continue -ne 'y') {
            exit 1
        }
    }
    
    Write-Host ""
}

# Create conda environment
function Initialize-CondaEnvironment {
    Write-Host "Setting up Conda environment..." -ForegroundColor Yellow
    
    # Check if environment already exists
    $envList = & conda env list 2>$null
    if ($envList -match $CONDA_ENV_NAME) {
        Write-Host "Environment '$CONDA_ENV_NAME' already exists." -ForegroundColor Yellow
        $recreate = Read-Host "Do you want to recreate it? (y/N)"
        if ($recreate -eq 'y') {
            Write-Host "Removing existing environment..." -ForegroundColor Yellow
            & conda env remove -n $CONDA_ENV_NAME -y
        } else {
            return
        }
    }
    
    # Create new environment
    Write-Host "Creating new conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..." -ForegroundColor Yellow
    & conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create conda environment"
    }
    
    Write-Host "✓ Conda environment created successfully" -ForegroundColor Green
}

# Install dependencies
function Install-Dependencies {
    Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
    
    # Activate environment and install packages
    $activateScript = (& conda info --base) + "\Scripts\activate.ps1"
    
    # Create a temporary batch file to run conda commands
    $tempBatch = Join-Path $env:TEMP "install_deps.bat"
    
    $batchContent = @"
@echo off
call conda activate $CONDA_ENV_NAME
echo Installing FAISS GPU (this is critical - using conda, not pip)...
call conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
if errorlevel 1 exit /b 1
echo Installing PyTorch with CUDA support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if errorlevel 1 exit /b 1
echo Installing Python requirements...
cd /d "$PROJECT_ROOT"
pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Downgrading transformers for GME compatibility...
pip install transformers==4.51.3
if errorlevel 1 exit /b 1
"@
    
    Set-Content -Path $tempBatch -Value $batchContent
    
    # Run the batch file
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $tempBatch -Wait -PassThru -NoNewWindow
    
    # Clean up
    Remove-Item $tempBatch -Force
    
    if ($process.ExitCode -ne 0) {
        throw "Failed to install dependencies"
    }
    
    Write-Host "✓ All dependencies installed successfully" -ForegroundColor Green
}

# Download with progress
function Download-WithProgress {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Description
    )
    
    try {
        $uri = New-Object System.Uri($Url)
        $request = [System.Net.HttpWebRequest]::Create($uri)
        $request.Method = "HEAD"
        $response = $request.GetResponse()
        $totalBytes = $response.ContentLength
        $response.Close()
        
        $webClient = New-Object System.Net.WebClient
        $global:downloadComplete = $false
        
        $webClient.DownloadProgressChanged += {
            param($sender, $e)
            $percent = [math]::Round(($e.BytesReceived / $totalBytes) * 100, 2)
            $downloaded = [math]::Round($e.BytesReceived / 1MB, 2)
            $total = [math]::Round($totalBytes / 1MB, 2)
            Write-Progress -Activity "Downloading $Description" -Status "$percent% Complete" -PercentComplete $percent -CurrentOperation "$downloaded MB / $total MB"
        }
        
        $webClient.DownloadFileCompleted += {
            $global:downloadComplete = $true
        }
        
        # Create directory if it doesn't exist
        $outputDir = Split-Path -Parent $OutputPath
        if (!(Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        # Start download
        $webClient.DownloadFileAsync($uri, $OutputPath)
        
        # Wait for completion
        while (!$global:downloadComplete) {
            Start-Sleep -Milliseconds 100
        }
        
        Write-Progress -Activity "Downloading $Description" -Completed
        return $true
    }
    catch {
        Write-Host "Error downloading $Description : $_" -ForegroundColor Red
        return $false
    }
}

# Download models interactively
function Install-Models {
    Write-Host "`nModel Download Configuration" -ForegroundColor Yellow
    Write-Host "The following models and data files are available for download:" -ForegroundColor Yellow
    Write-Host ""
    
    $downloadChoices = @{}
    $index = 1
    
    foreach ($modelName in $MODEL_CONFIGS.Keys) {
        $config = $MODEL_CONFIGS[$modelName]
        $required = if ($config.required) { "[REQUIRED]" } else { "[OPTIONAL]" }
        $exists = Test-Path (Join-Path $PROJECT_ROOT $config.path)
        $status = if ($exists) { "✓ Already exists" } else { "✗ Not found" }
        
        Write-Host "$index. $modelName $required - Size: $($config.size) - $status" -ForegroundColor $(if ($exists) { "Green" } else { "Yellow" })
        $downloadChoices[$index] = $modelName
        $index++
    }
    
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  A - Download all missing files" -ForegroundColor Cyan
    Write-Host "  R - Download only required missing files" -ForegroundColor Cyan
    Write-Host "  S - Select specific files to download" -ForegroundColor Cyan
    Write-Host "  N - Skip downloads (I'll download manually)" -ForegroundColor Cyan
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (A/R/S/N)"
    
    $toDownload = @()
    
    switch ($choice.ToUpper()) {
        "A" {
            foreach ($modelName in $MODEL_CONFIGS.Keys) {
                $config = $MODEL_CONFIGS[$modelName]
                $exists = Test-Path (Join-Path $PROJECT_ROOT $config.path)
                if (!$exists) {
                    $toDownload += $modelName
                }
            }
        }
        "R" {
            foreach ($modelName in $MODEL_CONFIGS.Keys) {
                $config = $MODEL_CONFIGS[$modelName]
                $exists = Test-Path (Join-Path $PROJECT_ROOT $config.path)
                if (!$exists -and $config.required) {
                    $toDownload += $modelName
                }
            }
        }
        "S" {
            Write-Host "Enter the numbers of files to download (comma-separated):" -ForegroundColor Yellow
            $selections = (Read-Host).Split(',') | ForEach-Object { $_.Trim() }
            foreach ($sel in $selections) {
                if ($downloadChoices.ContainsKey([int]$sel)) {
                    $toDownload += $downloadChoices[[int]$sel]
                }
            }
        }
        "N" {
            Write-Host "Skipping downloads. Please ensure you download the required files manually." -ForegroundColor Yellow
            return
        }
        default {
            Write-Host "Invalid choice. Skipping downloads." -ForegroundColor Red
            return
        }
    }
    
    if ($toDownload.Count -eq 0) {
        Write-Host "No files to download." -ForegroundColor Green
        return
    }
    
    Write-Host "`nPreparing to download $($toDownload.Count) file(s)..." -ForegroundColor Yellow
    
    foreach ($modelName in $toDownload) {
        $config = $MODEL_CONFIGS[$modelName]
        Write-Host "`nDownloading $modelName..." -ForegroundColor Yellow
        
        if ($config.url -eq "https://huggingface.co/YOUR_MODEL_URL_HERE" -or $config.url -like "*your-storage.com*") {
            Write-Host "⚠ Download URL not configured for $modelName" -ForegroundColor Red
            Write-Host "Please update the URL in this script or download manually to:" -ForegroundColor Yellow
            Write-Host "  $(Join-Path $PROJECT_ROOT $config.path)" -ForegroundColor Yellow
            
            $manualPath = Read-Host "Enter the path to the downloaded file (or press Enter to skip)"
            if ($manualPath -and (Test-Path $manualPath)) {
                $destination = Join-Path $PROJECT_ROOT $config.path
                if (Test-Path $manualPath -PathType Container) {
                    Write-Host "Copying directory..." -ForegroundColor Yellow
                    Copy-Item -Path $manualPath -Destination $destination -Recurse -Force
                } else {
                    Write-Host "Extracting archive..." -ForegroundColor Yellow
                    Expand-Archive -Path $manualPath -DestinationPath $destination -Force
                }
                Write-Host "✓ $modelName installed" -ForegroundColor Green
            }
        }
        else {
            # Download the file
            $outputPath = Join-Path $PROJECT_ROOT "$modelName.download"
            $success = Download-WithProgress -Url $config.url -OutputPath $outputPath -Description $modelName
            
            if ($success) {
                # Extract if it's a zip file
                if ($config.url -like "*.zip") {
                    Write-Host "Extracting $modelName..." -ForegroundColor Yellow
                    $destination = Join-Path $PROJECT_ROOT $config.path
                    Expand-Archive -Path $outputPath -DestinationPath $destination -Force
                    Remove-Item $outputPath -Force
                }
                Write-Host "✓ $modelName downloaded and installed" -ForegroundColor Green
            }
        }
    }
}

# Verify installation
function Test-Installation {
    Write-Host "`nVerifying installation..." -ForegroundColor Yellow
    
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

print("\nChecking model files:")
import os
project_root = os.path.dirname(os.path.abspath(__file__))

# Check for required files
required_files = [
    ("Model", "gme-Qwen2-VL-7B-Instruct"),
    ("LoRA", "loras/v11-20250620-105815/checkpoint-1095"),
    ("FAISS Index", "faiss_indexes/v11_complete_merged_20250625_115302.faiss"),
    ("Database", "database_results/DB_ACTIVE.csv")
]

for name, path in required_files:
    full_path = os.path.join(os.path.dirname(project_root), path)
    if os.path.exists(full_path):
        print(f"✓ {name}: Found")
    else:
        print(f"✗ {name}: Not found at {full_path}")
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
}

# Create shortcuts
function New-Shortcuts {
    Write-Host "Creating shortcuts..." -ForegroundColor Yellow
    
    $shortcuts = @{
        "FAISS Search (GPU)" = "start_gpu.bat"
        "FAISS Search (CPU)" = "start_cpu.bat"
        "FAISS Search (OpenCLIP)" = "start_openclip_gpu.bat"
    }
    
    foreach ($name in $shortcuts.Keys) {
        $targetPath = Join-Path $PROJECT_ROOT $shortcuts[$name]
        $shortcutPath = Join-Path $PROJECT_ROOT "$name.lnk"
        
        if (Test-Path $targetPath) {
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut($shortcutPath)
            $Shortcut.TargetPath = $targetPath
            $Shortcut.WorkingDirectory = $PROJECT_ROOT
            $Shortcut.IconLocation = "cmd.exe,0"
            $Shortcut.Save()
            Write-Host "✓ Created shortcut: $name" -ForegroundColor Green
        }
    }
}

# Main installation flow
function Start-Installation {
    Show-Banner
    
    try {
        # Check prerequisites
        Test-Prerequisites
        
        # Ask about conda environment
        Write-Host "Step 1: Conda Environment Setup" -ForegroundColor Cyan
        $setupEnv = Read-Host "Do you want to set up the conda environment? (Y/n)"
        if ($setupEnv -ne 'n') {
            Initialize-CondaEnvironment
            Install-Dependencies
        }
        
        # Model downloads
        Write-Host "`nStep 2: Model and Data Downloads" -ForegroundColor Cyan
        Install-Models
        
        # Verification
        Write-Host "`nStep 3: Installation Verification" -ForegroundColor Cyan
        Test-Installation
        
        # Create shortcuts
        Write-Host "`nStep 4: Create Shortcuts" -ForegroundColor Cyan
        $createShortcuts = Read-Host "Create desktop shortcuts? (Y/n)"
        if ($createShortcuts -ne 'n') {
            New-Shortcuts
        }
        
        # Success message
        Write-Host "`n============================================================" -ForegroundColor Green
        Write-Host "   Installation completed successfully!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Activate the conda environment: conda activate $CONDA_ENV_NAME" -ForegroundColor White
        Write-Host "2. Run the application: .\start_gpu.bat" -ForegroundColor White
        Write-Host "3. Access the web interface at: http://127.0.0.1:8080" -ForegroundColor White
        Write-Host ""
        
    }
    catch {
        Write-Host "`n============================================================" -ForegroundColor Red
        Write-Host "   Installation failed!" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please check the error message above and try again." -ForegroundColor Yellow
        exit 1
    }
}

# Run the installation
Start-Installation

# Keep window open
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")