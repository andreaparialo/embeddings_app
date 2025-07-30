@echo off
:: FAISS GPU Product Search System - Setup Launcher
:: This batch file launches the PowerShell setup script

echo ============================================================
echo    FAISS GPU Product Search System - Windows Setup
echo ============================================================
echo.
echo This will launch the interactive setup wizard.
echo Please ensure you have:
echo   - Miniconda/Anaconda installed
echo   - An NVIDIA GPU (optional but recommended)
echo   - At least 100GB free disk space
echo   - Internet connection for downloads
echo.
pause

:: Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell is not available on this system.
    echo Please ensure PowerShell is installed and in your PATH.
    pause
    exit /b 1
)

:: Launch PowerShell script with appropriate execution policy
echo.
echo Launching setup wizard...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"

:: Check exit code
if %errorlevel% neq 0 (
    echo.
    echo Setup failed with error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

pause