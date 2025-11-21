@echo off
REM CUDA Environment Variables Setup for GTX 1660 Super
REM Run as Administrator

echo ========================================
echo CUDA ENVIRONMENT VARIABLES SETUP
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ Running as Administrator
) else (
    echo ✗ ERROR: Please run as Administrator
    echo Right-click this file and select "Run as administrator"
    pause
    exit /b 1
)

echo.
echo Setting up CUDA environment variables...
echo.

REM Set CUDA_PATH
echo Setting CUDA_PATH...
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" /M
if %errorLevel% == 0 (
    echo ✓ CUDA_PATH set successfully
) else (
    echo ✗ Failed to set CUDA_PATH
)

REM Get current system PATH
echo.
echo Updating system PATH...
for /f "usebackq tokens=2,*" %%A in (`reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH`) do set SYS_PATH=%%B

REM Define CUDA paths
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
set "CUDA_LIBNVVP=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp"

REM Check if CUDA bin is already in PATH
echo %SYS_PATH% | findstr /C:"%CUDA_BIN%" >nul
if errorlevel 1 (
    echo Adding CUDA to system PATH...
    setx PATH "%SYS_PATH%;%CUDA_BIN%;%CUDA_LIBNVVP%" /M
    if %errorLevel% == 0 (
        echo ✓ CUDA added to system PATH
    ) else (
        echo ✗ Failed to update system PATH
    )
) else (
    echo ✓ CUDA already in system PATH
)

echo.
echo ========================================
echo ENVIRONMENT SETUP COMPLETE
echo ========================================
echo.
echo Environment variables configured:
echo   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
echo   PATH += CUDA bin and libnvvp directories
echo.
echo IMPORTANT: You must restart your computer for changes to take effect!
echo.
echo After restart, run:
echo   python verify_gpu_setup.py
echo.
pause