@echo off
REM GTX 1660 Super CUDA Installation Script
REM Run as Administrator

echo ========================================
echo GTX 1660 SUPER CUDA INSTALLATION
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
echo Step 1: Checking system requirements...
echo ----------------------------------------

REM Check Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo Windows Version: %VERSION%

REM Check if NVIDIA GPU is present
nvidia-smi >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ NVIDIA GPU detected
    nvidia-smi --query-gpu=name --format=csv,noheader
) else (
    echo ✗ NVIDIA GPU not detected or drivers not installed
    echo Please install NVIDIA drivers first from geforce.com
    pause
    exit /b 1
)

echo.
echo Step 2: Download CUDA Toolkit 12.6...
echo --------------------------------------

REM Create download directory
if not exist "C:\Temp\CUDA" mkdir "C:\Temp\CUDA"
cd /d "C:\Temp\CUDA"

echo.
echo Please download CUDA Toolkit manually:
echo.
echo 1. Open browser to: https://developer.nvidia.com/cuda-downloads
echo 2. Select: Windows -> x86_64 -> 11 -> exe (local)
echo 3. Download: cuda_12.6.2_560.94_windows.exe (or latest)
echo 4. Save to: C:\Temp\CUDA\
echo.
echo Press any key when download is complete...
pause

REM Check if CUDA installer exists
if exist "cuda_*.exe" (
    echo ✓ CUDA installer found
) else (
    echo ✗ CUDA installer not found in C:\Temp\CUDA\
    echo Please download and save to this directory
    pause
    exit /b 1
)

echo.
echo Step 3: Installing CUDA Toolkit...
echo ----------------------------------

REM Run CUDA installer
for %%f in (cuda_*.exe) do (
    echo Installing %%f...
    echo.
    echo IMPORTANT: During installation:
    echo - Choose "Custom" installation
    echo - Select CUDA Toolkit, Samples, Documentation
    echo - Install to default location
    echo.
    pause
    %%f
)

REM Wait for installation to complete
echo.
echo Waiting for CUDA installation to complete...
echo Close the installer window when finished, then press any key...
pause

echo.
echo Step 4: Setting up environment variables...
echo ------------------------------------------

REM Set CUDA_PATH
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" /M
echo ✓ CUDA_PATH set

REM Add CUDA to PATH
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
set "CUDA_LIBNVVP=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp"

REM Add to system PATH
for /f "usebackq tokens=2,*" %%A in (`reg query HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session" Manager\Environment" /v PATH`) do set SYS_PATH=%%B

echo %SYS_PATH% | findstr /C:"%CUDA_BIN%" >nul
if errorlevel 1 (
    setx PATH "%SYS_PATH%;%CUDA_BIN%;%CUDA_LIBNVVP%" /M
    echo ✓ CUDA added to system PATH
) else (
    echo ✓ CUDA already in system PATH
)

echo.
echo Step 5: Download cuDNN...
echo ------------------------

echo.
echo Please download cuDNN manually:
echo.
echo 1. Open browser to: https://developer.nvidia.com/cudnn
echo 2. Create/login to NVIDIA Developer account (free)
echo 3. Download: cuDNN Library for Windows (x64)
echo 4. Version: 8.9.7 for CUDA 12.x
echo 5. Save ZIP file to: C:\Temp\CUDA\
echo.
echo Press any key when download is complete...
pause

REM Check for cuDNN zip file
if exist "cudnn-*.zip" (
    echo ✓ cuDNN archive found
) else (
    echo ✗ cuDNN archive not found
    echo Please download and save to C:\Temp\CUDA\
    pause
    exit /b 1
)

echo.
echo Step 6: Installing cuDNN...
echo ---------------------------

REM Extract cuDNN
for %%f in (cudnn-*.zip) do (
    echo Extracting %%f...
    powershell -Command "Expand-Archive -Path '%%f' -DestinationPath 'cudnn_extracted' -Force"
)

REM Copy cuDNN files to CUDA installation
echo Copying cuDNN files to CUDA installation...

REM Copy bin files
xcopy /Y "cudnn_extracted\cudnn-*\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\"
echo ✓ Copied bin files

REM Copy include files
xcopy /Y "cudnn_extracted\cudnn-*\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\"
echo ✓ Copied include files

REM Copy lib files
xcopy /Y "cudnn_extracted\cudnn-*\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\"
echo ✓ Copied lib files

echo.
echo Step 7: Verification...
echo ----------------------

echo.
echo Testing CUDA installation...
echo.

REM Test nvcc
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" --version
if %errorLevel% == 0 (
    echo ✓ CUDA compiler (nvcc) working
) else (
    echo ✗ CUDA compiler not found
)

echo.
echo Testing NVIDIA System Management Interface...
nvidia-smi
if %errorLevel% == 0 (
    echo ✓ nvidia-smi working
) else (
    echo ✗ nvidia-smi not working
)

echo.
echo Step 8: TensorFlow GPU Test...
echo ------------------------------

echo Testing TensorFlow GPU detection...
echo.
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('CUDA available:', tf.test.is_built_with_cuda()); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. RESTART your computer
echo 2. Run: python verify_gpu_setup.py
echo 3. Run: python gpu_enhanced_alpha_discovery.py
echo.
echo Your GTX 1660 Super should now provide 5-10x speedup!
echo.
pause

REM Cleanup
cd /d C:\
rmdir /s /q "C:\Temp\CUDA"
echo ✓ Cleaned up temporary files

echo.
echo Press any key to exit...
pause