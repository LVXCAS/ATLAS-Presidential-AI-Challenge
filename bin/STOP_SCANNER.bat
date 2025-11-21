@echo off
REM ========================================================================
REM STOP_SCANNER.bat - Stop Running Scanner
REM ========================================================================

color 0C
title PC-HIVE - Stop Scanner

echo.
echo ========================================================================
echo   STOPPING AUTO OPTIONS SCANNER
echo ========================================================================
echo.

REM Kill all Python processes running auto_options_scanner
echo [INFO] Stopping scanner processes...
echo.

taskkill /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq auto_options_scanner*" /F >nul 2>&1
taskkill /FI "IMAGENAME eq pythonw.exe" /F >nul 2>&1

timeout /t 2 /nobreak > nul

REM Verify stopped
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "auto_options_scanner" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [OK] Scanner stopped successfully
) else (
    echo [WARNING] Some processes may still be running
    echo Try running this script again
)

echo.
echo ========================================================================
pause
