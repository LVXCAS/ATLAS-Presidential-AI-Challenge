@echo off
REM ATLAS Restart Script - Kills old processes and starts fresh

echo ============================================================
echo ATLAS RESTART PROCEDURE
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] Killing old Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
echo      Done.
echo.

timeout /t 2 /nobreak >nul

echo [2/4] Verifying processes stopped...
tasklist | findstr /I "python" >nul
if %ERRORLEVEL% EQU 0 (
    echo      WARNING: Some Python processes still running
) else (
    echo      All Python processes stopped.
)
echo.

echo [3/4] Starting ATLAS in background...
start /B pythonw run_paper_training.py --phase validation
echo      Started.
echo.

timeout /t 3 /nobreak >nul

echo [4/4] Verifying ATLAS started...
tasklist | findstr /I "pythonw" >nul
if %ERRORLEVEL% EQU 0 (
    echo      SUCCESS: ATLAS is running
) else (
    echo      ERROR: ATLAS did not start
)
echo.

echo ============================================================
echo To check status: python check_atlas_status.py
echo To view trades:  python check_trades.py
echo ============================================================
echo.
pause
