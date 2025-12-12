@echo off
REM ATLAS Startup Script - Exploration Phase (Paper Trading)
REM Starts ATLAS with OANDA demo account for 60-day training

echo ============================================================
echo ATLAS EXPLORATION PHASE STARTUP
echo ============================================================
echo.
echo Phase: EXPLORATION (20 days, threshold 2.5, fast learning)
echo Mode: Paper Trading (OANDA Demo)
echo Scan Interval: 5 minutes (or 1 min with --fast-scan)
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if user wants fast scan for testing
set /P FAST_SCAN="Use fast scan (1 min intervals) for testing? (y/N): "

if /I "%FAST_SCAN%"=="y" (
    echo Starting ATLAS in FAST SCAN mode (1-minute intervals)...
    start cmd /k "python run_paper_training.py --phase exploration --fast-scan"
) else (
    echo Starting ATLAS in normal mode (5-minute intervals)...
    start cmd /k "python run_paper_training.py --phase exploration"
)

echo.
echo ✓ ATLAS started in new window
echo.
echo WHAT YOU SHOULD SEE:
echo - TensorFlow loading (10-15 seconds)
echo - All 12 agents initializing
echo - OANDA connection established
echo - First scan completing in 30-60 seconds
echo - System waiting 5 minutes between scans
echo.
echo The system is NOT hanging - it waits between scans!
echo You'll see "[WAITING]" messages during the wait period.
echo.
echo Press any key to exit this window...
pause >nul
