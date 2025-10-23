@echo off
REM ========================================================================
REM START_TRADING.bat - One-Click Trading System Startup
REM ========================================================================
REM This script starts the automatic options scanner in daily mode
REM The scanner will run every day at 6:30 AM PT automatically
REM ========================================================================

color 0A
title PC-HIVE Auto Options Scanner - STARTING

echo.
echo ========================================================================
echo   PC-HIVE AUTO OPTIONS SCANNER
echo   One-Click Startup
echo ========================================================================
echo.
echo [%date% %time%] Starting automatic options scanner...
echo.

REM Change to the correct directory
cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Set log file with timestamp
set LOGFILE=logs\scanner_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo [INFO] Log file: %LOGFILE%
echo.

REM Start the scanner in daily mode with logging
echo [INFO] Launching scanner in DAILY mode (6:30 AM PT)...
echo [INFO] Press Ctrl+C in the window to stop
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak > nul

REM Run Python script with daily mode (default) and log output
python auto_options_scanner.py --daily > "%LOGFILE%" 2>&1

REM If script exits, show error
echo.
echo ========================================================================
echo   SCANNER STOPPED
echo ========================================================================
echo.
echo Check log file for details: %LOGFILE%
echo.
pause
