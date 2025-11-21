@echo off
REM ========================================================================
REM CHECK_SCANNER_STATUS.bat - Verify Scanner is Running
REM ========================================================================

color 0B
title PC-HIVE Scanner Status Check

cd /d "%~dp0"

echo.
echo ========================================================================
echo   PC-HIVE AUTO OPTIONS SCANNER - STATUS CHECK
echo ========================================================================
echo.

REM Check if Python process is running
echo [1] Checking for running scanner process...
echo.
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq auto_options_scanner*" 2>nul | find /I "python.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Scanner process is RUNNING
    tasklist /FI "IMAGENAME eq python.exe" /V | find /I "auto_options_scanner"
) else (
    tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | find /I "pythonw.exe" >nul
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Scanner process is RUNNING in background
        tasklist /FI "IMAGENAME eq pythonw.exe"
    ) else (
        echo [WARNING] No scanner process detected
        echo.
        echo The scanner may not be running!
        echo Run START_TRADING.bat to start it.
    )
)

echo.
echo ========================================================================
echo   SCANNER STATUS FILE
echo ========================================================================
echo.

REM Check status file
if exist "auto_scanner_status.json" (
    echo [OK] Status file found: auto_scanner_status.json
    echo.
    type auto_scanner_status.json
) else (
    echo [WARNING] Status file not found
    echo The scanner has never run or was deleted.
)

echo.
echo ========================================================================
echo   RECENT LOG FILES
echo ========================================================================
echo.

if exist "logs\" (
    echo Last 5 log files:
    echo.
    dir /B /O-D logs\scanner_*.log 2>nul | findstr /N "^" | findstr /R "^[1-5]:"
    echo.
    echo To view latest log: type logs\[filename]
) else (
    echo [WARNING] No log directory found
)

echo.
echo ========================================================================
echo   SCHEDULED TASK STATUS
echo ========================================================================
echo.

REM Check if scheduled task exists
schtasks /Query /TN "PC-HIVE Auto Scanner" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Scheduled task exists: PC-HIVE Auto Scanner
    echo.
    schtasks /Query /TN "PC-HIVE Auto Scanner" /V /FO LIST | find /I "Status"
    schtasks /Query /TN "PC-HIVE Auto Scanner" /V /FO LIST | find /I "Next Run Time"
    schtasks /Query /TN "PC-HIVE Auto Scanner" /V /FO LIST | find /I "Last Run Time"
) else (
    echo [WARNING] Scheduled task not found
    echo Run SETUP_AUTOMATED_STARTUP.bat to create it.
)

echo.
echo ========================================================================
pause
