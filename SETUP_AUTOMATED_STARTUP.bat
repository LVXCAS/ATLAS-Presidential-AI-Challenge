@echo off
REM ========================================================================
REM SETUP_AUTOMATED_STARTUP.bat - Configure Windows Task Scheduler
REM ========================================================================
REM This script creates a scheduled task to run the scanner daily at 6:30 AM PT
REM Requires Administrator privileges
REM ========================================================================

REM Check for admin rights
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================================
    echo   ADMINISTRATOR PRIVILEGES REQUIRED
    echo ========================================================================
    echo.
    echo This script must be run as Administrator to create scheduled tasks.
    echo.
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

color 0E
title PC-HIVE - Automated Startup Setup

echo.
echo ========================================================================
echo   PC-HIVE AUTO OPTIONS SCANNER
echo   Automated Startup Configuration
echo ========================================================================
echo.
echo This will create a Windows Scheduled Task to:
echo   - Run the scanner every day at 6:30 AM PT
echo   - Start automatically after system reboot
echo   - Run even if you're not logged in
echo   - Log all activity
echo.
echo ========================================================================
echo.

pause

REM Get current directory
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo.
echo [INFO] Script directory: %SCRIPT_DIR%
echo.

REM Delete existing task if it exists
echo [1] Removing any existing scheduled task...
schtasks /Delete /TN "PC-HIVE Auto Scanner" /F >nul 2>&1
echo [OK] Old task removed (if existed)
echo.

REM Create the scheduled task
echo [2] Creating new scheduled task...
echo.

schtasks /Create ^
    /TN "PC-HIVE Auto Scanner" ^
    /TR "\"%SCRIPT_DIR%\START_TRADING_BACKGROUND.bat\"" ^
    /SC DAILY ^
    /ST 06:30 ^
    /RU "%USERNAME%" ^
    /RL HIGHEST ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Scheduled task created successfully!
    echo.
) else (
    echo.
    echo [ERROR] Failed to create scheduled task
    echo.
    pause
    exit /b 1
)

REM Also create a task for system startup
echo [3] Creating system startup task...
echo.

schtasks /Create ^
    /TN "PC-HIVE Auto Scanner - Startup" ^
    /TR "\"%SCRIPT_DIR%\START_TRADING_BACKGROUND.bat\"" ^
    /SC ONSTART ^
    /RU "%USERNAME%" ^
    /RL HIGHEST ^
    /DELAY 0001:00 ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Startup task created successfully!
    echo.
) else (
    echo.
    echo [WARNING] Failed to create startup task (optional)
    echo.
)

echo.
echo ========================================================================
echo   SETUP COMPLETE
echo ========================================================================
echo.
echo The scanner is now configured to run automatically:
echo.
echo   1. Daily at 6:30 AM PT (market open)
echo   2. On system startup (with 1 minute delay)
echo.
echo You can verify this by running: CHECK_SCANNER_STATUS.bat
echo.
echo To manually start now: START_TRADING.bat
echo To view scheduled tasks: taskschd.msc
echo.
echo ========================================================================
echo.

pause
