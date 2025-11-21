@echo off
REM ========================================================================
REM EMERGENCY_STOP.bat - IMMEDIATELY STOP ALL TRADING
REM ========================================================================
REM This will:
REM   1. Stop the scanner
REM   2. Disable scheduled tasks
REM   3. Prevent any new trades
REM ========================================================================

color 4F
title EMERGENCY STOP - PC-HIVE TRADING

echo.
echo ========================================================================
echo   EMERGENCY STOP - SHUTTING DOWN ALL TRADING SYSTEMS
echo ========================================================================
echo.

REM Stop all scanner processes
echo [1] Stopping all scanner processes...
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM pythonw.exe /F >nul 2>&1
echo [OK] Processes stopped
echo.

REM Disable scheduled tasks
echo [2] Disabling scheduled tasks...
schtasks /Change /TN "PC-HIVE Auto Scanner" /DISABLE >nul 2>&1
schtasks /Change /TN "PC-HIVE Auto Scanner - Startup" /DISABLE >nul 2>&1
echo [OK] Scheduled tasks disabled
echo.

REM Create emergency stop flag file
echo [3] Creating emergency stop flag...
echo EMERGENCY_STOP_ACTIVATED > emergency_stop.flag
echo Timestamp: %date% %time% >> emergency_stop.flag
echo [OK] Emergency flag created
echo.

echo ========================================================================
echo   ALL TRADING SYSTEMS STOPPED
echo ========================================================================
echo.
echo What was done:
echo   [X] All scanner processes killed
echo   [X] Scheduled tasks disabled
echo   [X] Emergency stop flag created
echo.
echo To resume trading:
echo   1. Delete emergency_stop.flag file
echo   2. Run SETUP_AUTOMATED_STARTUP.bat
echo   3. Run START_TRADING.bat
echo.
echo ========================================================================
pause
