@echo off
REM ========================================================================
REM RIGHT-CLICK THIS FILE → "RUN AS ADMINISTRATOR"
REM ========================================================================

color 0A
title PC-HIVE - Quick Setup (Admin Required)

echo.
echo ========================================================================
echo   PC-HIVE AUTO SCANNER SETUP
echo ========================================================================
echo.
echo This will configure your system to automatically:
echo   - Run the scanner every day at 6:30 AM PT
echo   - Start after PC reboots
echo   - Never miss a trading day!
echo.
echo Press any key to begin setup...
echo.
pause >nul

REM Navigate to the correct directory
cd /d "%~dp0"

REM Run the main setup script
call SETUP_AUTOMATED_STARTUP.bat

echo.
echo ========================================================================
echo.
echo Setup complete! Check the results above.
echo.
echo To verify: Run CHECK_SCANNER_STATUS.bat
echo.
pause
