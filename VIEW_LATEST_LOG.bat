@echo off
REM ========================================================================
REM VIEW_LATEST_LOG.bat - View Most Recent Scanner Log
REM ========================================================================

color 0F
title PC-HIVE - Scanner Logs

cd /d "%~dp0"

echo.
echo ========================================================================
echo   SCANNER LOGS
echo ========================================================================
echo.

if not exist "logs\" (
    echo [ERROR] No logs directory found
    echo The scanner has never run yet.
    echo.
    pause
    exit /b 1
)

REM Get latest log file
for /f "delims=" %%i in ('dir /b /o-d logs\scanner_*.log 2^>nul') do (
    set LATEST_LOG=%%i
    goto :found
)

echo [ERROR] No log files found
echo.
pause
exit /b 1

:found
echo Latest log file: %LATEST_LOG%
echo.
echo ========================================================================
echo.

type "logs\%LATEST_LOG%"

echo.
echo ========================================================================
echo   End of log
echo ========================================================================
echo.
pause
