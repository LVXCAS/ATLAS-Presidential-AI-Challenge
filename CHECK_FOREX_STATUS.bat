@echo off
REM ========================================
REM CHECK FOREX TRADING STATUS
REM Shows active positions and recent logs
REM ========================================

echo.
echo ======================================================================
echo FOREX TRADING STATUS
echo ======================================================================
echo.

REM Check if trader is running
echo [PROCESS CHECK]
tasklist /FI "IMAGENAME eq python.exe" | find /I "python.exe" >nul
if %ERRORLEVEL% == 0 (
    echo Status: RUNNING
    echo Python processes found
) else (
    echo Status: STOPPED
    echo No Python processes detected
)
echo.

REM Check for stop file
echo [EMERGENCY STOP]
if exist STOP_FOREX_TRADING.txt (
    echo Stop File: EXISTS - Trader will stop soon
) else (
    echo Stop File: NOT PRESENT - Trader running normally
)
echo.

REM Show today's trades
echo [TODAY'S TRADES]
set TODAY=%date:~-4,4%%date:~-10,2%%date:~-7,2%
if exist forex_trades\execution_log_%TODAY%.json (
    echo Log file: forex_trades\execution_log_%TODAY%.json
    type forex_trades\execution_log_%TODAY%.json
) else (
    echo No trades today
)
echo.

REM Show active positions
echo [ACTIVE POSITIONS]
if exist forex_trades\positions_%TODAY%.json (
    echo Position file: forex_trades\positions_%TODAY%.json
    type forex_trades\positions_%TODAY%.json
) else (
    echo No position data available
)
echo.

REM Show recent log entries
echo [RECENT LOGS]
if exist logs\forex_trader_%TODAY%.log (
    echo Last 20 lines from logs\forex_trader_%TODAY%.log:
    powershell -Command "Get-Content logs\forex_trader_%TODAY%.log -Tail 20"
) else (
    echo No log file for today
)
echo.

echo ======================================================================
echo STATUS CHECK COMPLETE
echo ======================================================================
echo.

pause
