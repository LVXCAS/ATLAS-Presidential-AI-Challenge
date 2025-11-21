@echo off
REM ========================================
REM START FOREX AUTO-TRADER IN BACKGROUND
REM Runs silently without console window
REM ========================================

echo Starting Forex Auto-Trader in background...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Start in background (no window)
start /B pythonw forex_auto_trader.py > logs\forex_trader_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1

echo.
echo Forex Auto-Trader started in background
echo Check logs\forex_trader_YYYYMMDD.log for output
echo Use STOP_FOREX_TRADER.bat to stop
echo.

timeout /t 3
