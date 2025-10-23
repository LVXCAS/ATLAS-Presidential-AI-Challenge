@echo off
REM ========================================
REM START FOREX AUTO-TRADER
REM Paper trading by default
REM ========================================

echo.
echo ======================================================================
echo FOREX AUTO-TRADER - STARTING
echo ======================================================================
echo Mode: PAPER TRADING (Safe)
echo Pairs: EUR/USD, USD/JPY
echo Timeframe: 1 Hour
echo ======================================================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Start trader
echo Starting forex auto-trader...
echo Press Ctrl+C to stop
echo.

python forex_auto_trader.py

echo.
echo ======================================================================
echo FOREX AUTO-TRADER - STOPPED
echo ======================================================================
echo.

pause
