@echo off
REM ======================================================
REM START FOREX LIVE TRADER
REM Executes REAL trades on OANDA practice account
REM Scans EVERY SECOND for signals
REM ======================================================

echo.
echo ======================================================================
echo STARTING FOREX LIVE TRADER
echo ======================================================================
echo.
echo MODE: LIVE TRADING (OANDA Practice Account)
echo SCANNING: Every 1 second
echo PAIRS: EUR/USD, USD/JPY
echo TIMEFRAME: M1 (1-minute bars)
echo MAX TRADES: 50 per day
echo.
echo Press Ctrl+C to stop
echo.
echo ======================================================================
echo.

cd /d "%~dp0"
python forex_auto_trader.py

echo.
echo ======================================================================
echo FOREX TRADER STOPPED
echo ======================================================================
echo.
pause
