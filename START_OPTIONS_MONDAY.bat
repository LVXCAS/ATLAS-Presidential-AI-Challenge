@echo off
echo ======================================================================
echo OPTIONS SCANNER - S&P 500 (Runs Monday 6:30 AM - 1 PM PST)
echo ======================================================================
echo.
echo Coverage: 100 S&P 500 stocks
echo Strategies: Bull Put Spreads, Iron Condors, Long Calls/Puts
echo Scan Interval: 30 minutes
echo API: Working Alpaca keys (can execute trades)
echo.
echo Markets must be OPEN for this to work!
echo.
echo Press Ctrl+C to stop
echo ======================================================================
echo.

cd "c:\Users\lucas\PC-HIVE-TRADING"
python AGENTIC_OPTIONS_SCANNER_SP500.py

pause
