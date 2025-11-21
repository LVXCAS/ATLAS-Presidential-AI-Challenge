@echo off
echo ============================================================
echo CONTINUOUS WEEK 1 SCANNER - REAL MARKET DATA
echo ============================================================
echo.
echo Starting autonomous scanner...
echo Will scan every 5 minutes until market close (1:00 PM PDT)
echo Max trades: 2 (1 already executed - AAPL)
echo.
echo Press Ctrl+C to stop early
echo ============================================================
echo.

cd C:\Users\lucas\PC-HIVE-TRADING
python -u continuous_week1_scanner.py

echo.
echo ============================================================
echo Scanner stopped
echo ============================================================
pause
