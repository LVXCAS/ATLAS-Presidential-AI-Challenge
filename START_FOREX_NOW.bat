@echo off
echo ======================================================================
echo STARTING FOREX V4 OPTIMIZED - SCORE THRESHOLD 2.0
echo ======================================================================
echo.
echo EUR/USD + USD/JPY
echo Scan Interval: 1 hour
echo Score Threshold: 2.0 (AGGRESSIVE - will execute trades)
echo.
echo Press Ctrl+C to stop
echo ======================================================================
echo.

cd "c:\Users\lucas\PC-HIVE-TRADING"
python forex_v4_optimized.py

pause
