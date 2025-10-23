@echo off
REM MONDAY MORNING EXECUTION SCRIPT
REM Run this at 9:30 AM Monday to start enhanced scanner

echo ============================================================
echo MONDAY MORNING - WEEK 2 ENHANCED TRADING
echo ============================================================
echo.
echo Time: %TIME%
echo.
echo Starting enhanced scanner with all 4 learnings...
echo.
echo [1/4] Account verification
echo [2/4] Market regime detection
echo [3/4] Multi-source data fetcher (10x speed)
echo [4/4] Dynamic strategy selection
echo.
echo ============================================================
echo.

cd /d C:\Users\lucas\PC-HIVE-TRADING

REM Run the enhanced scanner
python week3_production_scanner.py

pause
