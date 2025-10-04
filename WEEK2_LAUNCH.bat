@echo off
REM ============================================================================
REM WEEK 2 S&P 500 SCANNER - FULL UNIVERSE LAUNCH
REM ============================================================================
REM
REM Week 2 Upgrades:
REM - 503 S&P 500 stocks (vs 5-8 in Week 1)
REM - 5-10 trades per day (vs 2 in Week 1)
REM - 10-15%% weekly ROI target (vs 5-8%% in Week 1)
REM - Full ML/DL/RL systems active
REM - All strategies: Intel dual, straddles, iron condors, butterflies
REM
REM ============================================================================

title WEEK 2 - S&P 500 MOMENTUM SCANNER

echo.
echo ========================================================================
echo WEEK 2 S&P 500 MOMENTUM SCANNER
echo ========================================================================
echo Universe: 503 S&P 500 stocks
echo Target: 10-15%% weekly ROI
echo Max trades: 5-10 per day
echo Risk per trade: 2%% (up from 1.5%%)
echo ========================================================================
echo.

echo [STARTING] Week 2 S&P 500 Scanner...
echo.

python week2_sp500_scanner.py

echo.
echo ========================================================================
echo WEEK 2 SCANNER STOPPED
echo ========================================================================
echo.

pause
