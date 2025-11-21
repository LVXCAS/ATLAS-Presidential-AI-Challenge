@echo off
REM LAUNCH ALL 3 OPTIMIZED E8 BOTS
REM Expected combined ROI: ~25% per 90 days

echo ================================================================
echo E8 OPTIMIZED MULTI-PAIR DEPLOYMENT
echo ================================================================
echo.
echo Starting 3 optimized bots:
echo   1. EUR/USD: Score=2.5, Risk=2.5%%, Target=2%%, Stop=1%% (+10.55%% ROI)
echo   2. GBP/USD: Score=2.0, Risk=2.0%%, Target=3%%, Stop=1%% (+3.58%% ROI)
echo   3. USD/JPY: Score=2.0, Risk=1.5%%, Target=3%%, Stop=1.5%% (+11.03%% ROI)
echo.
echo Combined Expected ROI: ~25%% per 90 days
echo Days to Pass E8: ~36 days (vs 183 days baseline)
echo ================================================================
echo.

REM Launch EUR/USD bot
echo [1/3] Launching EUR/USD bot...
start "E8-EUR/USD" pythonw E8_EUR_USD_BOT.py
timeout //t 2 //nobreak >nul

REM Launch GBP/USD bot
echo [2/3] Launching GBP/USD bot...
start "E8-GBP/USD" pythonw E8_GBP_USD_BOT.py
timeout //t 2 //nobreak >nul

REM Launch USD/JPY bot
echo [3/3] Launching USD/JPY bot...
start "E8-USD/JPY" pythonw E8_USD_JPY_BOT.py
timeout //t 2 //nobreak >nul

echo.
echo ================================================================
echo ALL 3 BOTS DEPLOYED!
echo ================================================================
echo.
echo Monitor with: tasklist ^| findstr pythonw
echo Kill all: taskkill //F //IM pythonw.exe
echo.
echo Press any key to exit...
pause >nul
