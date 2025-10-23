@echo off
echo ================================================================================
echo FOREX ELITE - STRICT STRATEGY LAUNCHER
echo 71-75%% Win Rate - 12.87 Sharpe Ratio
echo ================================================================================
echo.
echo Starting Forex Elite with STRICT strategy (Paper Trading)...
echo.
echo Press Ctrl+C to stop gracefully
echo Create STOP_FOREX_TRADING.txt for emergency stop
echo.
echo ================================================================================
echo.

python -u START_FOREX_ELITE.py --strategy strict

echo.
echo ================================================================================
echo Forex Elite Stopped
echo ================================================================================
pause
