@echo off
echo ================================================================================
echo FOREX ELITE - BALANCED STRATEGY LAUNCHER
echo 62-75%% Win Rate - More Frequent Trades
echo ================================================================================
echo.
echo Starting Forex Elite with BALANCED strategy (Paper Trading)...
echo.
echo Press Ctrl+C to stop gracefully
echo Create STOP_FOREX_TRADING.txt for emergency stop
echo.
echo ================================================================================
echo.

python -u START_FOREX_ELITE.py --strategy balanced

echo.
echo ================================================================================
echo Forex Elite Stopped
echo ================================================================================
pause
