@echo off
echo ================================================================================
echo FOREX ELITE STATUS CHECK
echo ================================================================================
echo.

echo [CONFIG FILE]
if exist "config\forex_elite_config.json" (
    echo   Config: FOUND
    python -c "import json; c=json.load(open('config/forex_elite_config.json')); print('  Strategy: ' + c['strategy']['name']); print('  Mode: ' + ('PAPER' if c['account']['paper_trading'] else 'LIVE')); print('  Pairs: ' + ', '.join(c['trading']['pairs'])); print('  Risk/Trade: ' + str(c['trading']['risk_per_trade']*100) + '%%')"
) else (
    echo   Config: NOT FOUND - Run START_FOREX_ELITE.py first
)

echo.
echo [RECENT LOGS]
if exist "logs\" (
    dir /O-D /B logs\forex_elite*.log 2>nul | findstr . >nul
    if errorlevel 1 (
        echo   No forex elite logs found
    ) else (
        for /f "tokens=*" %%a in ('dir /O-D /B logs\forex_elite*.log 2^>nul ^| more +0') do (
            echo   Latest: %%a
            goto :found_log
        )
        :found_log
    )
)

echo.
echo [TRADE LOGS]
if exist "forex_trades\" (
    dir /O-D /B forex_trades\execution_log*.json 2>nul | findstr . >nul
    if errorlevel 1 (
        echo   No trade logs found yet
    ) else (
        for /f "tokens=*" %%a in ('dir /O-D /B forex_trades\execution_log*.json 2^>nul ^| more +0') do (
            echo   Latest: %%a
            goto :found_trade
        )
        :found_trade
    )
)

echo.
echo [EMERGENCY STOP]
if exist "STOP_FOREX_TRADING.txt" (
    echo   Status: STOP FILE EXISTS - System will halt
    echo   Action: Delete STOP_FOREX_TRADING.txt to allow trading
) else (
    echo   Status: OK - No stop file
)

echo.
echo [OANDA CONNECTION]
python -c "import os; print('  Account ID: ' + ('SET' if os.getenv('OANDA_ACCOUNT_ID') else 'NOT SET')); print('  API Key: ' + ('SET' if os.getenv('OANDA_API_KEY') else 'NOT SET'))"

echo.
echo ================================================================================
echo.
pause
