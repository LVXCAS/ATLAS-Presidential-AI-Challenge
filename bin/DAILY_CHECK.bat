@echo off
REM ============================================================
REM DAILY E8 BOT CHECK - 2 Minute Morning/Evening Routine
REM ============================================================

echo.
echo ====================================================================
echo E8 BOT DAILY CHECK - %date% %time%
echo ====================================================================
echo.

REM 1. Check if bot is running
echo [1/4] Checking bot status...
tasklist | findstr pythonw.exe >nul
if %errorlevel% == 0 (
    echo   Status: RUNNING
) else (
    echo   Status: STOPPED - RESTART NEEDED!
    echo   Run: start pythonw BOTS\E8_FOREX_BOT.py
)
echo.

REM 2. Check for open positions
echo [2/4] Checking positions...
python GET_E8_TRADE_IDS.py
echo.

REM 3. Show recent score log activity
echo [3/4] Recent bot activity (last 3 scans)...
powershell -Command "Get-Content e8_score_log.csv | Select-Object -Last 9"
echo.

REM 4. Check state file
echo [4/4] Peak balance tracking...
type BOTS\e8_bot_state.json
echo.

echo ====================================================================
echo SAFETY CHECKS:
echo ====================================================================
echo.
echo - Equity should be ^>$194,978 (safe zone)
echo - No position should have ^>-$2,000 unrealized loss
echo - Bot should scan every hour (check timestamps)
echo.
echo If anything looks wrong: taskkill /F /IM pythonw.exe
echo ====================================================================
echo.

pause
