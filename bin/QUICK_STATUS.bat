@echo off
REM Quick 30-second E8 status check for Score 3.0 aggressive mode

echo.
echo ========================================
echo E8 QUICK STATUS - %time%
echo ========================================
echo.

REM Check if bot running
tasklist | findstr pythonw.exe >nul
if %errorlevel% == 0 (
    echo [1/3] Bot Status: RUNNING ✓
) else (
    echo [1/3] Bot Status: STOPPED - RESTART NOW!
    goto end
)

echo.
echo [2/3] Current Positions:
python GET_E8_TRADE_IDS.py 2>nul | findstr /C:"Total Unrealized" /C:"Symbol:" /C:"Unrealized P/L" /C:"Direction:"

echo.
echo [3/3] Recent Bot Activity:
powershell -Command "Get-Content e8_score_log.csv -Tail 1" 2>nul

echo.
echo ========================================
echo SAFETY REMINDERS:
echo - Stop bot if equity ^< $196,000
echo - Close position if unrealized ^< -$2,500
echo - Monitor 3x daily (morning/lunch/evening)
echo ========================================
echo.

:end
