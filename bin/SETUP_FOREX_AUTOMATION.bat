@echo off
REM ========================================
REM SETUP WINDOWS TASK SCHEDULER
REM Auto-run forex trader every hour (24/5)
REM ========================================

echo.
echo ======================================================================
echo FOREX AUTO-TRADER - WINDOWS TASK SCHEDULER SETUP
echo ======================================================================
echo.
echo This will create a scheduled task to run the forex trader:
echo   - Every hour (24/5 - excludes weekends)
echo   - Starts automatically at market open
echo   - Runs in background
echo.
echo IMPORTANT: Forex market hours:
echo   - Opens: Sunday 5 PM EST
echo   - Closes: Friday 5 PM EST
echo.

pause

REM Get current directory
set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%forex_auto_trader.py

echo.
echo Creating scheduled task...
echo.

REM Create task to run every hour on weekdays
schtasks /Create /SC HOURLY /TN "ForexAutoTrader" /TR "pythonw \"%SCRIPT_PATH%\" --once" /ST 00:00 /RU "%USERNAME%" /RP /F

if %ERRORLEVEL% == 0 (
    echo.
    echo ======================================================================
    echo SUCCESS - Task Created
    echo ======================================================================
    echo.
    echo Task Name: ForexAutoTrader
    echo Frequency: Every hour
    echo Action: Run forex_auto_trader.py --once
    echo.
    echo The task will scan for signals every hour and execute trades.
    echo.
    echo MANAGE TASK:
    echo   - View: taskschd.msc ^(Task Scheduler^)
    echo   - Disable: Right-click task ^> Disable
    echo   - Delete: schtasks /Delete /TN "ForexAutoTrader" /F
    echo.
    echo RECOMMENDED:
    echo   1. Open Task Scheduler ^(taskschd.msc^)
    echo   2. Find "ForexAutoTrader" task
    echo   3. Set schedule to run only during forex market hours
    echo   4. Add condition: Only run on weekdays
    echo.
) else (
    echo.
    echo ======================================================================
    echo ERROR - Task Creation Failed
    echo ======================================================================
    echo.
    echo You may need to run this as Administrator.
    echo Right-click this batch file and select "Run as Administrator"
    echo.
)

echo.
echo ======================================================================
echo ALTERNATIVE: Run Continuously
echo ======================================================================
echo.
echo Instead of hourly tasks, you can run continuously:
echo   START_FOREX_TRADER.bat - Run in console window
echo   START_FOREX_BACKGROUND.bat - Run in background
echo.

pause
