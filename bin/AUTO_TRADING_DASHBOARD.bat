@echo off
REM ================================================================
REM AUTOMATIC TRADING DASHBOARD
REM Control Center for All Trading Systems
REM ================================================================

:MENU
cls
echo ================================================================
echo          HIVE TRADING - AUTOMATIC SYSTEMS CONTROL
echo ================================================================
echo.
echo Date: %date%
echo Time: %time:~0,8%
echo.
echo ================================================================
echo ACTIVE SYSTEMS:
echo ================================================================
echo.
echo [1] Start ALL Systems (Options + Forex + Futures)
echo [2] Start Options Auto-Scanner (Daily Mode)
echo [3] Start Forex Paper Trader (15-min scans)
echo [4] Start Futures Observation (48 hours)
echo [5] Monitor Positions (Live Watch)
echo.
echo [6] Stop All Systems
echo [7] View System Status
echo [8] View Today's Performance
echo.
echo [9] Exit
echo.
echo ================================================================
echo.

set /p choice="Select option (1-9): "

if "%choice%"=="1" goto START_ALL
if "%choice%"=="2" goto START_OPTIONS
if "%choice%"=="3" goto START_FOREX
if "%choice%"=="4" goto START_FUTURES
if "%choice%"=="5" goto MONITOR
if "%choice%"=="6" goto STOP_ALL
if "%choice%"=="7" goto STATUS
if "%choice%"=="8" goto PERFORMANCE
if "%choice%"=="9" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:START_ALL
cls
echo ================================================================
echo STARTING ALL TRADING SYSTEMS
echo ================================================================
echo.
echo [1/4] Starting Options Auto-Scanner...
start /min python auto_options_scanner.py --daily
timeout /t 2 >nul
echo [OK] Options scanner running
echo.
echo [2/4] Starting Forex Paper Trader...
start /min python forex_paper_trader.py --interval 15
timeout /t 2 >nul
echo [OK] Forex trader running
echo.
echo [3/4] Starting Futures Observation...
start /min python futures_live_validation.py --duration 48
timeout /t 2 >nul
echo [OK] Futures observer running
echo.
echo [4/4] Starting Position Monitor...
start /min python monitor_positions.py --watch --interval 30
timeout /t 2 >nul
echo [OK] Position monitor running
echo.
echo ================================================================
echo ALL SYSTEMS ACTIVE
echo ================================================================
echo.
echo Your trading systems are now running 24/7:
echo - Options: Scanning daily at 6:30 AM PT
echo - Forex: Scanning every 15 minutes
echo - Futures: 48-hour observation mode
echo - Monitor: Updating every 30 seconds
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:START_OPTIONS
cls
echo ================================================================
echo STARTING OPTIONS AUTO-SCANNER
echo ================================================================
echo.
set /p mode="Select mode (1=Daily, 2=Continuous, 3=Once): "
if "%mode%"=="1" (
    echo Starting in DAILY mode (scans at 6:30 AM PT)...
    start python auto_options_scanner.py --daily
)
if "%mode%"=="2" (
    echo Starting in CONTINUOUS mode (scans every 4 hours)...
    start python auto_options_scanner.py --continuous
)
if "%mode%"=="3" (
    echo Running ONE scan now...
    python auto_options_scanner.py --once
    pause
    goto MENU
)
echo.
echo [OK] Options scanner started
timeout /t 3 >nul
goto MENU

:START_FOREX
cls
echo ================================================================
echo STARTING FOREX PAPER TRADER
echo ================================================================
echo.
echo Strategy: EUR/USD EMA v3.0 (63.6%% WR)
echo Scan Interval: Every 15 minutes
echo.
start python forex_paper_trader.py --interval 15
echo [OK] Forex trader started
timeout /t 3 >nul
goto MENU

:START_FUTURES
cls
echo ================================================================
echo STARTING FUTURES OBSERVATION MODE
echo ================================================================
echo.
echo Duration: 48 hours
echo Will track MES/MNQ signals without executing
echo Calculates win rate from live observations
echo.
start python futures_live_validation.py --duration 48
echo [OK] Futures observer started
timeout /t 3 >nul
goto MENU

:MONITOR
cls
echo ================================================================
echo POSITION MONITOR - LIVE VIEW
echo ================================================================
echo.
echo Launching live position monitor...
echo Updates every 30 seconds
echo Press Ctrl+C in monitor window to stop
echo.
python monitor_positions.py --watch --interval 30
goto MENU

:STOP_ALL
cls
echo ================================================================
echo STOPPING ALL TRADING SYSTEMS
echo ================================================================
echo.
taskkill /FI "WINDOWTITLE eq auto_options_scanner*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq forex_paper_trader*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq futures_live_validation*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq monitor_positions*" /F >nul 2>&1
echo.
echo [OK] All systems stopped
echo.
timeout /t 3 >nul
goto MENU

:STATUS
cls
echo ================================================================
echo SYSTEM STATUS
echo ================================================================
echo.
echo Checking active processes...
echo.
tasklist | findstr /i "python" | findstr /i "auto_options_scanner forex_paper_trader futures_live_validation monitor_positions"
if errorlevel 1 (
    echo No trading systems currently running
) else (
    echo Active trading systems detected
)
echo.
echo ================================================================
echo LOG FILES:
echo ================================================================
if exist forex_trading.log (
    echo [FOREX] forex_trading.log - Last updated:
    for %%f in (forex_trading.log) do echo %%~tf
)
if exist futures_validation.log (
    echo [FUTURES] futures_validation.log - Last updated:
    for %%f in (futures_validation.log) do echo %%~tf
)
if exist monitor_output.log (
    echo [MONITOR] monitor_output.log - Last updated:
    for %%f in (monitor_output.log) do echo %%~tf
)
echo.
pause
goto MENU

:PERFORMANCE
cls
echo ================================================================
echo TODAY'S PERFORMANCE
echo ================================================================
echo.
python monitor_positions.py
echo.
pause
goto MENU

:EXIT
cls
echo.
echo ================================================================
echo Thank you for using Hive Trading Systems
echo ================================================================
echo.
echo Systems Status:
tasklist | findstr /i "python" | findstr /i "auto_options_scanner forex_paper_trader" >nul
if not errorlevel 1 (
    echo [WARNING] Trading systems are still running
    echo Close them from Task Manager if needed
)
echo.
echo Goodbye!
timeout /t 3 >nul
exit
