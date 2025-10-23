@echo off
REM Quick launcher for Position Monitor
REM Usage: Double-click to run or use from command line

echo ================================================================================
echo POSITION MONITOR - Quick Launcher
echo ================================================================================
echo.
echo Choose an option:
echo.
echo 1. View Positions (Single Snapshot)
echo 2. Watch Mode (Auto-refresh every 30 seconds)
echo 3. Watch Mode (Custom interval)
echo 4. Export to JSON
echo 5. Watch Mode (10 second intervals)
echo.
echo Q. Quit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" goto snapshot
if "%choice%"=="2" goto watch
if "%choice%"=="3" goto custom
if "%choice%"=="4" goto json
if "%choice%"=="5" goto watch10
if /i "%choice%"=="q" goto end

echo Invalid choice!
pause
goto end

:snapshot
cls
python monitor_positions.py
pause
goto end

:watch
cls
echo Starting watch mode (30 second intervals)...
echo Press Ctrl+C to stop
echo.
python monitor_positions.py --watch
pause
goto end

:custom
set /p interval="Enter refresh interval (seconds): "
cls
echo Starting watch mode (%interval% second intervals)...
echo Press Ctrl+C to stop
echo.
python monitor_positions.py --watch --interval %interval%
pause
goto end

:json
set timestamp=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
set filename=positions_snapshot_%timestamp%.json
python monitor_positions.py --json > %filename%
echo.
echo ================================================================================
echo Positions exported to: %filename%
echo ================================================================================
pause
goto end

:watch10
cls
echo Starting watch mode (10 second intervals)...
echo Press Ctrl+C to stop
echo.
python monitor_positions.py --watch --interval 10
pause
goto end

:end
