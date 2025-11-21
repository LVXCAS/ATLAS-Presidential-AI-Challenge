@echo off
REM ========================================================================
REM START_TRADING_BACKGROUND.bat - Background Startup (No Window)
REM ========================================================================
REM This script starts the scanner in the background without showing a window
REM Perfect for automatic startup via Task Scheduler
REM ========================================================================

cd /d "%~dp0"

REM Create logs directory
if not exist "logs" mkdir logs

REM Set log file
set LOGFILE=logs\scanner_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

REM Log startup
echo [%date% %time%] Auto Options Scanner - Background startup >> "%LOGFILE%"

REM Start Python script hidden (no window)
start /B pythonw auto_options_scanner.py --daily >> "%LOGFILE%" 2>&1

REM Exit immediately
exit
