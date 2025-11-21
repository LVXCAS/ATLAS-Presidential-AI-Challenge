@echo off
REM Forex Scanner - Runs every hour
REM To stop: Press Ctrl+C or close this window

:LOOP
echo.
echo ================================================================================
echo FOREX SCAN - %date% %time%
echo ================================================================================

python WORKING_FOREX_MONITOR.py

echo.
echo [WAITING] Next scan in 60 minutes...
echo Press Ctrl+C to stop

REM Wait 1 hour (3600 seconds)
timeout /t 3600 /nobreak > nul

goto LOOP