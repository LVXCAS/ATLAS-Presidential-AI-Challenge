@echo off
REM CHECK SYSTEM HEALTH - Run health check on all systems

echo ===============================================================================
echo SYSTEM HEALTH CHECK
echo ===============================================================================
echo Checking health of all trading systems...
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run health monitor
python SYSTEM_HEALTH_MONITOR.py

echo.
echo ===============================================================================
echo Health check complete
echo ===============================================================================
pause
