@echo off
REM VIEW DAILY PERFORMANCE - Generate and display daily performance report

echo ===============================================================================
echo DAILY PERFORMANCE REPORT
echo ===============================================================================
echo Generating comprehensive performance report...
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run performance report
python DAILY_PERFORMANCE_REPORT.py

echo.
echo ===============================================================================
echo Report complete
echo ===============================================================================
pause
