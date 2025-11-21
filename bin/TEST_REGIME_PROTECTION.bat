@echo off
REM TEST REGIME PROTECTION - Test market regime protection system

echo ===============================================================================
echo REGIME PROTECTION TEST
echo ===============================================================================
echo Testing market regime detection and strategy blocking...
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run regime protection test
python REGIME_PROTECTED_TRADING.py

echo.
echo ===============================================================================
echo Test complete
echo ===============================================================================
pause
