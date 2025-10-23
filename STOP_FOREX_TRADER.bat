@echo off
REM ========================================
REM EMERGENCY STOP - FOREX AUTO-TRADER
REM Creates stop file to halt trading
REM ========================================

echo.
echo ======================================================================
echo EMERGENCY STOP - FOREX AUTO-TRADER
echo ======================================================================
echo.

REM Create emergency stop file
echo. > STOP_FOREX_TRADING.txt

echo [STOP FILE CREATED]
echo.
echo The trader will stop at next check and close all positions.
echo This may take up to 5 minutes (position check interval).
echo.
echo Stop file: STOP_FOREX_TRADING.txt
echo.

REM Also try to kill python process running forex_auto_trader.py
echo Attempting to terminate forex_auto_trader.py processes...
taskkill /F /FI "WINDOWTITLE eq forex_auto_trader*" >nul 2>&1
taskkill /F /FI "IMAGENAME eq python.exe" /FI "MEMUSAGE gt 10000" >nul 2>&1

echo.
echo ======================================================================
echo STOP SIGNAL SENT
echo ======================================================================
echo.
echo Check position status with: CHECK_FOREX_STATUS.bat
echo.

pause
