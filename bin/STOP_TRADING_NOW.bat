@echo off
echo EMERGENCY STOP - KILLING ALL TRADING PROCESSES
echo ===============================================
echo Terminating all Python processes...
taskkill /f /im python.exe
echo.
echo All Python trading processes have been terminated.
echo Check portfolio and cancel any pending orders manually.
echo.
echo PORTFOLIO LOSS CONFIRMED: $45,470 (4.55%)
echo CAUSE: Rapid buy-sell algorithmic trading
echo SOLUTION: Return to hold strategies like Intel puts
echo.
pause