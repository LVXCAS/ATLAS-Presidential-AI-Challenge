@echo off
echo ===================================
echo NUCLEAR CLEANUP - KILLING ALL
echo ===================================
echo.
echo Step 1: Killing ALL Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo Step 2: Killing ALL pythonw processes...
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo Step 3: Checking what's left...
powershell -Command "Get-Process python* -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, StartTime -AutoSize"

echo.
echo DONE! All Python processes killed.
echo.
echo Now restart only the systems you need:
echo   - WORKING_FOREX_OANDA.py (for FOREX)
echo   - OPTIONS_STYLE_STOCK_TRADER.py (for stocks)
echo.
pause
