@echo off
echo ========================================
echo RESTARTING OPTIONS BOT WITH FIXES
echo ========================================
echo.
echo Step 1: Killing old bot processes...
taskkill /F /IM python3.11.exe /FI "MEMUSAGE gt 300000"
timeout /t 3 /nobreak >nul
echo.
echo Step 2: Starting bot with fixed code...
cd /d "C:\Users\kkdo\PC-HIVE-TRADING"
echo Starting OPTIONS_BOT.py...
python OPTIONS_BOT.py
