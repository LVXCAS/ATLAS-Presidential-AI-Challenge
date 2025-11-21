@echo off
cd /d "c:\Users\lucas\PC-HIVE-TRADING"
start "Forex Bot" pythonw WORKING_FOREX_OANDA.py
echo Forex bot started in background!
timeout /t 2
