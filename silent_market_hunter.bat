@echo off
REM Silent Market Hunter - Runs in background without window
cd "C:\Users\kkdo\HiveTrading-2.0"

REM Start Python script minimized
start /min "HiveTrading Market Hunter" python start_real_market_hunter.py

REM Exit batch file immediately 
exit