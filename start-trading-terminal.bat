@echo off
cd "C:\Users\lucas\Hive Trade\trading-terminal"
start "Trading Terminal" cmd /k "npm run dev"
timeout /t 3 > nul
start "" "http://localhost:5173"