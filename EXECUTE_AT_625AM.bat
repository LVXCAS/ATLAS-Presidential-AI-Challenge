@echo off
echo AUTONOMOUS TRADING SYSTEM - MARKET OPEN EXECUTION
echo Current Time: %time%
echo Market Opens: 6:30 AM PT
echo ================================================

echo Launching autonomous execution system...
python LAUNCH_NOW_630AM.py

echo Checking if we should also run full autonomous system...
python autonomous_market_open_system.py

pause