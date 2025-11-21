@echo off
echo STARTING AUTONOMOUS TRADING SYSTEM
echo ====================================
echo System will execute trades automatically at 6:30 AM PT
echo Market Open: Monday-Friday 6:30 AM PT
echo Monitoring: Daily 10:00 AM PT
echo Review: Daily 2:00 PM PT
echo ====================================

echo Checking Python environment...
python --version

echo Starting autonomous trading system...
echo Check autonomous_trading.log for real-time updates
echo Press Ctrl+C to stop the system

python autonomous_market_open_system.py

pause