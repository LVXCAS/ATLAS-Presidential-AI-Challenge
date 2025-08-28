@echo off
title HIVE TRADE - Auto Trading System
echo.
echo HIVE TRADE - AUTOMATED PAPER TRADING SYSTEM
echo ============================================
echo.
echo This will start automatic trading using your Alpaca paper account
echo All trades are simulated - NO REAL MONEY IS AT RISK
echo.
echo The system will:
echo - Monitor market during trading hours (9 AM - 4 PM)
echo - Generate random trading signals
echo - Execute paper trades through Alpaca API
echo - Log all activity to logs/paper_trades.log
echo.
echo Press Ctrl+C to stop trading at any time
echo.
pause
echo.
echo Starting automated trading...
python start_simple_trading.py
pause