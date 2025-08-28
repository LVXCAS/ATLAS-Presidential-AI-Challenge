@echo off
title HIVE TRADE - Intelligent Auto Trading System
color 0A
echo.
echo  ██╗ ██╗██╗██╗   ██╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗
echo  ██║ ██║██║██║   ██║██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  
echo  ███████║██║██║   ██║█████╗         ██║   ██████╔╝███████║██║  ██║█████╗    
echo  ██╔══██║██║╚██╗ ██╔╝██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    
echo  ██║  ██║██║ ╚████╔╝ ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗  
echo  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  
echo.
echo                    INTELLIGENT AUTOMATED TRADING SYSTEM
echo                        Mean Reversion + Ensemble Strategies
echo.
echo ================================================================================
echo.
echo TRADING STRATEGIES ENABLED:
echo [✓] Mean Reversion Strategy
echo     • RSI (Relative Strength Index) - Detects oversold/overbought conditions
echo     • Bollinger Bands - Identifies price extremes for reversal opportunities  
echo     • Z-Score Analysis - Statistical measure of price deviation from mean
echo.
echo [✓] Ensemble Approach 
echo     • Momentum Strategy - Rides trends and price momentum
echo     • Multiple Signal Confirmation - Requires agreement between strategies
echo     • Confidence-Based Position Sizing - Larger trades for stronger signals
echo.
echo [✓] Risk Management
echo     • Paper Trading Mode - NO REAL MONEY AT RISK
echo     • Position Tracking - Monitors all open positions
echo     • Analysis-Based Decisions - No random trades
echo.
echo MONITORED SYMBOLS: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, SPY, QQQ
echo ANALYSIS FREQUENCY: Every 5 minutes during market hours (9 AM - 4 PM)
echo.
echo ================================================================================
echo.
echo This system will:
echo • Analyze market data using advanced technical indicators
echo • Generate intelligent buy/sell signals based on mean reversion
echo • Execute paper trades through your Alpaca account automatically
echo • Log all trades and reasoning to logs/intelligent_trades.log
echo • Trade only when confidence level exceeds 50%%
echo.
set /p confirm="Ready to start intelligent automated trading? (y/n): "
if /i not "%confirm%"=="y" (
    echo Trading cancelled by user.
    pause
    exit /b 1
)
echo.
echo Starting intelligent trading system...
echo Press Ctrl+C to stop trading at any time
echo.
python start_intelligent_trading.py
echo.
echo Intelligent trading system has stopped.
pause