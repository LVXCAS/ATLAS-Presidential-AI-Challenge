@echo off
title HIVE TRADE - Smart Mean Reversion Trading
color 0E
cls
echo.
echo ██╗  ██╗██╗██╗   ██╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗
echo ██║  ██║██║██║   ██║██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  
echo ███████║██║██║   ██║█████╗         ██║   ██████╔╝███████║██║  ██║█████╗    
echo ██╔══██║██║╚██╗ ██╔╝██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    
echo ██║  ██║██║ ╚████╔╝ ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗  
echo ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  
echo.
echo                    SMART AUTOMATED TRADING SYSTEM
echo                    Mean Reversion + Technical Analysis
echo.
echo ================================================================================
echo.
echo INTELLIGENT TRADING STRATEGIES NOW ACTIVE:
echo.
echo [✓] MEAN REVERSION STRATEGY
echo     • Z-Score Analysis    - Statistical deviation from price mean
echo     • RSI Indicators      - Oversold/Overbought detection  
echo     • Bollinger Bands     - Support/Resistance levels
echo     • Moving Averages     - Trend confirmation
echo.
echo [✓] ENSEMBLE DECISION ENGINE
echo     • Multiple Signal Confirmation - Requires agreement between indicators
echo     • Confidence-Based Position Sizing - Larger trades for stronger signals
echo     • Risk Management - Only trades signals above 50%% confidence
echo.
echo [✓] REAL MARKET ANALYSIS
echo     • Mean-Reverting Price Simulation - Realistic price movements
echo     • Technical Indicator Calculations - RSI, Bollinger Bands, Z-Score
echo     • 50-Period Price History - Sufficient data for accurate analysis
echo.
echo TRADING PARAMETERS:
echo • Monitored Assets: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, SPY, QQQ
echo • Analysis Frequency: Every 2 minutes during market hours
echo • Trading Hours: 9:00 AM - 4:00 PM
echo • Minimum Confidence: 50%% for trade execution
echo • Position Sizing: 1-5 shares based on signal strength
echo.
echo SAFETY FEATURES:
echo • Paper Trading Mode - NO REAL MONEY AT RISK
echo • Alpaca Paper Account - Virtual trading only
echo • Complete Trade Logging - All decisions recorded
echo • Position Tracking - Monitor all holdings
echo.
echo ================================================================================
echo.
echo This system will automatically:
echo 1. Analyze market data using advanced mean reversion algorithms
echo 2. Identify oversold/overbought conditions for optimal entry/exit
echo 3. Execute paper trades through your Alpaca account when confidence ^> 50%%
echo 4. Log all trading decisions with detailed explanations
echo 5. Track positions and performance in real-time
echo.
echo NO RANDOM TRADES - ALL DECISIONS BASED ON TECHNICAL ANALYSIS
echo.
set /p confirm="Start intelligent mean reversion trading? (y/n): "
if /i not "%confirm%"=="y" (
    echo.
    echo Trading cancelled. System remains in standby mode.
    pause
    exit /b 0
)
echo.
echo ================================================================================
echo STARTING INTELLIGENT TRADING SYSTEM...
echo ================================================================================
echo.
echo • Connecting to Alpaca Paper Trading Account...
echo • Initializing Mean Reversion Analysis Engine...
echo • Building Technical Indicator Database...
echo • Starting Automated Trading Loop...
echo.
echo Press Ctrl+C to stop trading at any time
echo All activity will be logged to: logs\smart_trades.log
echo.
python start_smart_trading.py
echo.
echo ================================================================================
echo TRADING SESSION ENDED
echo ================================================================================
echo.
echo Check logs\smart_trades.log for detailed trading history
echo.
pause