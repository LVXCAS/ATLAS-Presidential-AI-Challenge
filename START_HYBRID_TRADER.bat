@echo off
REM ============================================================================
REM START HYBRID TA + AI MULTI-MARKET TRADER
REM ============================================================================

echo.
echo ================================================================================
echo           HYBRID TA + AI MULTI-MARKET TRADING SYSTEM
echo ================================================================================
echo.
echo Starting unified trader with AI confirmation layer...
echo.
echo Markets: FOREX (E8) + FUTURES (Apex) + CRYPTO (CFT)
echo AI: DeepSeek V3.1 + MiniMax (Free via OpenRouter)
echo Mode: TA-Lib PRIMARY, AI SECONDARY confirmation
echo.
echo Press Ctrl+C to stop (will export logs on shutdown)
echo ================================================================================
echo.

REM Check if .env exists
if not exist .env (
    echo [ERROR] .env file not found!
    echo Please create .env file with:
    echo   OANDA_API_KEY=your_key_here
    echo   OANDA_ACCOUNT_ID=your_account_id
    echo   OPENROUTER_API_KEY=your_openrouter_key
    echo.
    pause
    exit /b 1
)

REM Check if OPENROUTER_API_KEY is set
findstr /C:"OPENROUTER_API_KEY" .env >nul 2>&1
if errorlevel 1 (
    echo [WARN] OPENROUTER_API_KEY not found in .env
    echo AI confirmation will be DISABLED - running in TA-only mode
    echo.
    echo To enable AI, add to .env:
    echo   OPENROUTER_API_KEY=sk-or-v1-your-key-here
    echo.
    echo Get free API key at: https://openrouter.ai
    echo.
    pause
)

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Launch the hybrid trader
python MULTI_MARKET_TRADER.py

REM On shutdown, open logs folder
echo.
echo ================================================================================
echo Trader stopped. Opening logs folder...
echo ================================================================================
start logs

pause
