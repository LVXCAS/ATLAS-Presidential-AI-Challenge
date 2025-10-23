@echo off
REM ========================================
REM FOREX AUTO-TRADER - QUICK START
REM Complete setup and launch in 5 minutes
REM ========================================

echo.
echo ======================================================================
echo FOREX AUTO-TRADER - QUICK START SETUP
echo ======================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Python found
python --version

REM Install dependencies
echo.
echo [2/6] Installing dependencies...
echo This may take a few minutes...
pip install v20 pandas numpy python-dotenv >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo Dependencies installed successfully
) else (
    echo [WARNING] Some dependencies may have failed to install
    echo Try running: pip install v20 pandas numpy python-dotenv
)

REM Create directories
echo.
echo [3/6] Creating directories...
if not exist forex_trades mkdir forex_trades
if not exist logs mkdir logs
echo Directories created

REM Run test suite
echo.
echo [4/6] Running system tests...
echo This will verify all components are working...
echo.
python test_forex_system.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some tests failed
    echo Review errors above before trading
    echo.
    pause
)

REM Check for OANDA credentials
echo.
echo [5/6] Checking OANDA API configuration...
echo.

if exist .env (
    findstr "OANDA_API_KEY" .env >nul
    if %ERRORLEVEL% EQU 0 (
        echo OANDA credentials found in .env file
    ) else (
        goto setup_credentials
    )
) else (
    goto setup_credentials
)

goto launch_trader

:setup_credentials
echo.
echo ======================================================================
echo OANDA API SETUP REQUIRED
echo ======================================================================
echo.
echo You need OANDA API credentials to trade forex.
echo.
echo STEP 1: Create FREE Practice Account
echo   1. Go to: https://www.oanda.com/us-en/trading/
echo   2. Click "Sign Up" - it's FREE and instant
echo   3. No credit card required for practice account
echo.
echo STEP 2: Get API Credentials
echo   1. Log in to OANDA
echo   2. Go to "Manage API Access"
echo   3. Generate API Token
echo   4. Copy your API Key and Account ID
echo.
echo STEP 3: Configure System
echo   1. Edit config\forex_config.json
echo   2. Update these fields:
echo      - account_id: "YOUR_OANDA_ACCOUNT_ID"
echo      - api_key: "YOUR_OANDA_API_KEY"
echo.
echo   OR create .env file with:
echo      OANDA_API_KEY=your_key_here
echo      OANDA_ACCOUNT_ID=your_account_id
echo.
echo ======================================================================
echo.

set /p continue="Press ENTER to open OANDA website, or Ctrl+C to exit..."
start https://www.oanda.com/us-en/trading/

echo.
echo After getting your credentials:
echo   1. Edit config\forex_config.json or create .env file
echo   2. Run this script again
echo.
pause
exit /b 0

:launch_trader
echo.
echo [6/6] Launching Forex Auto-Trader...
echo.
echo ======================================================================
echo FOREX AUTO-TRADER STARTING
echo ======================================================================
echo.
echo Mode: PAPER TRADING (Safe - No Real Money)
echo Pairs: EUR/USD, USD/JPY
echo Timeframe: 1 Hour
echo.
echo The system will:
echo   - Scan for signals every hour
echo   - Execute trades automatically (paper trading)
echo   - Manage positions with stop loss and take profit
echo   - Log all activity to forex_trades/ and logs/
echo.
echo Press Ctrl+C to stop at any time
echo.
echo ======================================================================
echo.

timeout /t 3

REM Start trader
python forex_auto_trader.py

echo.
echo ======================================================================
echo TRADER STOPPED
echo ======================================================================
echo.
echo Check logs in:
echo   - forex_trades\execution_log_YYYYMMDD.json (trade history)
echo   - logs\forex_trader_YYYYMMDD.log (system logs)
echo.
echo To restart: START_FOREX_TRADER.bat
echo To check status: CHECK_FOREX_STATUS.bat
echo Emergency stop: STOP_FOREX_TRADER.bat
echo.

pause
