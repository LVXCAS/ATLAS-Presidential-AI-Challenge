@echo off
echo ===============================================================
echo E8 ULTRA-CONSERVATIVE BOT - QUICK START
echo ===============================================================
echo.

REM Check if .env exists
if not exist .env (
    echo [ERROR] .env file not found!
    echo.
    echo Please create .env file with:
    echo   E8_ACCOUNT=your_account_number
    echo   E8_PASSWORD=your_password
    echo   E8_SERVER=match-trader-demo
    echo.
    pause
    exit /b 1
)

REM Check if credentials exist
findstr /C:"E8_ACCOUNT" .env >nul 2>&1
if errorlevel 1 (
    echo [ERROR] E8_ACCOUNT not found in .env!
    echo.
    echo Add your Match Trader credentials to .env:
    echo   E8_ACCOUNT=your_account_number
    echo   E8_PASSWORD=your_password
    echo   E8_SERVER=match-trader-demo
    echo.
    pause
    exit /b 1
)

echo [STEP 1/3] Testing connection to Match Trader...
python test_match_trader_connection.py
if errorlevel 1 (
    echo.
    echo [ERROR] Connection test failed. Check credentials in .env
    pause
    exit /b 1
)

echo.
echo [STEP 2/3] Starting ultra-conservative bot...
echo.
echo Bot will:
echo   - Scan every hour
echo   - Look for PERFECT setups (score 6.0+)
echo   - Trade 0-2 times per WEEK (not per day!)
echo   - Most scans: ZERO opportunities (expected!)
echo.
echo Press any key to start bot in background...
pause >nul

cd BOTS
start pythonw E8_ULTRA_CONSERVATIVE_BOT.py
cd ..

timeout /t 3 >nul

echo.
echo [STEP 3/3] Verifying bot is running...
tasklist | findstr "pythonw.exe" >nul
if errorlevel 1 (
    echo [WARN] Bot may not be running. Check for errors.
) else (
    echo [SUCCESS] Bot is running in background!
)

echo.
echo ===============================================================
echo BOT STARTED - MATCH TRADER DEMO VALIDATION
echo ===============================================================
echo.
echo What to expect:
echo   - First scan in ~1 minute
echo   - 0-2 trades per WEEK
echo   - Most days: ZERO trades (normal!)
echo   - Goal: ZERO daily DD violations over 60 days
echo.
echo Monitor with:
echo   python BOTS/demo_validator.py report    (daily check)
echo   python BOTS/demo_validator.py weekly    (weekly summary)
echo.
echo Stop bot:
echo   taskkill /F /IM pythonw.exe
echo.
echo ===============================================================
echo.
pause
