@echo off
REM ========================================================================
REM TEST_SETUP.bat - Verify Automated Startup Configuration
REM ========================================================================
REM This script tests that everything is configured correctly
REM ========================================================================

color 0D
title PC-HIVE - Setup Verification Test

cd /d "%~dp0"

echo.
echo ========================================================================
echo   PC-HIVE AUTO SCANNER - SETUP VERIFICATION
echo ========================================================================
echo.
echo This will verify your automated startup is configured correctly.
echo.
pause
echo.

set ERRORS=0
set WARNINGS=0

echo ========================================================================
echo   TEST 1: Python Installation
echo ========================================================================
echo.

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Python is installed
    python --version
) else (
    echo [FAIL] Python not found in PATH
    set /A ERRORS+=1
)
echo.

echo ========================================================================
echo   TEST 2: Required Python Packages
echo ========================================================================
echo.

python -c "import schedule" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] schedule package installed
) else (
    echo [FAIL] schedule package missing
    echo        Fix: pip install schedule
    set /A ERRORS+=1
)

python -c "import pandas" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] pandas package installed
) else (
    echo [FAIL] pandas package missing
    echo        Fix: pip install pandas
    set /A ERRORS+=1
)

python -c "import alpaca_trade_api" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] alpaca_trade_api package installed
) else (
    echo [WARN] alpaca_trade_api package missing
    echo        Fix: pip install alpaca-trade-api
    set /A WARNINGS+=1
)
echo.

echo ========================================================================
echo   TEST 3: Required Files
echo ========================================================================
echo.

if exist "auto_options_scanner.py" (
    echo [PASS] auto_options_scanner.py found
) else (
    echo [FAIL] auto_options_scanner.py not found
    set /A ERRORS+=1
)

if exist "MONDAY_AI_TRADING.py" (
    echo [PASS] MONDAY_AI_TRADING.py found
) else (
    echo [FAIL] MONDAY_AI_TRADING.py not found
    set /A ERRORS+=1
)

if exist "START_TRADING.bat" (
    echo [PASS] START_TRADING.bat found
) else (
    echo [FAIL] START_TRADING.bat not found
    set /A ERRORS+=1
)

if exist "START_TRADING_BACKGROUND.bat" (
    echo [PASS] START_TRADING_BACKGROUND.bat found
) else (
    echo [FAIL] START_TRADING_BACKGROUND.bat not found
    set /A ERRORS+=1
)
echo.

echo ========================================================================
echo   TEST 4: Scheduled Tasks
echo ========================================================================
echo.

schtasks /Query /TN "PC-HIVE Auto Scanner" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Daily scheduled task exists
    schtasks /Query /TN "PC-HIVE Auto Scanner" /V /FO LIST | find /I "Task To Run"
    schtasks /Query /TN "PC-HIVE Auto Scanner" /V /FO LIST | find /I "Status"
) else (
    echo [WARN] Scheduled task not found
    echo        Fix: Run SETUP_AUTOMATED_STARTUP.bat as administrator
    set /A WARNINGS+=1
)
echo.

schtasks /Query /TN "PC-HIVE Auto Scanner - Startup" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Startup scheduled task exists
) else (
    echo [WARN] Startup task not found
    echo        Fix: Run SETUP_AUTOMATED_STARTUP.bat as administrator
    set /A WARNINGS+=1
)
echo.

echo ========================================================================
echo   TEST 5: Logs Directory
echo ========================================================================
echo.

if exist "logs\" (
    echo [PASS] logs directory exists
    dir /B logs\*.log 2>nul | find /C /V "" > nul
    if %ERRORLEVEL% EQU 0 (
        echo [INFO] Log files found:
        dir /B logs\*.log 2>nul | findstr /N "^" | findstr /R "^[1-3]:"
    ) else (
        echo [INFO] No log files yet (scanner hasn't run)
    )
) else (
    echo [INFO] logs directory will be created on first run
)
echo.

echo ========================================================================
echo   TEST 6: Scanner Configuration
echo ========================================================================
echo.

if exist "auto_scanner_status.json" (
    echo [PASS] Status file exists
    type auto_scanner_status.json
) else (
    echo [INFO] Status file will be created on first run
)
echo.

echo ========================================================================
echo   TEST 7: Dry Run Test
echo ========================================================================
echo.

echo Testing scanner startup (--once mode)...
echo This will take 10-30 seconds...
echo.

timeout /t 3 /nobreak > nul

python auto_options_scanner.py --once >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Scanner runs without errors
) else (
    echo [WARN] Scanner encountered an error
    echo        Check logs for details
    set /A WARNINGS+=1
)
echo.

echo ========================================================================
echo   TEST RESULTS
echo ========================================================================
echo.

if %ERRORS% EQU 0 (
    if %WARNINGS% EQU 0 (
        echo [SUCCESS] All tests passed!
        echo.
        echo Your automated scanner is ready to go!
        echo.
        echo Next steps:
        echo   1. Run SETUP_AUTOMATED_STARTUP.bat if tasks not scheduled
        echo   2. Scanner will run automatically at 6:30 AM PT daily
        echo   3. Use CHECK_SCANNER_STATUS.bat to verify it's running
        echo.
    ) else (
        echo [PARTIAL] Tests passed with %WARNINGS% warning(s)
        echo.
        echo Scanner will work, but some optional features may be missing.
        echo Review warnings above and fix if needed.
        echo.
    )
) else (
    echo [FAILED] %ERRORS% critical error(s) found
    echo.
    echo Please fix the errors above before running the scanner.
    echo.
)

echo ========================================================================
echo.

pause
