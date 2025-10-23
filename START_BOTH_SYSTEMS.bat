@echo off
REM ========================================================================
REM START_BOTH_SYSTEMS.bat - Start Forex + Options Trading
REM ========================================================================

color 0A
title PC-HIVE Trading Empire - Starting Both Systems

echo.
echo ========================================================================
echo   PC-HIVE TRADING EMPIRE
echo   Starting Forex Elite + Options Scanner
echo ========================================================================
echo.

cd /d "%~dp0"

REM Start Forex Elite in background
echo [1/2] Starting Forex Elite (Strict Strategy)...
start "FOREX ELITE" /MIN python START_FOREX_ELITE.py --strategy strict
timeout /t 5 /nobreak > nul

REM Start Options Scanner in background
echo [2/2] Starting Options Scanner (Paper Mode)...
start "OPTIONS SCANNER" /MIN python auto_options_scanner.py --daily
timeout /t 3 /nobreak > nul

echo.
echo ========================================================================
echo   BOTH SYSTEMS STARTED
echo ========================================================================
echo.
echo [OK] Forex Elite:     Running (check forex_elite.log)
echo [OK] Options Scanner:  Running (check scanner_output.log)
echo.
echo Monitor positions: python monitor_positions.py --watch
echo Stop all systems:  EMERGENCY_STOP.bat
echo.
echo Press any key to close this window...
pause > nul
