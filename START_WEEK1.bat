@echo off
REM ===================================================================
REM WEEK 1 AUTONOMOUS TRADING SYSTEM - COMPLETE LAUNCHER
REM ===================================================================
REM Perfect Paper Month - Week 1: Sep 30 - Oct 4, 2025
REM Target: 5-8% weekly ROI with flawless execution
REM ===================================================================

echo.
echo ========================================
echo WEEK 1 TRADING SYSTEM - STARTING
echo ========================================
echo.
echo Week 1 Configuration:
echo - Threshold: 4.0+ confidence (80%%+)
echo - Max Trades: 2 per day
echo - Max Risk: 1.5%% per position
echo - Target: 1.0-1.5%% daily ROI
echo.
echo Systems Launching:
echo [1/2] Continuous Week 1 Scanner (every 5 min)
echo [2/2] Continuous R&D Discovery (every 6 hours)
echo.
echo ========================================
echo.

REM Launch Week 1 Scanner in background
echo Starting Week 1 Scanner...
start "Week1-Scanner" python continuous_week1_scanner.py

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Launch Continuous R&D Discovery
echo Starting R&D Discovery System...
start "R&D-Discovery" python continuous_rd_discovery.py

echo.
echo ========================================
echo WEEK 1 SYSTEM LAUNCHED SUCCESSFULLY
echo ========================================
echo.
echo Running Systems:
echo - Week 1 Scanner: Scanning every 5 minutes
echo - R&D Discovery: New strategies every 6 hours
echo.
echo To monitor: Check console windows
echo To stop: Close the console windows
echo.
echo Good luck trading! Target: 5-8%% weekly ROI
echo ========================================
echo.

pause
