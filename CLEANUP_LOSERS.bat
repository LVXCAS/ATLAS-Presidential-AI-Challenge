@echo off
title EMERGENCY CLEANUP - Close Losing Positions
color 0C

echo ================================================================================
echo EMERGENCY CLEANUP - CLOSE LOSING POSITIONS
echo ================================================================================
echo.
echo This script will close:
echo   1. All worthless options (current price = $0)
echo   2. All positions with losses greater than 30%%
echo   3. All positions with losses greater than $500
echo.
echo WARNING: This is IRREVERSIBLE and will realize losses!
echo.
pause

echo.
echo Starting cleanup...
echo.

python auto_cleanup_market_open.py

echo.
echo ================================================================================
echo CLEANUP COMPLETE
echo ================================================================================
echo.
echo Check the generated JSON report for details.
echo.
pause
