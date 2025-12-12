@echo off
echo ========================================
echo Starting ATLAS with ALL FIXES
echo ========================================
echo.
echo Fixes included:
echo 1. RSI Exhaustion Filter (blocks RSI >70 LONG, <30 SHORT)
echo 2. Adapter returns [] not None
echo 3. TechnicalAgent has veto authority
echo.
echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul
echo.
echo Starting ATLAS in exploration mode...
python run_paper_training.py --phase exploration
