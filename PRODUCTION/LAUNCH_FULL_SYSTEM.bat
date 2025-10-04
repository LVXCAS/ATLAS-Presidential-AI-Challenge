@echo off
echo ====================================================================
echo LAUNCHING FULL 3-TIER AUTONOMOUS TRADING EMPIRE
echo ====================================================================
echo.
echo System Configuration:
echo   TIER 1: Production R&D (yfinance + Alpaca) [ACTIVE]
echo   TIER 2: ML Strategy Generator (PyTorch, XGBoost) [ACTIVE]
echo   TIER 3: GPU Acceleration (GTX 1660 SUPER 6GB) [ACTIVE]
echo.
echo ====================================================================
echo.
echo Starting integrated system in continuous mode...
echo.
echo Operations:
echo   - Research cycles every 6 hours
echo   - ML auto-discovery enabled
echo   - GPU-accelerated backtesting ready
echo   - Scanner integration active
echo   - Performance feedback loops running
echo.
echo Press Ctrl+C to stop
echo ====================================================================
echo.

cd C:\Users\lucas\PC-HIVE-TRADING\PRODUCTION
python advanced_system_integrator.py --continuous

echo.
echo ====================================================================
echo System stopped
echo ====================================================================
pause
