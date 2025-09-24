@echo off
echo.
echo 🚀🚀🚀 HIVE TRADING EMPIRE - LEAN SETUP 🚀🚀🚀
echo ==============================================
echo.
echo This will setup LEAN to run your 353-file trading system
echo.
echo What this does:
echo   ✅ Install LEAN engine locally
echo   ✅ Install 46+ quantitative libraries  
echo   ✅ Create configuration files
echo   ✅ Setup launchers for backtest/paper/live
echo   ✅ Test everything works
echo.
echo After setup, you can:
echo   📊 Test strategies: python lean_runner.py backtest
echo   📝 Paper trade: python lean_runner.py paper
echo   💰 Live trade: python lean_runner.py live
echo.
echo This is SAFE - no real money until you explicitly go live.
echo.
pause
echo.
echo ⚡ STARTING LEAN SETUP...
python lean_local_setup.py

echo.
echo 🎯 SETUP COMPLETE! 
echo.
echo Next steps:
echo   1. Edit lean_config_paper_alpaca.json with your Alpaca API keys
echo   2. Run: python lean_runner.py backtest
echo   3. Run: python lean_runner.py paper  
echo   4. After success: python lean_runner.py live
echo.
pause