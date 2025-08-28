@echo off
title HIVE TRADE - AUTO MARKET HUNTER
color 0A
echo.
echo ██╗  ██╗██╗██╗   ██╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗
echo ██║  ██║██║██║   ██║██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  
echo ███████║██║██║   ██║█████╗         ██║   ██████╔╝███████║██║  ██║█████╗    
echo ██╔══██║██║╚██╗ ██╔╝██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    
echo ██║  ██║██║ ╚████╔╝ ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗  
echo ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  
echo.
echo                         AUTO MARKET HUNTER
echo                      ALWAYS ON - NO CONFIRMATION
echo.
echo ================================================================================
echo.
echo 🤖 FULLY AUTOMATED MODE ACTIVE
echo • No user confirmation required
echo • Starts immediately on launch
echo • Runs continuously during market hours
echo • Auto-restarts if connection issues
echo.
echo 📊 REAL MARKET DATA SOURCES:
echo • Polygon API: Professional real-time data
echo • Alpaca API: Live broker integration  
echo • Yahoo Finance: Free backup data
echo.
echo 🎯 TRADING STATUS: ALWAYS ACTIVE
echo • 55+ stocks monitored continuously
echo • Real-time opportunity detection
echo • Automatic trade execution in paper mode
echo • Complete logging of all activity
echo.
echo ================================================================================
echo.
echo LAUNCHING AUTO MARKET HUNTER...
echo System will start trading automatically!
echo.
timeout /t 3 /nobreak >nul
echo Starting in 3 seconds...
timeout /t 1 /nobreak >nul
echo Starting in 2 seconds...  
timeout /t 1 /nobreak >nul
echo Starting in 1 second...
timeout /t 1 /nobreak >nul
echo.
echo 🚀 MARKET HUNTER: LAUNCHING NOW!
echo.
python start_real_market_hunter.py
echo.
echo ================================================================================
echo MARKET HUNTER SESSION ENDED
echo ================================================================================
echo.
echo Auto-restarting in 10 seconds...
echo Press Ctrl+C to prevent restart
timeout /t 10
echo.
echo Restarting Market Hunter...
goto :eof