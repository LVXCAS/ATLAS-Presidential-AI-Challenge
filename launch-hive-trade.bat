@echo off
title HIVE TRADE - Bloomberg Terminal Launcher

echo.
echo  ██╗  ██╗██╗██╗   ██╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗
echo  ██║  ██║██║██║   ██║██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  
echo  ███████║██║██║   ██║█████╗         ██║   ██████╔╝███████║██║  ██║█████╗    
echo  ██╔══██║██║╚██╗ ██╔╝██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    
echo  ██║  ██║██║ ╚████╔╝ ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗  
echo  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  
echo.
echo                    BLOOMBERG TERMINAL STYLE TRADING SYSTEM
echo                           Professional Grade - Ready to Trade
echo.
echo ================================================================================
echo.

:MAIN_MENU
echo [1] Quick Start - Launch Infrastructure Only
echo [2] Full Setup - Backend + Frontend + Infrastructure  
echo [3] Training Mode - Run Intensive Backtesting and Training
echo [4] Production Deploy - Full Docker Stack
echo [5] Status Check - View System Health
echo [6] Exit
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto QUICK_START
if "%choice%"=="2" goto FULL_SETUP
if "%choice%"=="3" goto TRAINING_MODE  
if "%choice%"=="4" goto PRODUCTION_DEPLOY
if "%choice%"=="5" goto STATUS_CHECK
if "%choice%"=="6" goto EXIT
goto MAIN_MENU

:QUICK_START
echo.
echo 🚀 QUICK START - Infrastructure Only
echo ====================================
echo.
echo Starting Docker services...
call start-simple.bat
goto MAIN_MENU

:FULL_SETUP
echo.
echo 🏗️  FULL SETUP - Complete Development Environment
echo ===============================================
echo.
echo Step 1: Setting up infrastructure...
call start-simple.bat

echo.
echo Step 2: Setting up backend environment...
call setup-backend.bat

echo.
echo Step 3: Setting up frontend environment...
call setup-frontend.bat

echo.
echo ✅ FULL SETUP COMPLETE!
echo.
echo 🎯 NEXT STEPS:
echo 1. Edit backend\.env and add your Alpaca API keys
echo 2. Start backend: 
echo    cd backend 
echo    venv\Scripts\activate.bat
echo    python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
echo.
echo 3. Start frontend (new terminal):
echo    cd frontend
echo    npm start
echo.
echo 4. Access Bloomberg Terminal: http://localhost:3000
echo.
pause
goto MAIN_MENU

:TRAINING_MODE
echo.
echo 🧠 TRAINING MODE - Intensive Backtesting and Training
echo ===================================================
echo.
echo This will run comprehensive training and backtesting:
echo • Multi-agent strategy optimization
echo • Historical backtesting on 10+ symbols
echo • Hyperparameter optimization (100+ runs)
echo • Portfolio optimization and ensembling
echo.
set /p confirm="Are you sure you want to start intensive training? (y/n): "
if /i "%confirm%"=="y" (
    echo.
    echo 🚀 Starting intensive training and backtesting...
    python run-training.py
    echo.
    echo 🎉 Training complete! Check training_results.json for results.
) else (
    echo Training cancelled.
)
echo.
pause
goto MAIN_MENU

:PRODUCTION_DEPLOY
echo.
echo 🏭 PRODUCTION DEPLOY - Full Docker Stack
echo =======================================
echo.
echo This will deploy the complete production system with:
echo • All 15+ microservices
echo • Comprehensive monitoring stack
echo • Automated backups
echo • SSL/HTTPS configuration
echo.
set /p confirm="Deploy production stack? (y/n): "
if /i "%confirm%"=="y" (
    echo.
    echo 🚀 Starting production deployment...
    bash -c "./deploy.sh"
) else (
    echo Production deployment cancelled.
)
echo.
pause
goto MAIN_MENU

:STATUS_CHECK
echo.
echo 📊 SYSTEM STATUS CHECK
echo =====================
echo.

echo Checking Docker services...
docker-compose -f docker-compose.simple.yml ps

echo.
echo Checking network connectivity...
echo • Database: 
netstat -an | findstr :5432 > nul && echo   ✅ PostgreSQL (5432) - Running || echo   ❌ PostgreSQL (5432) - Not running

echo • Redis:
netstat -an | findstr :6379 > nul && echo   ✅ Redis (6379) - Running || echo   ❌ Redis (6379) - Not running

echo • Prometheus:
netstat -an | findstr :9090 > nul && echo   ✅ Prometheus (9090) - Running || echo   ❌ Prometheus (9090) - Not running

echo • Grafana:
netstat -an | findstr :3000 > nul && echo   ✅ Grafana (3000) - Running || echo   ❌ Grafana (3000) - Not running

echo • Backend API:
netstat -an | findstr :8001 > nul && echo   ✅ Backend API (8001) - Running || echo   ❌ Backend API (8001) - Not running

echo • Frontend:
netstat -an | findstr :3000 > nul && echo   ✅ Frontend (3000) - Running || echo   ❌ Frontend (3000) - Not running

echo.
pause
goto MAIN_MENU

:EXIT
echo.
echo 👋 Thanks for using Hive Trade!
echo.
echo  "The best time to plant a tree was 20 years ago.
echo   The second best time is now. The third best time
echo   is when the market opens." - Warren Buffet (probably)
echo.
echo Happy trading! 💰📈
echo.
exit /b 0