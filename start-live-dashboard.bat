@echo off
title HIVE TRADE - Live Dashboard Launcher

echo.
echo  ██╗  ██╗██╗██╗   ██╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗
echo  ██║  ██║██║██║   ██║██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  
echo  ███████║██║██║   ██║█████╗         ██║   ██████╔╝███████║██║  ██║█████╗    
echo  ██╔══██║██║╚██╗ ██╔╝██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    
echo  ██║  ██║██║ ╚████╔╝ ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗  
echo  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  
echo.
echo                    LIVE BLOOMBERG TERMINAL DASHBOARD
echo                           Real-time Trading Interface
echo.
echo ================================================================================
echo.

echo 🚀 Starting Hive Trade Live Dashboard System...
echo.

echo Step 1: Starting Backend API Server...
start "Hive Trade Backend" cmd /k "cd backend && python main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo Step 2: Backend started at http://localhost:8001
echo         - Dashboard API: http://localhost:8001/api/dashboard/live-feed
echo         - Health Check: http://localhost:8001/health
echo         - API Docs: http://localhost:8001/docs
echo.

echo Step 3: You can now:
echo         - Access the live dashboard data at: http://localhost:8001/api/dashboard/live-feed
echo         - View the React dashboard component at: frontend/src/components/LiveTradingDashboard.tsx
echo         - Integrate with your existing frontend setup
echo.

echo 📊 DASHBOARD FEATURES:
echo    ✅ Real-time Portfolio Tracking
echo    ✅ Live P&L Updates
echo    ✅ AI Agent Signals
echo    ✅ Market Data Feeds
echo    ✅ Risk Management Metrics
echo    ✅ Bloomberg Terminal Style UI
echo.

echo 🔗 API ENDPOINTS AVAILABLE:
echo    - GET /api/dashboard/portfolio    (Portfolio data)
echo    - GET /api/dashboard/positions    (Current positions)
echo    - GET /api/dashboard/signals      (AI agent signals)
echo    - GET /api/dashboard/market       (Market data)
echo    - GET /api/dashboard/stats        (System stats)
echo    - GET /api/dashboard/risk         (Risk metrics)
echo    - GET /api/dashboard/live-feed    (Complete live feed)
echo.

echo Backend is running! Press any key to open API documentation...
pause >nul

start http://localhost:8001/docs

echo.
echo 💡 NEXT STEPS:
echo    1. The backend is providing live data simulation
echo    2. Use the LiveTradingDashboard.tsx component in your React app
echo    3. The dashboard will automatically connect to the backend
echo    4. If backend is not available, it falls back to client-side simulation
echo.
echo ✨ Your Bloomberg Terminal dashboard is ready!
echo.
pause