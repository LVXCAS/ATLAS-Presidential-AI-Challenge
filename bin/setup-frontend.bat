@echo off
echo ================================================
echo    HIVE TRADE - FRONTEND SETUP
echo    Setting up React Environment
echo ================================================

cd frontend

echo [INFO] Installing Node.js dependencies...
npm install

echo [INFO] Creating environment file...
echo REACT_APP_API_URL=http://localhost:8001 > .env
echo REACT_APP_WS_URL=ws://localhost:8001 >> .env
echo GENERATE_SOURCEMAP=false >> .env
echo FAST_REFRESH=true >> .env

echo.
echo ================================================
echo    FRONTEND SETUP COMPLETE!
echo ================================================
echo.
echo Dependencies installed in: frontend/node_modules
echo Configuration file: frontend/.env
echo.
echo To start frontend:
echo 1. cd frontend
echo 2. npm start
echo.
echo The React app will open at: http://localhost:3000
echo.

pause