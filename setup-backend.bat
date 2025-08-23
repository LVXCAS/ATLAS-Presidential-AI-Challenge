@echo off
echo ================================================
echo    HIVE TRADE - BACKEND SETUP
echo    Setting up Python Environment
echo ================================================

cd backend

echo [INFO] Creating virtual environment...
python -m venv venv

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing Python dependencies...
pip install --upgrade pip
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis pandas numpy scikit-learn
pip install alpaca-trade-api websockets prometheus-client aioredis asyncpg
pip install python-multipart python-jose[cryptography] passlib[bcrypt]

echo [INFO] Creating environment file...
echo # Hive Trade Backend Configuration > .env
echo DATABASE_URL=postgresql://hive_user:hive123@localhost:5432/hive_trading >> .env
echo REDIS_URL=redis://localhost:6379/0 >> .env
echo ALPACA_API_KEY=your_paper_trading_key_here >> .env
echo ALPACA_SECRET_KEY=your_paper_trading_secret_here >> .env
echo ALPACA_BASE_URL=https://paper-api.alpaca.markets >> .env
echo JWT_SECRET=your_jwt_secret_key_here_make_it_long >> .env
echo DEBUG=true >> .env
echo API_PORT=8001 >> .env

echo.
echo ================================================
echo    BACKEND SETUP COMPLETE!
echo ================================================
echo.
echo Environment created in: backend/venv
echo Configuration file: backend/.env
echo.
echo IMPORTANT: Edit backend/.env and add your Alpaca API keys!
echo.
echo To start backend:
echo 1. cd backend
echo 2. venv\Scripts\activate.bat
echo 3. python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
echo.

pause