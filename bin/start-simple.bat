@echo off
echo ================================================
echo    HIVE TRADE - SIMPLE DEPLOYMENT
echo    Getting Infrastructure Ready for Trading
echo ================================================

echo.
echo [INFO] Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo [INFO] Waiting for Docker to start...
timeout /t 30 /nobreak > nul

echo [INFO] Starting infrastructure services...
docker-compose -f docker-compose.simple.yml up -d

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo    INFRASTRUCTURE READY!
    echo ================================================
    echo.
    echo Services started:
    echo • Database: localhost:5432
    echo • Redis: localhost:6379
    echo • Prometheus: http://localhost:9090
    echo • Grafana: http://localhost:3000 (admin/admin123)
    echo.
    echo Next steps:
    echo 1. Wait 30 seconds for services to initialize
    echo 2. Run backend: cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
    echo 3. Run frontend: cd frontend && npm start
    echo.
    echo To stop: docker-compose -f docker-compose.simple.yml down
    echo.
) else (
    echo [ERROR] Failed to start services
    echo Please make sure Docker Desktop is running
)

pause