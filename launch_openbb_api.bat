@echo off
REM ============================================================
REM OpenBB Platform API Launcher
REM ============================================================
REM Launches OpenBB Platform API with web interface
REM Access at: http://127.0.0.1:6900/docs
REM ============================================================

echo.
echo ====================================================================
echo LAUNCHING OPENBB PLATFORM API
echo ====================================================================
echo.
echo Web Interface URLs:
echo   - Swagger UI:  http://127.0.0.1:6900/docs
echo   - ReDoc:       http://127.0.0.1:6900/redoc
echo   - Workspace:   https://my.openbb.co/app/platform
echo.
echo Authentication: DISABLED (local access only)
echo.
echo Press CTRL+C to stop the server
echo ====================================================================
echo.

REM Launch OpenBB API on port 6900
openbb-api --host 127.0.0.1 --port 6900

pause
