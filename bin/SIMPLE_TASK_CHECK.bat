@echo off
echo.
echo Checking for PC-HIVE scheduled tasks...
echo.
schtasks /Query | findstr /I "HIVE"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] No PC-HIVE tasks found!
    echo.
    echo The setup script may not have run successfully.
    echo Please try running RUN_ME_AS_ADMIN.bat again.
    echo.
) else (
    echo.
    echo [OK] Tasks found above!
    echo.
)
pause
