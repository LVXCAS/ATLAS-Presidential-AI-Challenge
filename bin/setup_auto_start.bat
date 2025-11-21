@echo off
echo Setting up HiveTrading Auto-Start...
echo.

REM Create scheduled task to start Market Hunter automatically
schtasks /create /tn "HiveTrading Market Hunter" /tr "\"C:\Users\kkdo\HiveTrading-2.0\AUTO-START-MARKET-HUNTER.bat\"" /sc onstart /ru "%USERNAME%" /f

if %errorlevel% equ 0 (
    echo ✓ Auto-start task created successfully!
    echo.
    echo The Market Hunter will now start automatically when Windows boots.
    echo.
    echo To disable auto-start:
    echo schtasks /delete /tn "HiveTrading Market Hunter" /f
    echo.
) else (
    echo ❌ Failed to create auto-start task.
    echo You may need to run this as Administrator.
    echo.
)

pause