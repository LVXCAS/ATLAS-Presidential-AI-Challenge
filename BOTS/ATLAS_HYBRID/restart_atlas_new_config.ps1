# ATLAS Restart Script - Activates New 25/50 Pip Configuration
# This script kills all old instances and starts fresh ones with updated parameters

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ATLAS RESTART - NEW 60% WR CONFIG" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "New Configuration:" -ForegroundColor Yellow
Write-Host "  Stop Loss: 25 pips (was 14)" -ForegroundColor Green
Write-Host "  Take Profit: 50 pips (was 21)" -ForegroundColor Green
Write-Host "  Score Threshold: 1.5 (high conviction)" -ForegroundColor Green
Write-Host "  Trading Pairs: EUR_USD + USD_JPY only" -ForegroundColor Green
Write-Host ""

# Step 1: Kill all Python processes
Write-Host "[1/4] Killing all old ATLAS instances..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $pythonProcesses | Stop-Process -Force
    Write-Host "  Killed $($pythonProcesses.Count) Python processes" -ForegroundColor Green
    Start-Sleep -Seconds 3
} else {
    Write-Host "  No Python processes found" -ForegroundColor Gray
}

# Step 2: Verify clean slate
Write-Host "[2/4] Verifying clean environment..." -ForegroundColor Yellow
$remainingProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($remainingProcesses) {
    Write-Host "  WARNING: $($remainingProcesses.Count) Python processes still running" -ForegroundColor Red
} else {
    Write-Host "  All clear - ready to start fresh instances" -ForegroundColor Green
}
Write-Host ""

# Step 3: Start 3 fresh ATLAS instances
Write-Host "[3/4] Starting 3 ATLAS instances with NEW configuration..." -ForegroundColor Yellow
Set-Location "C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID"

# Start instance 1
Write-Host "  Starting Instance #1..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID'; python run_paper_training.py --phase exploration" -WindowStyle Minimized

Start-Sleep -Seconds 5

# Start instance 2
Write-Host "  Starting Instance #2..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID'; python run_paper_training.py --phase exploration" -WindowStyle Minimized

Start-Sleep -Seconds 5

# Start instance 3
Write-Host "  Starting Instance #3..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID'; python run_paper_training.py --phase exploration" -WindowStyle Minimized

Write-Host "  All 3 instances started!" -ForegroundColor Green
Start-Sleep -Seconds 10

# Step 4: Verify instances are running
Write-Host "[4/4] Verifying instances are running..." -ForegroundColor Yellow
$newProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($newProcesses) {
    Write-Host "  $($newProcesses.Count) Python processes active" -ForegroundColor Green
} else {
    Write-Host "  ERROR: No Python processes running!" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ATLAS RESTART COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Trade Will Use:" -ForegroundColor Yellow
Write-Host "  Win: +`$12,500 (50 pips x 25 lots)" -ForegroundColor Green
Write-Host "  Loss: -`$6,250 (25 pips x 25 lots)" -ForegroundColor Green
Write-Host "  Risk:Reward: 1:2.0" -ForegroundColor Green
Write-Host ""
Write-Host "Check logs in minimized windows to monitor trades" -ForegroundColor Cyan
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
