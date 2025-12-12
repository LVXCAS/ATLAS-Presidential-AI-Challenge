# ATLAS FORCE RESTART - Load 25/50 Pip Configuration
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "ATLAS FORCE RESTART - 25/50 PIP STOPS + 1.5 THRESHOLD" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill ALL Python processes (force unload old cached code)
Write-Host "[1/3] Killing all Python processes to clear module cache..." -ForegroundColor White
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $count = $pythonProcesses.Count
    $pythonProcesses | Stop-Process -Force
    Write-Host "  ✓ Killed $count Python processes" -ForegroundColor Green
} else {
    Write-Host "  ✓ No Python processes running" -ForegroundColor Green
}

# Wait for processes to fully terminate
Start-Sleep -Seconds 2

# Step 2: Verify configuration files
Write-Host ""
Write-Host "[2/3] Verifying configuration..." -ForegroundColor White

# Check live_trader.py for 25 pips
$liveTraderContent = Get-Content "C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID\live_trader.py" -Raw
if ($liveTraderContent -match "stop_loss_pips = 25") {
    Write-Host "  ✓ live_trader.py has 25-pip stops" -ForegroundColor Green
} else {
    Write-Host "  ✗ WARNING: live_trader.py still has 14-pip stops!" -ForegroundColor Red
}

# Check hybrid_optimized.json for threshold 1.5
$configContent = Get-Content "C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID\config\hybrid_optimized.json" -Raw
if ($configContent -match '"score_threshold": 1.5') {
    Write-Host "  ✓ hybrid_optimized.json has threshold 1.5" -ForegroundColor Green
} else {
    Write-Host "  ⚠ hybrid_optimized.json may not have threshold 1.5" -ForegroundColor Yellow
}

# Step 3: Start FRESH ATLAS instances with exploration phase
Write-Host ""
Write-Host "[3/3] Starting 3 fresh ATLAS instances..." -ForegroundColor White
Write-Host "  Configuration: Exploration Phase (threshold 1.0, 25-pip stops)" -ForegroundColor Gray
Write-Host ""

# Start 3 instances
for ($i = 1; $i -le 3; $i++) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\lucas\PC-HIVE-TRADING\BOTS\ATLAS_HYBRID'; Write-Host 'ATLAS INSTANCE #$i - 25/50 PIP CONFIG' -ForegroundColor Cyan; python run_paper_training.py --phase exploration" -WindowStyle Minimized
    Write-Host "  ✓ Instance #$i started" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "RESTART COMPLETE!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "All instances should now use:" -ForegroundColor White
Write-Host "  • 25-pip stop loss (was 14 pips)" -ForegroundColor Yellow
Write-Host "  • 50-pip take profit (was 21 pips)" -ForegroundColor Yellow
Write-Host "  • 1:2 Risk:Reward ratio" -ForegroundColor Yellow
Write-Host "  • Threshold 1.0 (exploration phase)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Monitor logs for '[KELLY] SL: 25 pips' to verify!" -ForegroundColor Cyan
Write-Host ""
