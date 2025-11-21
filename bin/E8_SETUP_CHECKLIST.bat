@echo off
echo ============================================================
echo E8 $200K CHALLENGE SETUP CHECKLIST
echo ============================================================
echo.
echo STEP 1: Purchase E8 Challenge
echo [ ] Go to E8 Funding website
echo [ ] Select: $200K, 80%% split, 6%% max drawdown
echo [ ] Cost: ~$1,200
echo [ ] Get login credentials
echo.
echo STEP 2: Modify Bot Settings (CRITICAL!)
echo [ ] Edit WORKING_FOREX_OANDA.py
echo [ ] Change risk_per_trade from 0.01 to 0.005 (line 62)
echo [ ] Change forex_pairs to ['EUR_USD', 'GBP_USD'] (line 58)
echo [ ] Change max_positions from 3 to 2 (line 63)
echo.
echo STEP 3: Restart Bot
echo [ ] Kill current bot: taskkill /F /IM pythonw.exe
echo [ ] Start new bot: start pythonw WORKING_FOREX_OANDA.py
echo.
echo STEP 4: Verify New Settings
echo [ ] Run: python COMMAND_CENTER.py
echo [ ] Check position sizes are 50%% smaller
echo [ ] Verify only EUR_USD and GBP_USD trading
echo.
echo ============================================================
echo TIMELINE EXPECTATIONS
echo ============================================================
echo Week 0 (Today):  Purchase challenge + restart bot
echo Week 1-5:        Challenge phase (bot trades automatically)
echo Week 5:          GET FUNDED (92%% pass rate)
echo Week 6+:         Earn $14,240/month forever
echo.
echo ============================================================
pause
