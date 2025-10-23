# 🚀 PC-HIVE AUTO SCANNER - SETUP INSTRUCTIONS

**Goal:** Never manually start the scanner again! Set it and forget it.

---

## ⚡ QUICK SETUP (2 Steps)

### **Step 1: Run Setup as Administrator**

1. Find this file in your directory: **`RUN_ME_AS_ADMIN.bat`**
2. **Right-click** the file
3. Select **"Run as administrator"**
4. Click "Yes" when Windows asks for permission
5. Press any key when prompted
6. Wait for "Setup Complete" message

**That's it!** Your scanner will now run automatically.

---

### **Step 2: Verify It Worked**

Double-click: **`CHECK_SCANNER_STATUS.bat`**

You should see:
```
✅ Daily task: PC-HIVE Auto Scanner (6:30 AM PT)
✅ Startup task: PC-HIVE Auto Scanner - Startup
```

---

## 📅 What Happens Now?

### **Every Morning at 6:30 AM PT:**
- Scanner automatically wakes up
- Scans S&P 500 for bull put spreads
- Executes up to 4 trades (if quality setups found)
- Logs everything to `logs/auto_scanner_YYYYMMDD.log`
- Stops automatically when done

### **After PC Reboot:**
- Scanner auto-starts 1 minute after Windows loads
- Checks if market is open
- Runs scan if it's a trading day
- Goes back to sleep if markets closed

---

## 🎯 Current Status

**Week 3 Goal:** 20 trades by Friday
- ✅ Completed: 6 trades (Oct 14)
- 🎯 Remaining: 14 trades
- 📅 Days left: 3 days (Tue, Wed, Thu)

**With automation:**
- 4 trades/day × 3 days = 12 trades
- Total: 6 + 12 = **18 trades** (close to goal!)

---

## 🛠️ Manual Controls (Optional)

### **Start Now (Manual Test):**
```
START_TRADING.bat          ← Visible window
START_TRADING_BACKGROUND.bat   ← Silent background
```

### **Check Status:**
```
CHECK_SCANNER_STATUS.bat   ← See scheduled tasks
```

### **Emergency Stop:**
```
EMERGENCY_STOP.bat         ← Kill all trading processes
```

---

## 📊 What Gets Logged?

**Location:** `logs/auto_scanner_YYYYMMDD.log`

**Contains:**
- Scan start/end times
- Opportunities found
- Trades executed
- Position details (strikes, premiums, etc.)
- Any errors or warnings

---

## ❓ Troubleshooting

### **Problem: "Access Denied" when running setup**
**Solution:** You must right-click → "Run as administrator"

### **Problem: Tasks not showing up**
**Solution:** Open Task Scheduler manually:
1. Press `Win + R`
2. Type: `taskschd.msc`
3. Look for tasks named "PC-HIVE Auto Scanner"

### **Problem: Scanner not running at 6:30 AM**
**Solution:**
1. Check PC is on at 6:30 AM PT
2. Or enable "Startup" task to run after reboot
3. Check logs for errors

---

## 🎓 System Architecture

```
Windows Task Scheduler
    ↓
START_TRADING_BACKGROUND.bat
    ↓
week3_production_scanner.py
    ↓
Alpaca API (Live Execution)
    ↓
4 trades max/day
    ↓
Logs to executions/execution_log_YYYYMMDD.json
```

---

## ✅ Next Steps

1. ✅ Run `RUN_ME_AS_ADMIN.bat` (right-click → admin)
2. ✅ Verify with `CHECK_SCANNER_STATUS.bat`
3. ✅ Go to bed tonight knowing it'll run tomorrow!
4. 🎯 Wake up to new trades at 6:35 AM PT

---

**Created:** October 15, 2025
**Agent:** Automation Agent #1
**Purpose:** Never miss a trading day due to forgetting to start the scanner!

---

## 💰 Path to $100M

**Month 3:** Pass FTMO $25k challenge
- Need consistent 20 trades/week with automation
- Target: 55%+ win rate, 1.5+ profit factor
- Automation ensures consistency!

**This automation is your first step to truly passive trading!** 🚀
