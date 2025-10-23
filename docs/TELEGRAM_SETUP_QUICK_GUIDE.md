# 📱 TELEGRAM SETUP - 10 MINUTE GUIDE

**Get trade alerts on your phone!**

---

## Step 1: Create Your Bot (2 minutes)

1. Open Telegram app on your phone
2. Search for: **@BotFather**
3. Click "START"
4. Send this message: `/newbot`
5. Follow prompts:
   - Bot name: "My Trading Bot" (or whatever you want)
   - Username: something ending in "bot" (e.g., "lucas_trading_bot")

6. **SAVE THE TOKEN** - looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

---

## Step 2: Get Your Chat ID (3 minutes)

1. Send any message to your new bot (e.g., "hello")

2. Open this URL in browser (replace `<YOUR_TOKEN>` with the token from Step 1):
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```

3. You'll see JSON response. Find the `chat_id` number:
   ```json
   {
     "result": [{
       "message": {
         "chat": {
           "id": 123456789  ← THIS IS YOUR CHAT_ID
         }
       }
     }]
   }
   ```

4. **SAVE THE CHAT_ID** - it's a number like: `123456789`

---

## Step 3: Add to Your System (5 minutes)

1. Open your `.env` file in the trading directory

2. Add these two lines (use your actual values):
   ```
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

3. Save the file

4. Restart your trading systems:
   ```bash
   # Stop current systems (if running)
   python EMERGENCY_STOP.bat

   # Start with Telegram enabled
   python START_ALL_WEATHER_TRADING.py
   ```

---

## ✅ TEST IT WORKS

Your bot should send you a message when systems start:
- "Trading systems activated"
- "Monitoring started"

When a trade happens, you'll get:
- "Trade opened: [symbol] [strategy]"
- "Stop loss hit: [symbol] -X%"
- "System restarted: [name]"

---

## 🎯 DONE!

You now get all trade alerts on your phone!

**Time invested:** 10 minutes
**Benefit:** Never miss a trade, 24/7 monitoring
