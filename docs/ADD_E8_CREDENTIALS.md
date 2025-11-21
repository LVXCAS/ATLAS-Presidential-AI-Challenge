# Add E8 TradeLocker Credentials to .env

## Step 1: Get Your E8 Credentials

You need to get these from your E8 account dashboard.

### Where to Find Them

1. Go to: https://client.e8markets.com
2. Log into your E8 account
3. Navigate to your **Challenge Dashboard**
4. Look for **Trading Platform Details** or **TradeLocker Access**

You should see something like:

```
Platform: TradeLocker
Email: your_email@example.com
Password: [Show Password]
Server: E8-Live
```

### What Each Field Means

| Field | Description | Example |
|-------|-------------|---------|
| **Email** | Your E8 account email | `yourname@gmail.com` |
| **Password** | Your E8 trading password (not login password!) | `Tr4d3P@ss123` |
| **Server** | E8 server name | `E8-Live` or `E8-Demo` |

**Important**: The TradeLocker password might be different from your E8 website login password.

---

## Step 2: Add Credentials to .env File

Open your `.env` file and add these lines at the end:

```bash
# E8 TRADELOCKER CREDENTIALS (for $200K challenge)
TRADELOCKER_EMAIL=your_email@example.com
TRADELOCKER_PASSWORD=your_e8_trading_password
TRADELOCKER_SERVER=E8-Live
TRADELOCKER_ENV=https://api.tradelocker.com
```

### Replace These Values

- `your_email@example.com` → Your actual E8 email
- `your_e8_trading_password` → Your actual E8 TradeLocker password
- `E8-Live` → Keep as-is for live challenge, or use `E8-Demo` for demo

### Example (Filled In)

```bash
# E8 TRADELOCKER CREDENTIALS (for $200K challenge)
TRADELOCKER_EMAIL=john.trader@gmail.com
TRADELOCKER_PASSWORD=MySecurePass123!
TRADELOCKER_SERVER=E8-Live
TRADELOCKER_ENV=https://api.tradelocker.com
```

---

## Step 3: Verify Server Name

E8 typically uses one of these server names:

- **E8-Live** - For real funded challenges
- **E8-Demo** - For demo/practice accounts
- **E8Markets-Live** - Alternative name (check your dashboard)

**Double-check** the exact server name in your E8 dashboard. Capitalization matters!

---

## Step 4: Security Note

⚠️ **Never commit .env to git!**

Your `.env` file contains sensitive credentials. Make sure it's in your `.gitignore`:

```bash
# Check if .env is ignored
git check-ignore .env
```

Should output: `.env` (means it's ignored ✓)

If not ignored:
```bash
echo ".env" >> .gitignore
```

---

## Complete .env Template

Your full `.env` file should now look like this:

```bash
# ... (your existing credentials above) ...

# OANDA CREDENTIALS (for paper trading / testing)
OANDA_API_KEY=0bff5dc7375409bb8747deebab8988a1-d8b26324102c95d6f2b6f641bc330a7c
OANDA_ACCOUNT_ID=101-001-37330890-001

# E8 TRADELOCKER CREDENTIALS (for $200K challenge)
TRADELOCKER_EMAIL=your_email@example.com
TRADELOCKER_PASSWORD=your_e8_trading_password
TRADELOCKER_SERVER=E8-Live
TRADELOCKER_ENV=https://api.tradelocker.com
```

---

## Troubleshooting

### "I can't find TradeLocker credentials in my E8 dashboard"

**Solution**: Contact E8 support and ask:
> "Where can I find my TradeLocker API credentials for automated trading?"

### "I have Match-Trader, not TradeLocker"

E8 offers multiple platforms. For this bot, you need TradeLocker access.

**Solutions**:
1. Check if you can switch platforms in your E8 dashboard
2. Ask E8 support: "Can I get TradeLocker access for my challenge?"
3. If E8 forces Match-Trader, we'll need to adapt the bot (different API)

### "What's the difference between Email and Username?"

TradeLocker uses **email** as the login username. Just use your E8 account email.

---

## Next Steps

After adding credentials to `.env`:

1. Save the file
2. Test connection: `python E8_TRADELOCKER_ADAPTER.py`
3. If successful, run bot: `python E8_FOREX_BOT.py`

---

## Need Help?

If you're stuck getting credentials:

1. **E8 Support**: https://e8markets.com/support or email support@e8markets.com
2. **E8 Discord/Telegram**: Check if they have a community channel
3. **TradeLocker Docs**: https://tradelocker.com/api

Most prop firms have live chat - that's usually the fastest way to get help.
