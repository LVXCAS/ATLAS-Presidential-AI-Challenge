# ACCOUNT ROUTING - QUICK REFERENCE GUIDE

## Problem Summary
The options scanner was connecting to the wrong Alpaca account (Account #2 with $0 options BP) instead of the main account (Account #1 with $948k options BP).

## Root Cause
Python's `load_dotenv()` does NOT override existing environment variables by default. Multiple modules were loading different .env files, causing the wrong account credentials to persist.

## Solution
Added `override=True` to all `load_dotenv()` calls and explicit credential passing between modules.

---

## Quick Verification (Run Before Trading)

```bash
# Check which account you're connected to
python check_current_account.py
```

**Expected Output:**
```
Account: PA3MS5F52RNL
Options BP: $948,570.66
```

If you see:
- Account: PA3RRV5YYKAS
- Options BP: $0.00

**DO NOT TRADE!** The wrong account is loaded.

---

## Files Modified

1. **account_verification_system.py**
   - Added credential parameters
   - Added `override=True` to load_dotenv()

2. **unified_validated_strategy_system.py**
   - Added `force_main_account` parameter
   - Added `override=True` to load_dotenv()

3. **week1_execution_system.py**
   - Added `force_main_account` parameter

4. **core/adaptive_dual_options_engine.py**
   - Added credential parameters
   - Added `override=True` to load_dotenv()

5. **week3_production_scanner.py**
   - Added explicit environment loading at startup
   - Added environment check display
   - Passes credentials explicitly to all systems

---

## Usage Examples

### Production Scanner (Use Main Account)
```python
from week3_production_scanner import Week2EnhancedScanner

# Scanner now automatically uses main account
scanner = Week2EnhancedScanner()
# Will show: "Trading on account: PA3MS5F52R... ($955,886)"
```

### Explicit Account Control
```python
from week1_execution_system import Week1ExecutionSystem

# Force main account
system = Week1ExecutionSystem(force_main_account=True)

# Or use paper account for testing
system = Week1ExecutionSystem(force_main_account=False)
```

### Manual Credential Passing
```python
from core.adaptive_dual_options_engine import AdaptiveDualOptionsEngine
import os

engine = AdaptiveDualOptionsEngine(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
```

---

## Account Information

### Account #1 (Main - Production)
- **Account ID:** PA3MS5F52RNL
- **API Key:** PKFGVU14XFD0FX0VP3B7
- **Equity:** ~$956k
- **Options BP:** ~$948k
- **File:** `.env` (main file)

### Account #2 (Testing)
- **Account ID:** PA3RRV5YYKAS
- **API Key:** PKXH5RG8WENVHSFVNCC0
- **Equity:** ~$100k
- **Options BP:** $0 (NOT approved)
- **File:** `.env.paper`

---

## Pre-Trading Checklist

Before starting any trading session:

1. **Verify Account Connection**
   ```bash
   python check_current_account.py
   ```
   Confirm: Account PA3MS5F52RNL, Options BP ~$948k

2. **Start Scanner with Verification**
   ```bash
   python week3_production_scanner.py
   ```
   Look for:
   ```
   [ENVIRONMENT CHECK]
     API Key: PKFGVU14XF...
     Base URL: https://paper-api.alpaca.markets

   [ACCOUNT VERIFICATION]
     Account ID: PA3MS5F52RNL
     Options BP: $948,570.66
   ```

3. **Verify First Trade**
   - Check first trade executes on correct account
   - Verify position appears in correct account

---

## Troubleshooting

### Problem: Scanner shows wrong account
**Solution:**
```bash
# Delete environment cache
set ALPACA_API_KEY=
set ALPACA_SECRET_KEY=
set ALPACA_BASE_URL=

# Re-run scanner
python week3_production_scanner.py
```

### Problem: Multiple .env files confusing
**Solution:**
1. Main trading always uses `.env`
2. Testing uses `.env.paper`
3. Never manually switch - use `force_main_account` parameter

### Problem: Account verification fails
**Check:**
1. Internet connection
2. Alpaca API status
3. API keys in .env file
4. Account not suspended

---

## Safety Features Added

1. **Environment Check Display**
   - Shows first 10 chars of API key at startup
   - Shows base URL being used

2. **Account Verification**
   - Displays full account details before trading
   - Shows options buying power
   - Fails if account not ready

3. **Explicit Credential Passing**
   - Systems can accept credentials directly
   - Prevents environment conflicts

4. **Force Main Account Mode**
   - Production scanners force correct account
   - Cannot accidentally use wrong account

---

## Testing Commands

```bash
# Test 1: Direct account check
python check_current_account.py

# Test 2: Scanner initialization
python -c "from week3_production_scanner import Week2EnhancedScanner; scanner = Week2EnhancedScanner()"

# Test 3: Verify account match
python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os; print(f'API Key: {os.getenv(\"ALPACA_API_KEY\")[:10]}...')"
```

Expected API Key prefix: `PKFGVU14XF...` (Account #1)

---

## Important Notes

1. **Always verify account at startup** - Don't assume environment is correct
2. **Monitor first few trades** - Ensure they execute on correct account
3. **Don't modify .env files manually** - Use code parameters instead
4. **Keep .env.paper for testing** - But production uses main .env
5. **Document any account switches** - For audit trail

---

## Questions?

If account routing issues persist:
1. Check this guide first
2. Run verification commands above
3. Review ACCOUNT_ROUTING_FIX_REPORT.md for detailed analysis
4. Contact system administrator if unresolved

---

**Last Updated:** 2025-10-16
**Status:** PRODUCTION READY
**Verification:** PASSED
