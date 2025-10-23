# ACCOUNT ROUTING FIX REPORT
## Alpaca Account Routing Issue - Resolution

**Date:** 2025-10-16
**Issue:** Options scanner connecting to wrong Alpaca account
**Status:** RESOLVED
**Severity:** CRITICAL (could result in trading on wrong account)

---

## EXECUTIVE SUMMARY

The options scanner (week3_production_scanner.py) was connecting to Account #2 (PA3RRV5YYKAS) with $0 options buying power instead of Account #1 (PA3MS5F52RNL) with $948k options buying power. This was caused by a cascading environment variable loading issue where multiple modules loaded different .env files, and Python's dotenv library does not override existing environment variables by default.

**Root Cause:** Environment variable loading order and dotenv's non-override default behavior
**Fix Applied:** Added `override=True` to all load_dotenv() calls and explicit credential passing
**Verification:** Both check_current_account.py and week3_production_scanner.py now connect to correct account

---

## PROBLEM ANALYSIS

### Evidence
1. Running `check_current_account.py` connected to Account #1 (PA3MS5F52RNL) with $948k options BP
2. Running `week3_production_scanner.py` connected to Account #2 (PA3RRV5YYKAS) with $0 options BP
3. Multiple .env files exist: `.env`, `.env.paper`, `.env.backup`, `.env.prod`, etc.

### Root Cause

The issue was a **cascading environment variable loading problem**:

1. **Main .env file** contains Account #1 credentials:
   - API Key: PKFGVU14XFD0FX0VP3B7
   - Account: PA3MS5F52RNL ($948k options BP)

2. **.env.paper file** contains Account #2 credentials:
   - API Key: PKXH5RG8WENVHSFVNCC0
   - Account: PA3RRV5YYKAS ($0 options BP)

3. **Module Loading Order**:
   - `week3_production_scanner.py` imports `Week1ExecutionSystem`
   - `Week1ExecutionSystem` inherits from `ValidatedStrategySystem`
   - `ValidatedStrategySystem.__init__(use_paper=True)` calls `load_dotenv('.env.paper')`
   - This sets environment variables to Account #2
   - Later, `AccountVerificationSystem.__init__()` calls `load_dotenv()`
   - **CRITICAL:** `load_dotenv()` by default does NOT override existing variables
   - Result: Account #2 credentials remain active

4. **Why check_current_account.py worked**:
   - It loads .env first (main account)
   - No other module loads .env.paper before it
   - So it connects to Account #1

### Code Flow Analysis

```
week3_production_scanner.py
  → imports Week1ExecutionSystem
    → inherits ValidatedStrategySystem
      → __init__(use_paper=True)
        → load_dotenv('.env.paper')  # Sets Account #2
  → imports AccountVerificationSystem
    → __init__()
      → load_dotenv()  # Tries to set Account #1, but FAILS (no override)
  → imports AdaptiveDualOptionsEngine
    → __init__()
      → load_dotenv()  # Also fails to override

Result: All systems use Account #2 credentials loaded by ValidatedStrategySystem
```

---

## SOLUTION IMPLEMENTED

### Fix Strategy
1. Add `override=True` to all `load_dotenv()` calls
2. Allow passing credentials explicitly to prevent environment conflicts
3. Add `force_main_account` parameter to force loading correct .env file
4. Load main .env file FIRST in week3_production_scanner.py

### Files Modified

#### 1. account_verification_system.py
**Changes:**
- Modified `__init__()` to accept optional credentials
- Added `override=True` to load_dotenv() call
- Prevents environment conflicts by accepting pre-loaded credentials

```python
def __init__(self, api_key=None, secret_key=None, base_url=None):
    if not api_key:
        load_dotenv(override=True)  # Force override
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL')
```

#### 2. unified_validated_strategy_system.py
**Changes:**
- Added `force_main_account` parameter
- Added `override=True` to all load_dotenv() calls
- Added `account_mode` tracking

```python
def __init__(self, use_paper=True, force_main_account=False):
    if force_main_account:
        load_dotenv(override=True)  # Main account
    else:
        load_dotenv('.env.paper' if use_paper else '.env', override=True)

    self.account_mode = 'MAIN_ACCOUNT' if force_main_account else ('PAPER' if use_paper else 'LIVE')
```

#### 3. week1_execution_system.py
**Changes:**
- Added `force_main_account` parameter
- Passes parameter to parent ValidatedStrategySystem

```python
def __init__(self, force_main_account=False):
    super().__init__(use_paper=True, force_main_account=force_main_account)
```

#### 4. core/adaptive_dual_options_engine.py
**Changes:**
- Modified `__init__()` to accept optional credentials
- Added `override=True` to load_dotenv() call

```python
def __init__(self, api_key=None, secret_key=None, base_url=None):
    if not api_key:
        load_dotenv(override=True)
        api_key = os.getenv('ALPACA_API_KEY')
        # ... get other credentials
```

#### 5. week3_production_scanner.py
**Changes:**
- Added explicit `load_dotenv(override=True)` at start of `__init__()`
- Added environment check to display loaded credentials
- Pass `force_main_account=True` to Week1ExecutionSystem
- Pass explicit credentials to AdaptiveDualOptionsEngine

```python
def __init__(self):
    # CRITICAL FIX: Force load main .env file FIRST
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Environment check
    print(f"API Key: {os.getenv('ALPACA_API_KEY')[:10]}...")

    # Force main account for all systems
    self.system = Week1ExecutionSystem(force_main_account=True)
    self.options_engine = AdaptiveDualOptionsEngine(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL')
    )
```

---

## VERIFICATION RESULTS

### Test 1: check_current_account.py
```bash
$ python check_current_account.py
Account: PA3MS5F52RNL
Equity: $955,885.66
Options BP: $948,570.66
Regular BP: $3,171,186.88
Cash: $959,070.66
Open Positions: 17
```
**Result:** PASS - Connects to Account #1

### Test 2: week3_production_scanner.py
```bash
$ python week3_production_scanner.py
[ENVIRONMENT CHECK]
  API Key: PKFGVU14XF... (first 10 chars)
  Base URL: https://paper-api.alpaca.markets

[ACCOUNT VERIFICATION]
  Account ID: PA3MS5F52RNL
  Equity: $955,885.66
  Options Buying Power: $948,570.66
  Options Level: 3

[OK] Trading on account: PA3MS5F52R... ($955,886)
```
**Result:** PASS - Connects to Account #1

### Comparison
| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| check_current_account.py | PA3MS5F52RNL | PA3MS5F52RNL |
| week3_production_scanner.py | PA3RRV5YYKAS | PA3MS5F52RNL |
| Options BP | $0 (wrong) | $948k (correct) |

---

## SAFETY IMPROVEMENTS

### 1. Environment Visibility
Added environment check at scanner startup:
```python
print(f"[ENVIRONMENT CHECK]")
print(f"  API Key: {os.getenv('ALPACA_API_KEY')[:10]}...")
print(f"  Base URL: {os.getenv('ALPACA_BASE_URL')}")
```

### 2. Account Verification Enhanced
Account verification now displays full account details:
```python
print(f"[ACCOUNT DETAILS]")
print(f"  Account ID: {account_id}")
print(f"  Options Buying Power: ${options_buying_power:,.2f}")
```

### 3. Explicit Credential Passing
Systems can now accept credentials explicitly:
```python
# Instead of relying on environment
engine = AdaptiveDualOptionsEngine(
    api_key=explicit_key,
    secret_key=explicit_secret,
    base_url=explicit_url
)
```

### 4. Force Main Account Mode
Production scanners can force main account:
```python
system = Week1ExecutionSystem(force_main_account=True)
```

---

## LESSONS LEARNED

### 1. Environment Variable Loading
- `load_dotenv()` does NOT override by default
- Always use `load_dotenv(override=True)` in production
- Load environment variables ONCE at application entry point

### 2. Multiple .env Files
- Having multiple .env files (.env, .env.paper, etc.) increases complexity
- Document which file is for which purpose
- Consider using a single .env with environment-specific sections

### 3. Dependency Injection
- Passing credentials explicitly is safer than relying on environment
- Allows better control over which account each component uses
- Easier to test and debug

### 4. Verification is Critical
- Always verify account connection at startup
- Display partial credentials (first 10 chars) for debugging
- Check account ID, not just that connection works

---

## RECOMMENDATIONS

### Immediate Actions
1. Run `python check_current_account.py` before every trading session
2. Verify scanner shows correct account ID at startup
3. Monitor first few trades to ensure correct account

### Short-term
1. Add account ID validation to all trading systems
2. Create a centralized configuration manager
3. Add logging of account switches

### Long-term
1. Consolidate .env files into single source of truth
2. Implement account configuration validation on startup
3. Add automated tests for account routing
4. Consider using environment-specific configuration classes

---

## TESTING CHECKLIST

- [x] check_current_account.py connects to Account #1
- [x] week3_production_scanner.py connects to Account #1
- [x] Scanner displays correct API key prefix
- [x] Account verification shows $948k options BP
- [x] Scanner initializes without errors
- [x] All imported modules use same account

---

## CONCLUSION

The account routing issue has been successfully resolved. The root cause was Python's dotenv library not overriding environment variables by default, combined with multiple modules loading different .env files in sequence. The fix ensures that:

1. All modules use `override=True` when loading environment variables
2. The main .env file is loaded first in production scanners
3. Credentials can be passed explicitly to prevent conflicts
4. Account verification displays clear connection information

**Status:** PRODUCTION READY
**Confidence:** HIGH (tested and verified)
**Risk:** LOW (all safety checks in place)

---

## APPENDIX: Environment File Contents

### .env (Main - Account #1)
```
ALPACA_API_KEY=PKFGVU14XFD0FX0VP3B7
ALPACA_SECRET_KEY=DNmBOxJTU8gK1ua7VXRtPiyMnxz1PF2JYXVdaYlM
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```
Account: PA3MS5F52RNL
Options BP: $948,570.66

### .env.paper (Testing - Account #2)
```
ALPACA_API_KEY=PKXH5RG8WENVHSFVNCC0
ALPACA_SECRET_KEY=9Targ01OCMicdxpdVgkSxZXWwfMpEzgYTfKNPen6
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```
Account: PA3RRV5YYKAS
Options BP: $0 (not approved for options)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-16
**Author:** Claude Code Agent
**Status:** RESOLVED
