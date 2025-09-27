# 2-Week Expiry Requirement Implementation

## ✅ CHANGES MADE

### 1. **Updated Options Trader Configuration**
**File**: `agents/options_trading_agent.py`

**New Parameter** (Line 99):
```python
self.min_days_to_expiry = 14  # Only trade options with > 2 weeks to expiry
```

### 2. **Enhanced Expiration Filtering**
**Lines 112-135**: Added comprehensive expiration date filtering

**Before**:
```python
target_exp = expirations[0]  # Used nearest expiration
```

**After**:
```python
# Filter expirations to only those > 2 weeks out
valid_expirations = []
today = datetime.now()

for exp_str in expirations:
    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
    days_to_exp = (exp_date - today).days
    if days_to_exp > self.min_days_to_expiry:
        valid_expirations.append(exp_str)

if not valid_expirations:
    logger.warning(f"No options with > {self.min_days_to_expiry} days to expiry for {symbol}")
    return []

target_exp = valid_expirations[0]  # Use nearest VALID expiration
```

### 3. **Double-Check Contract Filtering**
**Lines 166-167 & 191-192**: Added expiration check to contract filtering

**Before**:
```python
if contract.spread / contract.mid_price <= self.max_spread_ratio:
    contracts.append(contract)
```

**After**:
```python
# Filter by spread ratio and expiration time
if (contract.spread / contract.mid_price <= self.max_spread_ratio and 
    contract.days_to_expiry > self.min_days_to_expiry):
    contracts.append(contract)
```

### 4. **Updated Time Decay Exit Logic**
**Lines 411-414**: Adjusted exit logic for longer expiry options

**Before**:
```python
elif position.contracts[0].days_to_expiry <= 5 and pnl < 0:
    should_exit = True
    exit_reason = "Time Decay"
```

**After**:
```python
# Close if < 7 days to expiry and losing, or < 3 days regardless
elif (position.contracts[0].days_to_expiry <= 7 and pnl < 0) or position.contracts[0].days_to_expiry <= 3:
    should_exit = True
    exit_reason = "Time Decay"
```

## 📊 BEHAVIOR CHANGES

### **Options Selection**:
- **Before**: Could trade weekly options (2-7 days to expiry)
- **After**: Only trades options with **>14 days to expiry**

### **Time Decay Management**:
- **Before**: Closed losing positions at 5 days to expiry
- **After**: 
  - Closes losing positions at **7 days to expiry**
  - Force closes ALL positions at **3 days to expiry**
  - Allows winning positions to run longer

### **Risk Profile**:
- **Lower time decay risk** - more time for trades to work
- **Better liquidity** - longer-dated options typically more liquid
- **Higher premium cost** - but more time value protection

## 🎯 TESTING RESULTS

**Test Results from `test_2week_expiry.py`**:
- ✅ All 40 AAPL contracts found had >14 days to expiry
- ✅ Minimum expiry: 15 days (meets requirement)
- ✅ Time decay logic tested for all scenarios
- ✅ System properly filters out short-dated options

## 📈 EXPECTED IMPACT

### **Advantages**:
1. **More time for trades to develop** - reduced time decay pressure
2. **Better risk management** - avoid rapid theta burn
3. **Higher success probability** - more time for price movements
4. **Improved liquidity** - longer-dated options typically more active

### **Trade-offs**:
1. **Higher premium costs** - longer-dated options more expensive
2. **Fewer opportunities** - some short-term plays excluded
3. **Capital efficiency** - more capital tied up per trade

## 🚀 USAGE

The change is **automatic** - no configuration needed. The system now:

1. **Automatically filters** options chains for >14 day expiry
2. **Logs the requirement** when retrieving options
3. **Applies smart exit logic** based on longer holding periods
4. **Warns when no valid expirations** are found

**Example Log Output**:
```
Retrieved 40 liquid options for AAPL with > 14 days to expiry
```

The enhanced trading system will now only consider options positions that give sufficient time for the underlying thesis to play out, reducing the risk of time decay losses that can occur with shorter-dated options.