# ATLAS Track II Unit Tests

Comprehensive unit tests for ATLAS core agent logic.

## Running Tests

```bash
# From the ATLAS_HYBRID directory
python3 -m unittest tests.test_agents -v

# Or run specific test classes
python3 -m unittest tests.test_agents.TestAgentAggregation -v
python3 -m unittest tests.test_agents.TestBaselineRisk -v
```

## Test Coverage

### 1. Agent Aggregation Logic (10 tests)
- **Veto mechanism**: Tests that veto agents with score >= 0.80 force STAND_DOWN
- **Threshold tests**:
  - GREENLIGHT: score < 0.25
  - WATCH: 0.25 <= score < 0.36
  - STAND_DOWN: score >= 0.36
- **Data sufficiency**: Agents with insufficient data get weight 0
- **Edge cases**: All insufficient data defaults to WATCH

### 2. Baseline Risk Calculation (6 tests)
- **ATR thresholds**: >= 25 pips triggers STAND_DOWN
- **RSI extremes**: >= 72 or <= 28 triggers WATCH
- **ADX flags**: <= 18 triggers WATCH (choppy/uncertain)
- **Combined conditions**: Multiple moderate conditions
- **Normal conditions**: Proper GREENLIGHT behavior

### 3. Baseline Assessment Mapping (3 tests)
- GREENLIGHT → score 0.20
- WATCH → score 0.55
- STAND_DOWN → score 0.90

### 4. Stress Detection (4 tests)
- **Volatility threshold**: 8 pips (configurable)
- **High volatility detection**: Properly flags stress
- **Low volatility**: Doesn't trigger false positives
- **Edge cases**: First 2 points never marked as stress

### 5. Weighted Aggregation (5 tests)
- **Weight normalization**: Different weights properly normalized
- **Score formula**: `Σ(score × weight) / Σ(weight)`
- **Equal weights**: Produces simple average
- **Single agent**: Returns exact score
- **Zero weight**: Excluded from calculation

### 6. ATR Calculation (3 tests)
- Stable prices → zero ATR
- Volatile prices → measurable ATR
- Insufficient data → zero ATR

### 7. Drivers and Explanations (3 tests)
- Top drivers included and sorted
- Explanations contain risk scores
- Risk flag codes properly populated

## Test Statistics

- **Total tests**: 31
- **Test classes**: 7
- **Code coverage**: Core aggregation, baseline, and stress detection logic
- **Runtime**: < 0.01s

## Key Test Patterns

### Testing Veto Mechanism
```python
agent_assessments = {
    "VetoAgent": {
        "score": 0.85,
        "is_veto": True,
        # ... forces STAND_DOWN
    }
}
```

### Testing Thresholds
```python
# GREENLIGHT: < 0.25
{"score": 0.20} → "GREENLIGHT"

# WATCH: 0.25 to 0.36
{"score": 0.30} → "WATCH"

# STAND_DOWN: >= 0.36
{"score": 0.40} → "STAND_DOWN"
```

### Testing Data Sufficiency
```python
{
    "score": 0.90,
    "details": {"data_sufficiency": "insufficient"}
    # Weight set to 0, excluded from aggregation
}
```

## Dependencies

Tests use only Python standard library:
- `unittest` - Test framework
- No external dependencies required

## Integration

These tests validate the core decision logic used by:
- `quant_team_utils.py` - Main aggregation functions
- `core/coordinator.py` - ATLAS coordinator
- All agent implementations inheriting from `BaseAgent`
