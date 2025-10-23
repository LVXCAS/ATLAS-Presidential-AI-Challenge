# Jupyter Notebook Quick Start Guide

## ✅ Setup Complete!

Jupyter Notebook is now installed with a comprehensive ML experimentation notebook ready to use.

---

## How to Start Jupyter

### Option 1: Use the Launcher Script (Recommended)
```bash
python start_jupyter.py
```

This will:
- Start the Jupyter server
- Automatically open your browser
- Navigate you to the notebook directory

### Option 2: Manual Start
```bash
python -m notebook
```

Then navigate to: `http://localhost:8888`

---

## What's in the ML Experimentation Notebook?

The `ML_Experimentation.ipynb` notebook includes:

### 1. **Load Existing Models**
- Inspect your trained RandomForest and XGBoost models
- See what features they expect
- Check model parameters

### 2. **Download Fresh Data**
- Gets 2 years of data for 8 stocks (AAPL, MSFT, GOOGL, etc.)
- Fast and ready to experiment with

### 3. **Feature Engineering**
- Calculates all 26 technical indicators
- Returns, SMAs, RSI, MACD, Bollinger Bands, Volatility, Volume, ATR, etc.

### 4. **Model Testing**
- Test existing models on new data
- See accuracy scores
- Compare RF vs XGB vs Ensemble

### 5. **Visualizations**
- Confusion matrices for each model
- Feature importance charts
- Algorithm comparison bar charts

### 6. **Experiments**
- **Feature Reduction**: Try using only top 10 most important features
- **Algorithm Comparison**: Test 6 different algorithms (RF, XGB, AdaBoost, etc.)
- **Hyperparameter Tuning**: Find optimal model settings

### 7. **Save Best Models**
- Automatically saves improved models to `models/experimental_rf_model.pkl`

---

## How to Use the Notebook

### Basic Navigation
1. **Run a cell**: Click on it, then press `Shift + Enter`
2. **Add new cell**: Press `B` (below) or `A` (above) when not editing
3. **Delete cell**: Press `D` twice
4. **Save notebook**: Press `Ctrl + S`

### Running the Experiments

**Step 1**: Start from the top and run each cell sequentially

**Step 2**: Wait for each cell to finish (you'll see `[*]` while running, then `[1]`, `[2]`, etc.)

**Step 3**: Visualizations will appear directly below each cell

**Step 4**: Modify code in cells to try your own experiments!

---

## Example Experiments You Can Try

### 1. Add More Stocks
Edit cell #3:
```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN',
           'NFLX', 'SPY', 'QQQ', 'DIA', 'IWM']  # Add more tickers!
```

### 2. Try Different Time Periods
Edit cell #3:
```python
start_date = end_date - timedelta(days=365*3)  # 3 years instead of 2
```

### 3. Test Different Feature Combinations
Create a new cell after cell #8:
```python
# Try only momentum features
momentum_features = ['returns_1d', 'returns_5d', 'returns_20d', 'rsi', 'macd']
X_momentum = X[momentum_features]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_momentum, y, test_size=0.2)

rf_momentum = RandomForestClassifier(n_estimators=100, random_state=42)
rf_momentum.fit(X_train_m, y_train_m)
print(f"Momentum-only accuracy: {rf_momentum.score(X_test_m, y_test_m)*100:.2f}%")
```

### 4. Analyze Individual Stocks
Add a cell after cell #4:
```python
# Analyze AAPL specifically
aapl_df = featured_data['AAPL']
aapl_labels = create_labels(aapl_df)

plt.figure(figsize=(12, 6))
plt.plot(aapl_df.index, aapl_df['Close'], label='Price')
plt.scatter(aapl_df.index[aapl_labels == 1], aapl_df['Close'][aapl_labels == 1],
            color='green', alpha=0.3, label='Up Next Day')
plt.scatter(aapl_df.index[aapl_labels == 0], aapl_df['Close'][aapl_labels == 0],
            color='red', alpha=0.3, label='Down Next Day')
plt.title('AAPL Price with Next-Day Direction Labels')
plt.legend()
plt.show()
```

---

## Tips & Tricks

### Speed Up Experimentation
- **Use fewer stocks** while testing (just 2-3 stocks trains much faster)
- **Reduce n_estimators** to 50 for quick tests
- Once you find settings that work, scale up to full dataset

### Save Your Work
- Notebooks auto-save every 2 minutes
- But **manually save** (`Ctrl+S`) before closing
- You can export results: `File → Download as → HTML/PDF`

### Visualize Everything
```python
# Quick visualization template
plt.figure(figsize=(10, 6))
plt.plot(your_data)
plt.title('Your Title Here')
plt.show()
```

### If Something Breaks
- **Restart kernel**: `Kernel → Restart & Clear Output`
- **Re-run all cells**: `Kernel → Restart & Run All`
- Variables persist between cells - restart kernel to start fresh

---

## Next Steps After Experimentation

Once you find better models or parameters:

### 1. Save the Model
```python
# In notebook
with open('models/new_improved_model.pkl', 'wb') as f:
    pickle.dump(your_best_model, f)
```

### 2. Update ML Ensemble Wrapper
Edit `ai/ml_ensemble_wrapper.py` to load your new model

### 3. Test in OPTIONS_BOT
Restart OPTIONS_BOT to use the updated models

---

## Common Issues & Solutions

### Issue: "Kernel busy"
**Solution**: Wait for current cell to finish, or interrupt kernel (`Kernel → Interrupt`)

### Issue: "No module named 'xyz'"
**Solution**: Install in a new cell:
```python
!pip install xyz
```

### Issue: "Can't find file"
**Solution**: Check you're in the right directory:
```python
import os
print(os.getcwd())  # Should show PC-HIVE-TRADING
```

### Issue: Out of memory
**Solution**: Reduce dataset size - use fewer stocks or shorter time period

---

## Useful Jupyter Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | `Shift + Enter` |
| Run cell (stay in cell) | `Ctrl + Enter` |
| Insert cell below | `B` |
| Insert cell above | `A` |
| Delete cell | `D D` (press D twice) |
| Undo delete | `Z` |
| Save notebook | `Ctrl + S` |
| Find and replace | `Ctrl + F` |
| Command palette | `Ctrl + Shift + P` |

---

## Have Fun Experimenting! 🚀

The notebook is designed for exploration. Try things! Break things! Learn!

**Remember**: The best way to learn ML is by experimenting interactively, and that's exactly what Jupyter notebooks are built for.

Good luck finding those alpha-generating models! 📈
