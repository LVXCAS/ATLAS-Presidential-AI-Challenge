# 📦 LIBRARIES TO KEEP (From Your 623 Packages)

**Analysis Date:** October 14, 2025, 11:45 AM PT
**Total Installed:** 623 packages
**Should Keep:** 20 packages (3%)
**Should Delete:** 603 packages (97%)

---

## ✅ CORE PRODUCTION (MUST KEEP - 15 libraries)

These are **actively used** in your production trading system:

### **Trading Execution (3):**
```
1. alpaca-trade-api (3.2.0)     ← Used in 15 files (old SDK)
2. alpaca-py (0.42.1)           ← Used in 49 files (new SDK)
3. v20 (3.0.25.0)               ← OANDA forex trading
```
**Why both Alpaca packages?** You're transitioning from old SDK to new SDK. Both are currently used.

---

### **Data Processing (3):**
```
4. pandas (2.3.2)               ← Used in 100+ files (core data structure)
5. numpy (2.2.6)                ← Used in 80+ files (numerical operations)
6. yfinance                     ← Market data from Yahoo Finance
```

---

### **Machine Learning (2):**
```
7. scikit-learn (1.7.0)         ← RandomForest, GradientBoosting, preprocessing
8. scipy (1.15.3)               ← Used for options pricing (scipy.stats.norm)
```

---

### **AI Integration (2):**
```
9. anthropic (0.58.2)           ← Claude API for AI agents
10. openai (1.97.1)             ← OpenAI API (backup provider)
```

---

### **Utilities (5):**
```
11. python-dotenv (1.1.1)       ← Load .env files
12. requests (2.32.5)           ← HTTP requests for APIs
13. schedule (1.2.2)            ← Job scheduling (auto_options_scanner.py)
14. python-dateutil (2.9.0)     ← Date/time parsing
15. pytz (2025.2)               ← Timezone handling
```

---

## 🟡 OPTIONAL (NICE TO HAVE - 5 libraries)

These are used in **some files** but not critical:

### **Async/Networking (2):**
```
16. aiohttp (3.13.0)            ← Used in: market_scanner.py, broker_connector.py
17. websockets (15.0.1)         ← Used in: broker_connector.py
```
**Status:** Used in 2-3 files. If you don't use those files in production, can skip.

---

### **Terminal UI (1):**
```
18. colorama (0.4.6)            ← Colored terminal output (mission_control_logger.py)
```
**Status:** Nice visual output but not required for trading.

---

### **Development (2):**
```
19. pytest (8.4.1)              ← Unit testing
20. black (25.1.0)              ← Code formatting
```
**Status:** Development tools only. Not needed for production trading.

---

## ❌ DELETE EVERYTHING ELSE (603 packages!)

### **Unused Quant Platforms (29 packages):**
```
❌ QuantLib                     - Complex derivatives library (you don't use)
❌ quantconnect + stubs         - QuantConnect platform (abandoned)
❌ lean                         - QuantConnect LEAN engine (100+ MB!)
❌ backtrader                   - Backtesting platform (not used)
❌ bt                           - Another backtesting platform
❌ zipline + bcolz-zipline      - Zipline platform (abandoned)
❌ vectorbt                     - Vector backtesting (HUGE, unused)
❌ pyfolio-reloaded             - Portfolio analytics (unused)
❌ empyrical-reloaded           - Financial stats (unused)
❌ QuantStats                   - Quant statistics (unused)
❌ Riskfolio-Lib                - Portfolio optimization (unused)
❌ pyportfolioopt               - Portfolio optimization (unused)
❌ FinRL                        - Reinforcement learning (unused)
❌ FinQuant                     - Financial analysis (unused)
❌ fastquant                    - Fast backtesting (unused)
❌ financetoolkit               - Finance toolkit (unused)
❌ financepy                    - Finance library (unused)
❌ financedatabase              - Financial database (unused)
❌ ffn                          - Financial functions (unused)
❌ gs-quant                     - Goldman Sachs library (HUGE, unused)
❌ freqtrade + client           - Crypto trading bot (you don't trade crypto!)
❌ cvxpy                        - Convex optimization (unused)
❌ PuLP                         - Linear programming (unused)
❌ deap                         - Genetic algorithms (unused)
❌ bayesian-optimization        - Bayesian optimization (unused)
... and 4 more
```
**SAVES:** ~500-800 MB

---

### **Unused ML/AI Frameworks (20+ packages):**
```
❌ keras                        - Deep learning (you use scikit-learn)
❌ tensorboard + server         - TensorFlow visualization
❌ tensorflow (implied)         - Deep learning framework (500+ MB!)
❌ torch                        - PyTorch (2+ GB with CUDA!)
❌ torchvision                  - PyTorch vision (500+ MB)
❌ transformers                 - HuggingFace models (HUGE)
❌ tokenizers                   - Transformer tokenizers
❌ stable_baselines3            - Reinforcement learning
❌ jax + jaxlib                 - Google ML framework (200+ MB)
❌ optax                        - JAX optimization
❌ pymc + pytensor              - Bayesian modeling (HUGE)
❌ arviz                        - Bayesian visualization
❌ lightgbm                     - Gradient boosting (unused)
❌ xgboost                      - Gradient boosting (unused)
❌ gymnasium                    - RL environments
❌ chex                         - JAX testing
❌ mctx                         - Monte Carlo tree search
... and 3+ more
```
**SAVES:** ~3-4 GB (!!)

---

### **Unused Data Sources (40+ packages):**
```
❌ polygon-api-client           - Polygon.io (costs money, unused)
❌ alpha_vantage                - Alpha Vantage API (unused)
❌ Quandl                       - Quandl API (deprecated)
❌ fredapi                      - Federal Reserve API (unused)
❌ ccxt                         - Crypto exchange API (you don't trade crypto)
❌ python-binance               - Binance API (crypto)
❌ pycoingecko                  - CoinGecko API (crypto)
❌ tradingview-ta               - TradingView TA (unused)
❌ ib-insync                    - Interactive Brokers (you use Alpaca)
❌ MetaTrader5                  - MetaTrader 5 (you use OANDA)
❌ kaggle                       - Kaggle API (unused)

OPENBB PLATFORM (30 PACKAGES!):
❌ openbb                       - Only in example file!
❌ openbb-benzinga              - News provider
❌ openbb-bls                   - Bureau of Labor Statistics
❌ openbb-cftc                  - CFTC data
❌ openbb-commodity             - Commodity data
❌ openbb-congress-gov          - Congress data
❌ openbb-crypto                - Crypto data
❌ openbb-currency              - Currency data
❌ openbb-derivatives           - Derivatives data
❌ openbb-econdb                - Economic database
❌ openbb-economy               - Economy data
❌ openbb-equity                - Equity data
❌ openbb-etf                   - ETF data
❌ openbb-federal-reserve       - Fed data
❌ openbb-fixedincome           - Fixed income
❌ openbb-fmp                   - Financial Modeling Prep
❌ openbb-fred                  - FRED data
❌ openbb-imf                   - IMF data
❌ openbb-index                 - Index data
❌ openbb-intrinio              - Intrinio data
❌ openbb-news                  - News data
❌ openbb-oecd                  - OECD data
❌ openbb-platform-api          - Platform API
❌ openbb-polygon               - Polygon integration
❌ openbb-regulators            - Regulator data
❌ openbb-sec                   - SEC filings
❌ openbb-tiingo                - Tiingo data
❌ openbb-tradingeconomics      - Trading Economics
❌ openbb-us-eia                - US Energy Info
❌ openbb-yfinance              - YFinance integration
```
**SAVES:** ~200-300 MB (OpenBB alone!)

---

### **Unused Technical Analysis (6 packages):**
```
❌ TA-Lib                       - C library for TA (you calculate manually)
❌ pandas-ta                    - Pandas TA wrapper
❌ ta                           - Another TA library
❌ finta                        - Financial TA
❌ ft-pandas-ta                 - Freqtrade TA
❌ technical                    - Freqtrade indicators
```
**Why excluded?** You calculate EMA, RSI, ATR manually using pandas in your strategies.
**SAVES:** ~100 MB

---

### **Unused Visualization (12 packages):**
```
❌ plotly                       - Interactive plots
❌ cufflinks                    - Plotly for pandas
❌ matplotlib                   - Static plots (HUGE)
❌ seaborn                      - Statistical plots
❌ dash                         - Dashboard framework
❌ streamlit                    - Dashboard framework (50+ MB)
❌ altair                       - Declarative viz
❌ pydeck                       - Map viz
❌ graphviz                     - Graph viz
❌ pyvis                        - Network viz
❌ pyqtgraph                    - Qt graphs
❌ pygame                       - Game library (??)
```
**Why excluded?** Only used in multi_strategy_backtesting.py (legacy file). Production system doesn't visualize.
**SAVES:** ~200 MB

---

### **Unused Web/Scraping (15 packages):**
```
❌ Scrapy                       - Web scraping framework (HUGE)
❌ selenium                     - Browser automation
❌ beautifulsoup4 + bs4         - HTML parsing
❌ lxml                         - XML/HTML parser
❌ newspaper3k                  - News scraping
❌ feedparser                   - RSS feeds
❌ trafilatura                  - Web scraping
❌ courlan                      - URL handling
❌ jusText                      - Text extraction
❌ inscriptis                   - HTML to text
❌ pdfminer.six                 - PDF extraction
❌ pdfplumber                   - PDF extraction
❌ tweepy                       - Twitter API
❌ curl_cffi                    - Curl bindings
... and more
```
**SAVES:** ~100 MB

---

### **Unused Agent Frameworks (15 packages):**
```
❌ langchain                    - LangChain framework
❌ langchain-anthropic          - Anthropic integration
❌ langchain-community          - Community tools
❌ langchain-core               - Core functionality
❌ langchain-experimental       - Experimental
❌ langchain-openai             - OpenAI integration
❌ langchain-text-splitters     - Text splitting
❌ langgraph                    - Graph workflows
❌ langgraph-checkpoint         - Checkpointing
❌ langgraph-prebuilt           - Prebuilt graphs
❌ langgraph-sdk                - SDK
❌ langsmith                    - Observability
❌ crewai                       - Multi-agent framework
❌ chromadb                     - Vector database
❌ instructor                   - Structured LLM outputs
❌ litellm                      - LLM proxy
```
**Why excluded?** You use Anthropic API directly (or Claude Code agents). If you DO use LangChain, add back.
**SAVES:** ~100 MB

---

### **Random Bloat (400+ packages):**
```
❌ kubernetes                   - K8s API (??)
❌ docker                       - Docker API (??)
❌ twilio                       - SMS API
❌ sendgrid                     - Email API
❌ python-telegram-bot          - Telegram bot
❌ auth0-python                 - Auth0
❌ flask                        - Web framework
❌ fastapi                      - API framework
❌ supabase                     - Supabase client
❌ sqlalchemy                   - ORM (you use JSON)
❌ alembic                      - Database migrations
❌ redis                        - Redis client
❌ cryptography                 - Crypto library (huge)
❌ nltk                         - NLP toolkit
❌ textblob                     - NLP
❌ sympy                        - Symbolic math
❌ astropy                      - Astronomy (??)
❌ geopy                        - Geocoding (??)
❌ pillow                       - Image processing
❌ imageio                      - Image I/O
❌ reportlab                    - PDF generation
❌ tables                       - HDF5 tables
❌ blosc2                       - Compression
❌ numba                        - JIT compiler (HUGE)
... and 370+ MORE packages
```
**SAVES:** ~500+ MB

---

## 📊 FINAL SUMMARY

### **Keep These 20 Libraries:**

**CORE (15):**
1. alpaca-trade-api
2. alpaca-py
3. v20
4. pandas
5. numpy
6. yfinance
7. scikit-learn
8. scipy
9. anthropic
10. openai
11. python-dotenv
12. requests
13. schedule
14. python-dateutil
15. pytz

**OPTIONAL (5):**
16. aiohttp
17. websockets
18. colorama
19. pytest
20. black

### **Delete 603 Packages:**
- 29 quant platforms
- 20+ ML frameworks
- 40+ data sources
- 6 TA libraries
- 12 visualization libraries
- 15 web scraping libraries
- 15 agent frameworks
- 400+ random bloat

### **Savings:**
```
Current:  623 packages, ~6 GB
After:     20 packages, ~300 MB
Saved:    603 packages, ~5.7 GB
```

---

## 🎯 WHAT TO DO

### **Option A: Conservative (Recommended for Now)**
```bash
# Just use requirements_production_REAL.txt going forward
# Don't mess with current environment while trading
# When deploying to Raspberry Pi, use clean requirements
```

### **Option B: Clean Environment**
```bash
# 1. Create clean environment
python -m venv venv_clean

# 2. Activate
venv_clean\Scripts\activate

# 3. Install CORE 15 libraries
pip install alpaca-trade-api alpaca-py v20 pandas numpy yfinance scikit-learn scipy anthropic openai python-dotenv requests schedule python-dateutil pytz

# 4. Test production systems
python auto_options_scanner.py --once
python monitor_positions.py

# 5. If works, add optional 5
pip install aiohttp websockets colorama pytest black
```

### **Option C: Nuclear (Most Aggressive)**
```bash
# Uninstall ALL 603 bloat packages
# Keep ONLY the 20 needed
# Frees 5.7 GB disk space
# (Agent can generate uninstall script)
```

---

**Path:** `LIBRARIES_TO_KEEP.md`
**Status:** Analysis complete, ready for cleanup
