# 🗑️ ACTUAL BLOAT TO DELETE (From 623 Packages)

**Date:** October 14, 2025

**Professional Libraries to KEEP:** ~85 libraries
**Actual Bloat to DELETE:** ~538 libraries

---

## ❌ DEEP LEARNING FRAMEWORKS (Delete - You Use Scikit-Learn)

### **TensorFlow Ecosystem (~500-600 MB):**
```
✗ keras
✗ tensorboard
✗ tensorboard-data-server
✗ tensorflow (implied by dependencies)
✗ gast
✗ google-pasta
✗ astunparse
✗ libclang
✗ ml_dtypes
✗ opt_einsum
✗ flatbuffers
```
**Why Delete:** You use scikit-learn for ML, not deep learning.
**Saves:** ~500 MB

---

### **PyTorch Ecosystem (~2+ GB!):**
```
✗ torch
✗ torchvision
✗ torchaudio (if installed)
```
**Why Delete:** HUGE (2+ GB with CUDA), you don't use PyTorch.
**Saves:** ~2+ GB

---

### **Other ML Frameworks:**
```
✗ jax
✗ jaxlib (200+ MB)
✗ optax
✗ chex
✗ mctx
✗ optree
✗ lightgbm (you have scikit-learn)
✗ xgboost (you have scikit-learn)
```
**Saves:** ~300 MB

---

### **Bayesian/Statistical Modeling:**
```
✗ pymc
✗ pytensor
✗ arviz
```
**Why Delete:** Unless you're doing Bayesian modeling, don't need.
**Saves:** ~150 MB

---

### **Reinforcement Learning:**
```
✗ stable_baselines3
✗ gymnasium
```
**Saves:** ~100 MB

---

### **NLP/Transformers:**
```
✗ transformers (HUGE - 500+ MB)
✗ tokenizers
✗ huggingface-hub
✗ safetensors
```
**Why Delete:** Unless doing sentiment analysis with transformers.
**Saves:** ~600 MB

**TOTAL DEEP LEARNING SAVINGS: ~3.5-4 GB**

---

## ❌ ABANDONED/ALTERNATIVE PLATFORMS

### **QuantConnect:**
```
✗ quantconnect
✗ quantconnect-stubs
✗ lean (100+ MB!)
```
**Why Delete:** You use Alpaca, not QuantConnect.
**Saves:** ~150 MB

---

### **Zipline:**
```
✗ zipline (if installed)
✗ bcolz-zipline
✗ trading-calendars
✗ exchange_calendars
```
**Saves:** ~100 MB

---

### **Other Platforms:**
```
✗ ib-insync (Interactive Brokers - you use Alpaca)
✗ MetaTrader5 (you use OANDA for forex)
```
**Saves:** ~50 MB

---

## ❌ CRYPTO TRADING (If Not Trading Crypto)

```
✗ ccxt (crypto exchange API)
✗ python-binance
✗ pycoingecko
✗ freqtrade
✗ freqtrade-client
```
**Why Delete:** You're trading options/forex/futures, not crypto.
**Saves:** ~100 MB

---

## ❌ DUPLICATE/ALTERNATIVE DATA SOURCES

**Keep:** OpenBB, polygon, alpha_vantage, fredapi
**Delete:**
```
✗ Quandl (deprecated API)
✗ twelvedata
✗ iexfinance
✗ kaggle
```
**Saves:** ~30 MB

---

## ❌ WEB SCRAPING (If Not Scraping)

```
✗ Scrapy (HUGE web scraping framework)
✗ selenium (browser automation)
✗ beautifulsoup4 / bs4 (HTML parsing)
✗ lxml
✗ newspaper3k
✗ feedparser
✗ trafilatura
✗ courlan
✗ jusText
✗ inscriptis
```
**Why Delete:** Unless you're scraping news/data, don't need.
**Saves:** ~150 MB

---

## ❌ PDF/DOCUMENT PROCESSING

```
✗ pdfminer.six
✗ pdfplumber
✗ reportlab
✗ pypdfium2
```
**Why Delete:** Unless processing SEC filings/documents.
**Saves:** ~50 MB

---

## ❌ COMMUNICATION APIs

```
✗ twilio (SMS)
✗ sendgrid (Email)
✗ python-telegram-bot
```
**Saves:** ~30 MB

---

## ❌ DATABASE/BACKEND (If Using JSON Logging)

```
✗ SQLAlchemy (ORM)
✗ alembic (migrations)
✗ redis
✗ asyncpg
✗ psycopg2-binary
✗ peewee
```
**Why Delete:** You log to JSON files, not databases.
**Saves:** ~80 MB

---

## ❌ WEB FRAMEWORKS (If Not Building Web Apps)

```
✗ Flask
✗ fastapi
✗ starlette
✗ uvicorn
✗ werkzeug
```
**Why Delete:** Unless building web API, don't need.
**Saves:** ~50 MB

---

## ❌ CLOUD/INFRASTRUCTURE

```
✗ kubernetes
✗ docker
✗ supabase
✗ auth0-python
✗ Authlib
```
**Why Delete:** You're not deploying to Kubernetes/Docker (yet).
**Saves:** ~100 MB

---

## ❌ RANDOM/UNRELATED

```
✗ astropy (astronomy)
✗ geopy (geocoding)
✗ pygame (game engine)
✗ pillow (image processing)
✗ imageio
✗ numba (JIT compiler - HUGE)
✗ nltk (NLP toolkit)
✗ textblob
✗ sympy (symbolic math)
```
**Why Delete:** Not related to trading at all.
**Saves:** ~300 MB

---

## ❌ OPTIMIZATION/SPECIALIZED MATH

```
✗ cvxpy (convex optimization)
✗ PuLP (linear programming)
✗ deap (genetic algorithms)
✗ bayesian-optimization
✗ optuna
✗ clarabel
✗ ecos
✗ osqp
✗ scs
```
**Why Delete:** You have Riskfolio-Lib for portfolio optimization.
**Saves:** ~150 MB

---

## ❌ ALTERNATIVE QUANT LIBRARIES (Duplicates)

**Keep:** QuantStats, pyfolio, Riskfolio-Lib, pyportfolioopt
**Delete:**
```
✗ FinRL (reinforcement learning)
✗ FinQuant
✗ fastquant
✗ financetoolkit
✗ financepy
✗ financedatabase
✗ ffn
✗ gs-quant (Goldman Sachs library - HUGE)
✗ Quantsbin
✗ QuantLib (unless pricing exotic derivatives)
```
**Saves:** ~500 MB

---

## ❌ TESTING/BUILD TOOLS (Extras)

**Keep:** pytest, black
**Delete:**
```
✗ pre_commit
✗ coverage
✗ pytest-asyncio (unless testing async code)
✗ pytest-mock
```
**Saves:** ~20 MB

---

## ❌ MISC DEPENDENCIES (Brought In By Other Packages)

```
✗ Tables (HDF5)
✗ blosc2 (compression)
✗ h5py
✗ h5netcdf
✗ cosmpy
✗ uagents / uagents-core
✗ bech32
✗ ecdsa
✗ cosmos stuff
✗ agent (generic)
```
**Saves:** ~100 MB

---

## 📊 SUMMARY

### **KEEP (85 Professional Libraries):**
- Core Trading (3)
- Data Processing (3)
- Machine Learning (2)
- AI Agents & LangChain (15)
- OpenBB Platform (30+)
- Technical Analysis (3)
- Backtesting (3)
- Portfolio Analytics (5)
- Visualization (6)
- Utilities (8)
- Development (2)

### **DELETE (538 Bloat Libraries):**
- Deep Learning (~3.5 GB)
- Abandoned Platforms (~300 MB)
- Crypto Trading (~100 MB)
- Web Scraping (~150 MB)
- Databases (~80 MB)
- Web Frameworks (~50 MB)
- Cloud/Infrastructure (~100 MB)
- Random Unrelated (~300 MB)
- Duplicate Quant Tools (~500 MB)
- Everything Else (~400 MB)

**TOTAL SAVINGS: ~5+ GB**

---

## 🎯 FINAL NUMBERS

```
Current:     623 packages, ~6 GB
Keep:         85 packages, ~1-1.5 GB (professional toolkit)
Delete:      538 packages, ~5 GB of bloat

Reduction:   86% fewer packages, 83% less disk space
```

---

## ✅ NEXT STEP

Use `requirements_professional.txt` which has the 85 libraries you actually want!

**Path:** `ACTUAL_BLOAT_TO_DELETE.md`
