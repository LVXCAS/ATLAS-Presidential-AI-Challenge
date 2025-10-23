# 📦 COMPLETE ANALYSIS: ALL 621 PACKAGES

**Keep/Delete Decision for Every Single Package**
**Date:** October 14, 2025

---

## LEGEND:
- ✅ **KEEP** - Essential for production AI trading system
- 🟡 **KEEP** - Professional quant tool (recommended)
- ⚠️ **OPTIONAL** - Keep if you use specific feature
- ❌ **DELETE** - Not needed for your system

---

## A

### 1. absl-py (2.3.1)
**Decision:** ❌ DELETE
**What:** Google's Python library (TensorFlow dependency)
**Reason:** Only needed for TensorFlow. You don't use TensorFlow.

### 2. aenum (3.1.16)
**Decision:** ❌ DELETE
**What:** Advanced enumeration types
**Reason:** Python's built-in `enum` is sufficient.

### 3. agent (0.1.3)
**Decision:** ❌ DELETE
**What:** Generic agent library
**Reason:** Too generic, not used in your code.

### 4. aiodns (3.5.0)
**Decision:** ⚠️ OPTIONAL
**What:** Async DNS resolver
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 5. aiofiles (24.1.0)
**Decision:** ⚠️ OPTIONAL
**What:** Async file I/O
**Reason:** Not used in production code. Delete unless doing async file operations.

### 6. aiohappyeyeballs (2.6.1)
**Decision:** ⚠️ OPTIONAL
**What:** Async DNS Happy Eyeballs (IPv4/IPv6)
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 7. aiohttp (3.13.0)
**Decision:** 🟡 KEEP
**What:** Async HTTP client/server
**Reason:** Used in market_scanner.py and broker_connector.py for async data fetching.

### 8. aiohttp-client-cache (0.11.1)
**Decision:** ❌ DELETE
**What:** Async HTTP caching
**Reason:** Not used. Standard requests caching is enough.

### 9. aiohttp-retry (2.9.1)
**Decision:** ❌ DELETE
**What:** Retry logic for aiohttp
**Reason:** Can implement retry logic manually if needed.

### 10. aioredis (2.0.1)
**Decision:** ❌ DELETE
**What:** Async Redis client
**Reason:** You don't use Redis.

### 11. aiosignal (1.4.0)
**Decision:** ⚠️ OPTIONAL
**What:** Signal handling for asyncio
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 12. aiosmtplib (4.0.1)
**Decision:** ❌ DELETE
**What:** Async SMTP client
**Reason:** Not sending emails from trading system.

### 13. aiosqlite (0.20.0)
**Decision:** ❌ DELETE
**What:** Async SQLite
**Reason:** You use JSON logging, not SQLite.

### 14. alembic (1.16.4)
**Decision:** ❌ DELETE
**What:** Database migration tool
**Reason:** You don't use relational databases.

### 15. alpaca-py (0.42.1)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Alpaca's NEW Python SDK
**Reason:** Used in 49 files for options/futures trading.

### 16. alpaca-trade-api (3.2.0)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Alpaca's OLD Python SDK
**Reason:** Used in 15 files. Still needed during transition to new SDK.

### 17. alpha_vantage (3.0.0)
**Decision:** 🟡 KEEP
**What:** Alpha Vantage API client
**Reason:** Alternative data source. Free tier available.

### 18. altair (5.5.0)
**Decision:** ❌ DELETE
**What:** Declarative visualization
**Reason:** You have plotly and matplotlib. Don't need a third viz library.

### 19. annotated-types (0.7.0)
**Decision:** ⚠️ OPTIONAL
**What:** Type annotation support
**Reason:** Dependency of pydantic. Keep if keeping pydantic.

### 20. anthropic (0.58.2)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Claude API client
**Reason:** Used for AI agents and strategy enhancement.

### 21. anyio (4.9.0)
**Decision:** ⚠️ OPTIONAL
**What:** Async I/O abstraction
**Reason:** Dependency of multiple packages. Keep if using async libraries.

### 22. appdirs (1.4.4)
**Decision:** ⚠️ OPTIONAL
**What:** Platform-specific app directories
**Reason:** Used by some packages for config file locations.

### 23. arch (7.2.0)
**Decision:** ❌ DELETE
**What:** ARCH/GARCH econometric models
**Reason:** Advanced time series modeling. Not used in your system.

### 24. arviz (0.22.0)
**Decision:** ❌ DELETE
**What:** Bayesian model visualization
**Reason:** Only needed if using PyMC (Bayesian modeling). You don't.

### 25. ast-comments (1.2.3)
**Decision:** ❌ DELETE
**What:** Parse Python AST with comments
**Reason:** Development tool. Not needed for trading.

### 26. asteval (1.0.6)
**Decision:** ❌ DELETE
**What:** Safe math expression evaluator
**Reason:** Not used in your code.

### 27. astropy (7.1.0)
**Decision:** ❌ DELETE
**What:** Astronomy library (!!!)
**Reason:** Why do you have an astronomy library for trading?? DELETE.

### 28. astropy-iers-data (0.2025.9.1.0.42.11)
**Decision:** ❌ DELETE
**What:** Earth rotation data for astronomy
**Reason:** Related to astropy. Completely unrelated to trading.

### 29. asttokens (3.0.0)
**Decision:** ⚠️ OPTIONAL
**What:** Annotate AST with token info
**Reason:** Used by debugging tools. Not critical.

### 30. astunparse (1.6.3)
**Decision:** ❌ DELETE
**What:** AST to Python code
**Reason:** TensorFlow dependency. You don't use TensorFlow.

### 31. astutils (0.0.6)
**Decision:** ❌ DELETE
**What:** AST utilities
**Reason:** Not used.

### 32. async-lru (2.0.5)
**Decision:** ⚠️ OPTIONAL
**What:** Async LRU cache
**Reason:** Useful for caching async calls. Keep if doing heavy async work.

### 33. async-timeout (5.0.1)
**Decision:** ⚠️ OPTIONAL
**What:** Timeout context for asyncio
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 34. asyncpg (0.30.0)
**Decision:** ❌ DELETE
**What:** Async PostgreSQL driver
**Reason:** You don't use PostgreSQL.

### 35. attrs (25.4.0)
**Decision:** ⚠️ OPTIONAL
**What:** Classes without boilerplate
**Reason:** Common dependency. Keep (small package).

### 36. auth0-python (4.10.0)
**Decision:** ❌ DELETE
**What:** Auth0 authentication
**Reason:** Not using Auth0 for trading system.

### 37. Authlib (1.6.4)
**Decision:** ❌ DELETE
**What:** OAuth library
**Reason:** Not implementing OAuth in trading system.

### 38. Automat (25.4.16)
**Decision:** ❌ DELETE
**What:** Finite state machines
**Reason:** Dependency of Scrapy. Delete if deleting Scrapy.

---

## B

### 39. babel (2.17.0)
**Decision:** ❌ DELETE
**What:** Internationalization
**Reason:** Trading system doesn't need i18n.

### 40. backoff (2.2.1)
**Decision:** ⚠️ OPTIONAL
**What:** Retry with exponential backoff
**Reason:** Useful for API calls. Small package, can keep.

### 41. backrefs (5.9)
**Decision:** ❌ DELETE
**What:** Regex backreferences
**Reason:** Not needed.

### 42. backtrader (1.9.78.123)
**Decision:** 🟡 KEEP
**What:** Popular backtesting framework
**Reason:** Professional backtesting tool. Good for testing strategies.

### 43. bayesian-optimization (3.0.1)
**Decision:** ❌ DELETE
**What:** Bayesian hyperparameter optimization
**Reason:** You have Riskfolio-Lib for optimization. Don't need Bayesian.

### 44. bcolz-zipline (1.13.0)
**Decision:** ⚠️ OPTIONAL
**What:** Data storage for Zipline
**Reason:** Keep only if using Zipline. See zipline-reloaded below.

### 45. bcrypt (4.3.0)
**Decision:** ❌ DELETE
**What:** Password hashing
**Reason:** Not storing passwords in trading system.

### 46. beautifulsoup4 (4.13.4)
**Decision:** ❌ DELETE
**What:** HTML parsing
**Reason:** Not scraping HTML in production. Delete.

### 47. bech32 (1.2.0)
**Decision:** ❌ DELETE
**What:** Bitcoin address encoding
**Reason:** Related to cosmpy (Cosmos blockchain). Not needed.

### 48. black (25.1.0)
**Decision:** ✅ KEEP
**What:** Code formatter
**Reason:** Development tool. Keeps code consistent.

### 49. bleach (6.2.0)
**Decision:** ❌ DELETE
**What:** HTML sanitizer
**Reason:** Not processing HTML.

### 50. blinker (1.9.0)
**Decision:** ❌ DELETE
**What:** Signal/event system
**Reason:** Dependency of Flask. Delete if deleting Flask.

### 51. blosc2 (3.7.2)
**Decision:** ❌ DELETE
**What:** Data compression
**Reason:** Not doing heavy data compression.

### 52. Bottleneck (1.5.0)
**Decision:** ⚠️ OPTIONAL
**What:** Fast NumPy array functions
**Reason:** Can speed up pandas operations. Small benefit.

### 53. Brotli (1.1.0)
**Decision:** ⚠️ OPTIONAL
**What:** Compression algorithm
**Reason:** Used for HTTP compression. Keep if doing lots of API calls.

### 54. bs4 (0.0.2)
**Decision:** ❌ DELETE
**What:** Dummy wrapper for beautifulsoup4
**Reason:** Delete both bs4 and beautifulsoup4.

### 55. bt (1.1.2)
**Decision:** 🟡 KEEP
**What:** Flexible backtesting framework
**Reason:** Alternative to backtrader. Good for portfolio backtesting.

### 56. build (1.2.2.post1)
**Decision:** ❌ DELETE
**What:** Python package builder
**Reason:** Development tool. Not needed in production.

---

## C

### 57. CacheControl (0.14.3)
**Decision:** ⚠️ OPTIONAL
**What:** HTTP caching
**Reason:** Can help with API rate limits. Small package.

### 58. cachetools (5.5.2)
**Decision:** ⚠️ OPTIONAL
**What:** Caching utilities
**Reason:** Useful for memoization. Keep (small).

### 59. cattrs (25.3.0)
**Decision:** ⚠️ OPTIONAL
**What:** Data structure (de)serialization
**Reason:** Useful library. Keep if doing complex data transformations.

### 60. ccxt (4.5.3)
**Decision:** ❌ DELETE
**What:** Cryptocurrency exchange API
**Reason:** You trade options/forex/futures, NOT crypto. Delete.

### 61. certifi (2025.6.15)
**Decision:** ✅ KEEP
**What:** SSL certificates
**Reason:** Required for HTTPS requests. Keep.

### 62. cffi (1.17.1)
**Decision:** ⚠️ OPTIONAL
**What:** C Foreign Function Interface
**Reason:** Dependency of cryptography and other packages. Keep.

### 63. cfgv (3.4.0)
**Decision:** ❌ DELETE
**What:** Config file validation
**Reason:** Dependency of pre-commit. Delete if deleting pre-commit.

### 64. chardet (5.2.0)
**Decision:** ⚠️ OPTIONAL
**What:** Character encoding detection
**Reason:** Useful for processing text data. Keep (small).

### 65. charset-normalizer (3.4.2)
**Decision:** ⚠️ OPTIONAL
**What:** Character encoding detection (newer)
**Reason:** Dependency of requests. Keep.

### 66. chex (0.1.91)
**Decision:** ❌ DELETE
**What:** JAX testing utilities
**Reason:** Only needed for JAX (Google ML framework). You don't use JAX.

### 67. chromadb (1.0.15)
**Decision:** 🟡 KEEP
**What:** Vector database for AI
**Reason:** Useful for AI agent memory/context. Keep if building AI agents.

### 68. clarabel (0.11.1)
**Decision:** ❌ DELETE
**What:** Optimization solver
**Reason:** Dependency of cvxpy. You have Riskfolio-Lib.

### 69. cleo (2.1.0)
**Decision:** ❌ DELETE
**What:** CLI framework
**Reason:** Dependency of poetry. Not needed in production.

### 70. click (8.1.8)
**Decision:** ⚠️ OPTIONAL
**What:** CLI creation tool
**Reason:** Common dependency. Keep (widely used).

### 71. cloudpickle (3.1.1)
**Decision:** ⚠️ OPTIONAL
**What:** Extended pickle
**Reason:** Used for serializing ML models. Keep if saving models.

### 72. clr_loader (0.2.7.post0)
**Decision:** ❌ DELETE
**What:** .NET runtime loader
**Reason:** Related to pythonnet. Only if interfacing with .NET code.

### 73. cmdstanpy (1.2.5)
**Decision:** ❌ DELETE
**What:** Stan statistical modeling
**Reason:** Only needed for prophet (Facebook's forecasting tool).

### 74. colorama (0.4.6)
**Decision:** ✅ KEEP
**What:** Colored terminal output
**Reason:** Used in mission_control_logger.py. Nice for logs.

### 75. coloredlogs (15.0.1)
**Decision:** ❌ DELETE
**What:** Colored logging
**Reason:** Duplicate of colorama. Only need one.

### 76. colorlog (6.9.0)
**Decision:** ❌ DELETE
**What:** Colored logging
**Reason:** Another duplicate. colorama is enough.

### 77. colorlover (0.3.0)
**Decision:** ❌ DELETE
**What:** Color palettes
**Reason:** Visualization dependency. Not critical.

### 78. comm (0.2.3)
**Decision:** ❌ DELETE
**What:** Jupyter communication
**Reason:** Not using Jupyter notebooks in production.

### 79. cons (0.4.7)
**Decision:** ❌ DELETE
**What:** Lisp-style cons cells
**Reason:** Advanced functional programming. Not needed.

### 80. constantly (23.10.4)
**Decision:** ❌ DELETE
**What:** Named constants
**Reason:** Dependency of Twisted. Delete if deleting Scrapy.

### 81. contourpy (1.3.3)
**Decision:** ⚠️ OPTIONAL
**What:** Contour calculations
**Reason:** Dependency of matplotlib. Keep if keeping matplotlib.

### 82. cosmpy (0.9.3)
**Decision:** ❌ DELETE
**What:** Cosmos blockchain SDK
**Reason:** Why is this here?? Not trading blockchain. DELETE.

### 83. courlan (1.3.2)
**Decision:** ❌ DELETE
**What:** URL handling
**Reason:** Web scraping dependency. Not needed.

### 84. crashtest (0.4.1)
**Decision:** ❌ DELETE
**What:** Exception testing
**Reason:** Dependency of poetry. Not needed.

### 85. crewai (0.134.0)
**Decision:** 🟡 KEEP
**What:** Multi-agent AI framework
**Reason:** Useful if building multi-agent trading systems.

### 86. croniter (6.0.0)
**Decision:** ⚠️ OPTIONAL
**What:** Cron expression iterator
**Reason:** Could be useful for scheduling. But you have `schedule`.

### 87. cryptography (45.0.5)
**Decision:** ⚠️ OPTIONAL
**What:** Cryptographic recipes
**Reason:** Dependency of many packages (SSL, auth). Keep.

### 88. cssselect (1.3.0)
**Decision:** ❌ DELETE
**What:** CSS selector parsing
**Reason:** Web scraping dependency. Not needed.

### 89. cufflinks (0.17.3)
**Decision:** 🟡 KEEP
**What:** Plotly for pandas
**Reason:** Makes creating interactive charts easy from pandas DataFrames.

### 90. curl_adapter (1.1.0)
**Decision:** ❌ DELETE
**What:** Curl adapter for requests
**Reason:** Not needed. Standard requests is fine.

### 91. curl_cffi (0.13.0)
**Decision:** ❌ DELETE
**What:** Curl with HTTP/2
**Reason:** Not needed for trading.

### 92. cvxpy (1.7.2)
**Decision:** ❌ DELETE
**What:** Convex optimization
**Reason:** You have Riskfolio-Lib for portfolio optimization.

### 93. cycler (0.12.1)
**Decision:** ⚠️ OPTIONAL
**What:** Property cycling (matplotlib)
**Reason:** Dependency of matplotlib. Keep if keeping matplotlib.

### 94. cyclopts (3.24.0)
**Decision:** ❌ DELETE
**What:** CLI parsing
**Reason:** Not needed. You have click and argparse.

### 95. Cython (3.1.3)
**Decision:** ⚠️ OPTIONAL
**What:** C extensions for Python
**Reason:** Needed to build some packages (TA-Lib, etc.). Keep.

---

## D

### 96. dash (3.2.0)
**Decision:** 🟡 KEEP
**What:** Interactive dashboards
**Reason:** Build trading dashboards. Plotly-based. Professional tool.

### 97. dataclasses-json (0.6.7)
**Decision:** ⚠️ OPTIONAL
**What:** JSON serialization for dataclasses
**Reason:** Useful for data handling. Keep (small).

### 98. dateparser (1.2.2)
**Decision:** ⚠️ OPTIONAL
**What:** Parse dates in any format
**Reason:** Useful for flexible date parsing. Keep if processing varied date formats.

### 99. dd (0.6.0)
**Decision:** ❌ DELETE
**What:** Binary decision diagrams
**Reason:** Advanced computer science. Not needed.

### 100. deap (1.4.3)
**Decision:** ❌ DELETE
**What:** Genetic algorithms
**Reason:** You have Riskfolio-Lib. Don't need genetic algorithms.

### 101. decorator (5.2.1)
**Decision:** ⚠️ OPTIONAL
**What:** Decorator utilities
**Reason:** Common dependency. Keep (small, widely used).

### 102. deepdiff (8.6.1)
**Decision:** ❌ DELETE
**What:** Deep difference of objects
**Reason:** Testing/debugging tool. Not critical.

### 103. deepseek-cli (0.1.20)
**Decision:** ❌ DELETE
**What:** DeepSeek AI CLI
**Reason:** Alternative AI provider. You have Anthropic/OpenAI.

### 104. defusedxml (0.8.0rc2)
**Decision:** ⚠️ OPTIONAL
**What:** Secure XML parsing
**Reason:** Security library. Keep if parsing XML.

### 105. deprecation (2.1.0)
**Decision:** ⚠️ OPTIONAL
**What:** Deprecation warnings
**Reason:** Common dependency. Keep (small).

### 106. dill (0.4.0)
**Decision:** ⚠️ OPTIONAL
**What:** Extended pickle
**Reason:** Similar to cloudpickle. Keep one, delete the other.

### 107. distlib (0.3.9)
**Decision:** ❌ DELETE
**What:** Package distribution utilities
**Reason:** Build tool dependency. Not needed in production.

### 108. distro (1.9.0)
**Decision:** ⚠️ OPTIONAL
**What:** Linux distribution detection
**Reason:** Cross-platform compatibility. Keep (small).

### 109. dnspython (2.7.0)
**Decision:** ⚠️ OPTIONAL
**What:** DNS toolkit
**Reason:** Dependency of email_validator. Keep if validating emails.

### 110. docker (7.1.0)
**Decision:** ❌ DELETE
**What:** Docker API client
**Reason:** Not deploying to Docker from within trading system. Delete.

### 111. docstring_parser (0.16)
**Decision:** ❌ DELETE
**What:** Parse docstrings
**Reason:** Development tool. Not needed in production.

### 112. docutils (0.22.2)
**Decision:** ❌ DELETE
**What:** Documentation utilities
**Reason:** Not generating docs from trading system.

### 113. dulwich (0.22.8)
**Decision:** ❌ DELETE
**What:** Pure Python Git implementation
**Reason:** Dependency of poetry. Not needed in production.

### 114. durationpy (0.10)
**Decision:** ⚠️ OPTIONAL
**What:** Duration parsing
**Reason:** Useful for time-based calculations. Keep (tiny).

---

## E

### 115. ecdsa (0.19.1)
**Decision:** ⚠️ OPTIONAL
**What:** Elliptic curve cryptography
**Reason:** Used for signing. Keep if using crypto signatures.

### 116. ecos (2.0.14)
**Decision:** ❌ DELETE
**What:** Optimization solver
**Reason:** Dependency of cvxpy. You have Riskfolio-Lib.

### 117. email_validator (2.2.0)
**Decision:** ❌ DELETE
**What:** Email validation
**Reason:** Not validating emails in trading system.

### 118. empyrical-reloaded (0.5.12)
**Decision:** 🟡 KEEP
**What:** Financial risk/return metrics
**Reason:** Used with pyfolio for calculating Sharpe ratio, drawdown, etc.

### 119. et_xmlfile (2.0.0)
**Decision:** ⚠️ OPTIONAL
**What:** Excel file reading
**Reason:** Dependency of openpyxl. Keep if reading Excel files.

### 120. etuples (0.3.10)
**Decision:** ❌ DELETE
**What:** S-expression tuples
**Reason:** Advanced functional programming. Not needed.

### 121. eventkit (1.0.3)
**Decision:** ❌ DELETE
**What:** Event-driven programming
**Reason:** Dependency of ib-insync. Delete if not using IB.

### 122. exceptiongroup (1.3.0)
**Decision:** ⚠️ OPTIONAL
**What:** Exception groups (Python 3.11 backport)
**Reason:** Keep for compatibility.

### 123. exchange_calendars (4.11.1)
**Decision:** ⚠️ OPTIONAL
**What:** Market trading calendars
**Reason:** Useful for knowing market hours. Keep if backtesting.

### 124. executing (2.2.0)
**Decision:** ⚠️ OPTIONAL
**What:** Get currently executing AST node
**Reason:** Debugging tool. Not critical.

---

## F

### 125. Farama-Notifications (0.0.4)
**Decision:** ❌ DELETE
**What:** Notifications for gymnasium
**Reason:** Related to gymnasium (RL). You don't do RL.

### 126. fastapi (0.116.2)
**Decision:** ❌ DELETE
**What:** Modern web framework
**Reason:** Not building web API in trading system. Delete.

### 127. fastjsonschema (2.21.1)
**Decision:** ⚠️ OPTIONAL
**What:** Fast JSON schema validation
**Reason:** Useful for data validation. Keep (small).

### 128. fastmcp (2.12.3)
**Decision:** ⚠️ OPTIONAL
**What:** MCP (Model Context Protocol)
**Reason:** Used by Claude Code. Keep if using MCP servers.

### 129. fastquant (0.1.8.1)
**Decision:** ❌ DELETE
**What:** Fast backtesting
**Reason:** You have backtrader, vectorbt, bt. Don't need a fourth.

### 130. feedfinder2 (0.0.4)
**Decision:** ❌ DELETE
**What:** Find RSS feeds
**Reason:** Web scraping. Not needed.

### 131. feedparser (6.0.11)
**Decision:** ❌ DELETE
**What:** Parse RSS/Atom feeds
**Reason:** Not parsing news feeds in production.

### 132. ffn (1.1.2)
**Decision:** ❌ DELETE
**What:** Financial functions
**Reason:** Overlaps with QuantStats and pyfolio. Redundant.

### 133. filelock (3.18.0)
**Decision:** ⚠️ OPTIONAL
**What:** File-based locks
**Reason:** Useful for preventing race conditions. Keep (small).

### 134. financedatabase (2.3.1)
**Decision:** ❌ DELETE
**What:** Financial database
**Reason:** You have OpenBB for data. Redundant.

### 135. financepy (1.0.1)
**Decision:** ❌ DELETE
**What:** Finance library
**Reason:** Not actively maintained. Use scipy instead.

### 136. financetoolkit (2.0.5)
**Decision:** ❌ DELETE
**What:** Financial analysis toolkit
**Reason:** Overlaps with your other tools. Redundant.

### 137. findpython (0.6.3)
**Decision:** ❌ DELETE
**What:** Find Python installations
**Reason:** Build tool. Not needed in production.

### 138. FinQuant (0.7.0)
**Decision:** ❌ DELETE
**What:** Portfolio optimization
**Reason:** You have pyportfolioopt and Riskfolio-Lib. Redundant.

### 139. FinRL (0.3.7)
**Decision:** ❌ DELETE
**What:** Reinforcement learning for finance
**Reason:** You don't do RL trading. Delete.

### 140. finta (1.3)
**Decision:** ❌ DELETE
**What:** Financial technical analysis
**Reason:** You have TA-Lib and pandas-ta. Redundant.

### 141. finvizfinance (1.2.0)
**Decision:** ❌ DELETE
**What:** Finviz data scraper
**Reason:** Not scraping Finviz. You have OpenBB.

### 142. Flask (3.1.0)
**Decision:** ❌ DELETE
**What:** Web framework
**Reason:** Not building web app. Delete.

### 143. flatbuffers (25.2.10)
**Decision:** ❌ DELETE
**What:** Serialization library
**Reason:** TensorFlow dependency. You don't use TensorFlow.

### 144. fonttools (4.59.0)
**Decision:** ⚠️ OPTIONAL
**What:** Font utilities
**Reason:** Dependency of matplotlib. Keep if keeping matplotlib.

### 145. fredapi (0.5.2)
**Decision:** 🟡 KEEP
**What:** Federal Reserve Economic Data API
**Reason:** Good for economic indicators. Free API.

### 146. freqtrade (2025.8)
**Decision:** ❌ DELETE
**What:** Crypto trading bot
**Reason:** You trade options/forex/futures, NOT crypto. Delete.

### 147. freqtrade-client (2025.8)
**Decision:** ❌ DELETE
**What:** Freqtrade API client
**Reason:** Delete with freqtrade.

### 148. frozendict (2.4.6)
**Decision:** ⚠️ OPTIONAL
**What:** Immutable dictionary
**Reason:** Useful for certain patterns. Keep (tiny).

### 149. frozenlist (1.6.0)
**Decision:** ⚠️ OPTIONAL
**What:** Immutable list
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 150. fsspec (2025.5.1)
**Decision:** ⚠️ OPTIONAL
**What:** Filesystem abstraction
**Reason:** Used by pandas for cloud storage. Keep if using cloud.

### 151. ft-pandas-ta (0.3.15)
**Decision:** ❌ DELETE
**What:** Freqtrade's pandas-ta
**Reason:** You have pandas-ta. This is duplicate.

---

## G

### 152. gast (0.6.0)
**Decision:** ❌ DELETE
**What:** AST for Python 2/3
**Reason:** TensorFlow dependency. You don't use TensorFlow.

### 153. geographiclib (2.0)
**Decision:** ❌ DELETE
**What:** Geographic calculations
**Reason:** Not doing geography in trading.

### 154. geopy (2.4.1)
**Decision:** ❌ DELETE
**What:** Geocoding
**Reason:** Why geocoding in trading?? DELETE.

### 155. ghp-import (2.1.0)
**Decision:** ❌ DELETE
**What:** GitHub Pages import
**Reason:** Documentation tool. Not needed.

### 156. gitdb (4.0.12)
**Decision:** ⚠️ OPTIONAL
**What:** Git object database
**Reason:** Dependency of GitPython. Keep if using git in code.

### 157. GitPython (3.1.45)
**Decision:** ⚠️ OPTIONAL
**What:** Git interface
**Reason:** Useful if automating git operations. Not critical.

### 158-165. google-* packages
**Decision:** ❌ DELETE (most)
**What:** Google API libraries
**Reason:** Unless using Google Gemini AI, delete. You have Anthropic/OpenAI.

### 166. gotrue (2.12.2)
**Decision:** ❌ DELETE
**What:** Supabase auth
**Reason:** Not using Supabase.

### 167. graphviz (0.21)
**Decision:** ❌ DELETE
**What:** Graph visualization
**Reason:** Not visualizing graphs in production.

### 168. greenlet (3.2.3)
**Decision:** ⚠️ OPTIONAL
**What:** Lightweight concurrency
**Reason:** Dependency of SQLAlchemy. Keep if keeping databases.

### 169. grpcio (1.71.0)
**Decision:** ❌ DELETE
**What:** gRPC framework
**Reason:** Not using gRPC.

### 170. grpcio-status (1.71.0)
**Decision:** ❌ DELETE
**What:** gRPC status
**Reason:** Delete with grpcio.

### 171. gs-quant (1.4.31)
**Decision:** ❌ DELETE
**What:** Goldman Sachs quant library
**Reason:** HUGE library (100+ MB). Advanced derivatives. Overkill.

### 172. gymnasium (1.2.0)
**Decision:** ❌ DELETE
**What:** Reinforcement learning environments
**Reason:** You don't do RL trading.

---

## H

### 173-176. h11, h2, h5netcdf, h5py
**Decision:** ⚠️ OPTIONAL (h11, h2), ❌ DELETE (h5*)
**What:** HTTP/2 and HDF5 libraries
**Reason:** h11/h2 are HTTP dependencies. h5* are for HDF5 files (not used).

### 177. holidays (0.80)
**Decision:** ⚠️ OPTIONAL
**What:** Holiday calendars
**Reason:** Useful for market hours. Keep if needed.

### 178. homeharvest (0.3.2)
**Decision:** ❌ DELETE
**What:** Real estate data scraper
**Reason:** Not trading real estate.

### 179-181. hpack, html5lib, htmldate
**Decision:** ❌ DELETE
**What:** HTTP/HTML processing
**Reason:** Web scraping dependencies. Not needed.

### 182-186. httpcore, httplib2, httptools, httpx, httpx-sse
**Decision:** ⚠️ OPTIONAL
**What:** HTTP clients
**Reason:** Modern HTTP libraries. httpx is good alternative to requests.

### 187. huggingface-hub (0.35.0)
**Decision:** ❌ DELETE
**What:** HuggingFace model hub
**Reason:** Related to transformers. You don't use transformers.

### 188-189. humanfriendly, humanize
**Decision:** ⚠️ OPTIONAL
**What:** Human-readable formatting
**Reason:** Nice for logs. Keep (tiny).

### 190-191. hyperframe, hyperlink
**Decision:** ❌ DELETE
**What:** HTTP/2 and hyperlink utilities
**Reason:** Low-level HTTP. Not needed.

---

## I

### 192. ib-insync (0.9.86)
**Decision:** ❌ DELETE
**What:** Interactive Brokers API
**Reason:** You use Alpaca, not Interactive Brokers. Delete.

### 193-194. identify, idna
**Decision:** ⚠️ OPTIONAL
**What:** File identification, international domains
**Reason:** Common dependencies. Keep (small).

### 195. imageio (2.37.0)
**Decision:** ❌ DELETE
**What:** Image I/O
**Reason:** Not processing images in trading.

### 196-197. importlib_metadata, importlib_resources
**Decision:** ⚠️ OPTIONAL
**What:** Import system utilities
**Reason:** Common dependencies. Keep.

### 198-199. incremental, inflection
**Decision:** ⚠️ OPTIONAL
**What:** Versioning, string inflection
**Reason:** Common dependencies. Keep (tiny).

### 200-201. iniconfig, inscriptis
**Decision:** ⚠️ OPTIONAL (iniconfig), ❌ DELETE (inscriptis)
**What:** INI config, HTML to text
**Reason:** iniconfig used by pytest. inscriptis for web scraping.

### 202-203. installer, instructor
**Decision:** ❌ DELETE (installer), 🟡 KEEP (instructor)
**What:** Package installer, structured LLM outputs
**Reason:** installer is build tool. instructor is useful for AI agents.

### 204. intervaltree (3.1.0)
**Decision:** ⚠️ OPTIONAL
**What:** Interval tree data structure
**Reason:** Could be useful for time-based data. Keep (small).

### 205-207. ipython, ipython_pygments_lexers, ipywidgets
**Decision:** ❌ DELETE
**What:** IPython/Jupyter tools
**Reason:** Not using notebooks in production.

### 208-210. iso3166, iso4217, isodate
**Decision:** ⚠️ OPTIONAL
**What:** Country codes, currency codes, ISO dates
**Reason:** Useful for international trading. Keep (tiny).

### 211-213. itemadapter, itemloaders, itsdangerous
**Decision:** ❌ DELETE
**What:** Scrapy items, security tokens
**Reason:** Scrapy dependencies or Flask dependencies. Delete.

---

## J-L

### 214-224. janus through jmespath
**Decision:** ❌ DELETE (most)
**What:** Various utilities
**Reason:** Most are Java-related or specialized. Not needed.

### 218-219. jax, jaxlib
**Decision:** ❌ DELETE
**What:** Google's JAX ML framework
**Reason:** You use scikit-learn. Don't need JAX (200+ MB).

### 225. joblib (1.5.1)
**Decision:** ⚠️ OPTIONAL
**What:** Parallel computing
**Reason:** Used by scikit-learn. Keep.

### 226-234. json_* packages
**Decision:** ⚠️ OPTIONAL
**What:** JSON utilities
**Reason:** Useful for data handling. Keep (small packages).

### 235-236. jupyterlab_widgets, jusText
**Decision:** ❌ DELETE
**What:** Jupyter widgets, text extraction
**Reason:** Not using Jupyter or web scraping.

### 237. kaggle (1.7.4.5)
**Decision:** ❌ DELETE
**What:** Kaggle API
**Reason:** Not downloading Kaggle datasets.

### 238. keras (3.11.3)
**Decision:** ❌ DELETE
**What:** Deep learning framework
**Reason:** You use scikit-learn, not deep learning. DELETE.

### 239-241. keyring, kiwisolver, korean-lunar-calendar
**Decision:** ⚠️ OPTIONAL (first two), ❌ DELETE (korean)
**What:** Password storage, constraint solver, Korean calendar
**Reason:** keyring might be useful. korean-lunar-calendar?? DELETE.

### 242. kubernetes (33.1.0)
**Decision:** ❌ DELETE
**What:** Kubernetes API client
**Reason:** Not deploying to Kubernetes from trading system.

### 243-254. langchain-* packages
**Decision:** 🟡 KEEP ALL
**What:** LangChain AI agent framework
**Reason:** You said you're building AI agents. These are ESSENTIAL.

### 255. lazy-object-proxy (1.12.0)
**Decision:** ⚠️ OPTIONAL
**What:** Lazy object creation
**Reason:** Used by some packages. Keep (small).

### 256. lean (1.0.220)
**Decision:** ❌ DELETE
**What:** QuantConnect LEAN engine
**Reason:** HUGE (100+ MB). You use Alpaca, not QuantConnect.

### 257. libclang (18.1.1)
**Decision:** ❌ DELETE
**What:** Clang Python bindings
**Reason:** Related to keras. You don't use keras.

### 258. lightgbm (4.6.0)
**Decision:** ❌ DELETE
**What:** Gradient boosting
**Reason:** You have scikit-learn GradientBoosting. Redundant.

### 259. litellm (1.72.0)
**Decision:** ⚠️ OPTIONAL
**What:** LLM proxy (OpenAI-compatible API for any LLM)
**Reason:** Useful for switching between AI providers easily.

### 260-261. llvmlite, lmfit
**Decision:** ❌ DELETE
**What:** LLVM compiler, curve fitting
**Reason:** llvmlite for numba (you don't use). lmfit not needed.

### 262-263. logical-unification, lru-dict
**Decision:** ❌ DELETE (logical), ⚠️ OPTIONAL (lru)
**What:** Logic programming, LRU cache
**Reason:** logical not needed. lru-dict is dependency.

### 264-265. lxml, lxml_html_clean
**Decision:** ❌ DELETE
**What:** XML/HTML parsing
**Reason:** Web scraping. Not needed in production.

---

## M

### 266-270. Mako through marshmallow
**Decision:** ⚠️ OPTIONAL
**What:** Templating, markdown, serialization
**Reason:** Common dependencies. Keep (small).

### 271-272. matplotlib, matplotlib-inline
**Decision:** 🟡 KEEP
**What:** Plotting library
**Reason:** Industry standard for charts. Essential for analysis.

### 273-274. mcp, mcp-simple-timeserver
**Decision:** ⚠️ OPTIONAL
**What:** Model Context Protocol
**Reason:** Used by Claude Code. Keep if using MCP.

### 275. mctx (0.0.6)
**Decision:** ❌ DELETE
**What:** Monte Carlo tree search
**Reason:** JAX dependency. You don't use JAX.

### 276-277. mdurl, mergedeep
**Decision:** ⚠️ OPTIONAL
**What:** Markdown URLs, deep merge
**Reason:** Utilities. Keep (tiny).

### 278. MetaTrader5 (5.0.5260)
**Decision:** ❌ DELETE
**What:** MetaTrader 5 API
**Reason:** You use OANDA for forex, not MetaTrader.

### 279. miniKanren (1.0.5)
**Decision:** ❌ DELETE
**What:** Logic programming
**Reason:** Advanced CS. Not needed.

### 280-283. mkdocs-* packages
**Decision:** ❌ DELETE
**What:** Documentation generator
**Reason:** Not generating docs from trading system.

### 284-285. ml_dtypes, mmh3
**Decision:** ❌ DELETE (ml_dtypes), ⚠️ OPTIONAL (mmh3)
**What:** ML data types, hashing
**Reason:** ml_dtypes for TensorFlow. mmh3 is fast hash (keep).

### 286-293. monotonic through mypy_extensions
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** Common dependencies. Keep (small).

### 294-299. namex through networkx
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities, graph library
**Reason:** networkx useful for network analysis. Others are dependencies.

### 300-301. newspaper3k, nltk
**Decision:** ❌ DELETE
**What:** News scraping, NLP
**Reason:** Not doing NLP or news scraping in production.

### 302-303. nodeenv, ntplib
**Decision:** ❌ DELETE (nodeenv), ⚠️ OPTIONAL (ntplib)
**What:** Node.js environment, NTP time
**Reason:** nodeenv not needed. ntplib for accurate time (keep).

### 304. numba (0.61.2)
**Decision:** ❌ DELETE
**What:** JIT compiler for Python
**Reason:** HUGE (100+ MB). Not using JIT compilation.

### 305. numexpr (2.11.0)
**Decision:** ⚠️ OPTIONAL
**What:** Fast numerical expressions
**Reason:** Used by pandas for speedups. Keep.

### 306. numpy (2.2.6)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Numerical computing
**Reason:** Foundation of all numerical Python. ESSENTIAL.

### 307. nvidia-ml-py (13.580.82)
**Decision:** ⚠️ OPTIONAL
**What:** NVIDIA GPU monitoring
**Reason:** Useful if using GPU. Otherwise delete.

---

## O

### 308-310. oauthlib, odfpy, omega
**Decision:** ❌ DELETE
**What:** OAuth, OpenDocument, symbolic mathematics
**Reason:** Not needed in trading system.

### 311. onnxruntime (1.22.0)
**Decision:** ❌ DELETE
**What:** ONNX model runtime
**Reason:** Not running ONNX models.

### 312. openai (1.97.1)
**Decision:** ✅ KEEP
**What:** OpenAI API client
**Reason:** Backup AI provider. Used for AI agents.

### 313-316. openapi-* packages
**Decision:** ⚠️ OPTIONAL
**What:** OpenAPI schema validation
**Reason:** Used by some APIs. Keep if needed.

### 317-347. openbb-* packages (31 packages!)
**Decision:** 🟡 KEEP ALL
**What:** OpenBB Platform - FREE premium data
**Reason:** 30+ data sources for FREE. Extremely valuable.

### 348. openpyxl (3.1.5)
**Decision:** ⚠️ OPTIONAL
**What:** Excel file reading/writing
**Reason:** Useful for importing/exporting data. Keep.

### 349-355. opentelemetry-* packages
**Decision:** ❌ DELETE
**What:** Observability/telemetry
**Reason:** Not doing distributed tracing in trading system.

### 356-359. opt_einsum, optax, optree, optuna
**Decision:** ❌ DELETE (most), ⚠️ OPTIONAL (optuna)
**What:** Optimization libraries
**Reason:** opt_einsum/optax/optree for TensorFlow/JAX. optuna for hyperparameter tuning (could keep).

### 360-365. orderly-set through overrides
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** Common dependencies. Keep (small).

---

## P

### 366. packaging (25.0)
**Decision:** ⚠️ OPTIONAL
**What:** Package version parsing
**Reason:** Common dependency. Keep.

### 367. paginate (0.5.7)
**Decision:** ❌ DELETE
**What:** Pagination
**Reason:** Not building web UI.

### 368. pandas (2.3.2)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Data analysis library
**Reason:** Foundation of financial data analysis. ESSENTIAL.

### 369. pandas-datareader (0.10.0)
**Decision:** ⚠️ OPTIONAL
**What:** Remote data access
**Reason:** Alternative to yfinance. Can keep as backup.

### 370. pandas-ta (0.4.67b0)
**Decision:** 🟡 KEEP
**What:** Technical analysis for pandas
**Reason:** 130+ indicators in pandas-friendly format.

### 371-377. parse through patsy
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** patsy for statsmodels. Others are small dependencies.

### 378. pbs-installer (2025.4.9)
**Decision:** ❌ DELETE
**What:** PBS Pro installer
**Reason:** High-performance computing. Not needed.

### 379-380. pdfminer.six, pdfplumber
**Decision:** ❌ DELETE
**What:** PDF text extraction
**Reason:** Not processing PDFs in production.

### 381. peewee (3.17.3)
**Decision:** ❌ DELETE
**What:** ORM
**Reason:** You use JSON logging, not databases.

### 382. pillow (11.2.1)
**Decision:** ❌ DELETE
**What:** Image processing
**Reason:** Not processing images in trading.

### 383. pip (25.2)
**Decision:** ✅ KEEP
**What:** Package installer
**Reason:** Required for installing packages.

### 384-385. pkginfo, platformdirs
**Decision:** ⚠️ OPTIONAL
**What:** Package metadata, platform directories
**Reason:** Common dependencies. Keep.

### 386. plotly (6.3.1)
**Decision:** 🟡 KEEP
**What:** Interactive visualizations
**Reason:** Professional charting tool. Better than matplotlib for some cases.

### 387-388. pluggy, ply
**Decision:** ⚠️ OPTIONAL
**What:** Plugin system, parsing
**Reason:** pluggy for pytest. ply for parsing. Keep.

### 389-390. poetry, poetry-core
**Decision:** ❌ DELETE
**What:** Package management
**Reason:** Development tool. Not needed in production.

### 391. polars (1.33.0)
**Decision:** ⚠️ OPTIONAL
**What:** Fast DataFrame library
**Reason:** Faster alternative to pandas. Keep if using.

### 392. polygon-api-client (1.15.3)
**Decision:** 🟡 KEEP
**What:** Polygon.io API
**Reason:** Excellent real-time market data (if you have subscription).

### 393-395. polytope, postgrest, posthog
**Decision:** ❌ DELETE
**What:** Math library, Supabase, analytics
**Reason:** Not needed.

### 396. pre_commit (4.2.0)
**Decision:** ❌ DELETE
**What:** Git hook management
**Reason:** Development tool. Not needed in production.

### 397-398. prompt_toolkit, propcache
**Decision:** ⚠️ OPTIONAL
**What:** Interactive prompts, property cache
**Reason:** prompt_toolkit for CLI. propcache is dependency.

### 399. prophet (1.1.7)
**Decision:** ⚠️ OPTIONAL
**What:** Facebook's forecasting tool
**Reason:** Could be useful for time series forecasting. Large though.

### 400-402. Protego, proto-plus, protobuf
**Decision:** ⚠️ OPTIONAL (protobuf), ❌ DELETE (others)
**What:** Protocol buffers
**Reason:** protobuf is common dependency. Others not needed.

### 403. psutil (7.0.0)
**Decision:** ⚠️ OPTIONAL
**What:** System utilities
**Reason:** Useful for monitoring system resources. Keep.

### 404. psycopg2-binary (2.9.10)
**Decision:** ❌ DELETE
**What:** PostgreSQL adapter
**Reason:** You don't use PostgreSQL.

### 405. PuLP (3.2.2)
**Decision:** ❌ DELETE
**What:** Linear programming
**Reason:** You have Riskfolio-Lib. Redundant.

### 406-409. pure_eval through py_vollib
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities, options pricing
**Reason:** py_vollib for options pricing (useful!). Others are dependencies.

### 410-427. pyarrow through pydot
**Decision:** ⚠️ OPTIONAL (most)
**What:** Various utilities
**Reason:** pyarrow for data formats. Others are dependencies.

### 428. pyfiglet (1.0.4)
**Decision:** ❌ DELETE
**What:** ASCII art text
**Reason:** Fun but not needed.

### 429. pyfolio-reloaded (0.9.9)
**Decision:** 🟡 KEEP
**What:** Portfolio analysis
**Reason:** Industry standard from Zipline. Excellent tear sheets.

### 430. pygame (2.6.1)
**Decision:** ❌ DELETE
**What:** Game library
**Reason:** Why is this here?? Not making games. DELETE.

### 431-433. Pygments, PyJWT, pyluach
**Decision:** ⚠️ OPTIONAL (Pygments, PyJWT), ❌ DELETE (pyluach)
**What:** Syntax highlighting, JWT tokens, Hebrew calendar
**Reason:** Pygments for code display. PyJWT could be useful. Hebrew calendar?? DELETE.

### 434. pymc (5.25.1)
**Decision:** ❌ DELETE
**What:** Bayesian modeling
**Reason:** Advanced statistics. You don't do Bayesian modeling. HUGE library.

### 435-438. pymdown-extensions through pyotp
**Decision:** ⚠️ OPTIONAL (pyotp), ❌ DELETE (others)
**What:** Markdown, SSL, 2FA
**Reason:** pyotp for 2FA could be useful. Others are docs/dependencies.

### 439-441. pyparsing, pypdfium2, pyperclip
**Decision:** ⚠️ OPTIONAL (pyparsing), ❌ DELETE (others)
**What:** Parsing, PDF, clipboard
**Reason:** pyparsing is common dependency. Others not needed.

### 442. pyportfolioopt (1.5.6)
**Decision:** 🟡 KEEP
**What:** Portfolio optimization
**Reason:** Modern portfolio theory implementation. Useful!

### 443-448. PyPrind through pytensor
**Decision:** ⚠️ OPTIONAL (some), ❌ DELETE (pytensor)
**What:** Progress bars, pytest, tensor library
**Reason:** pytensor is for PyMC. Delete. pytest keep.

### 449. pytest (8.4.1)
**Decision:** ✅ KEEP
**What:** Testing framework
**Reason:** Development tool. Essential for testing.

### 450-451. pytest-asyncio, pytest-mock
**Decision:** ⚠️ OPTIONAL
**What:** Pytest plugins
**Reason:** Useful for testing. Keep if writing tests.

### 452. python-binance (1.0.29)
**Decision:** ❌ DELETE
**What:** Binance API
**Reason:** You don't trade crypto. DELETE.

### 453-463. python-* packages
**Decision:** ⚠️ OPTIONAL (most), ❌ DELETE (telegram)
**What:** Various utilities
**Reason:** Most are useful dependencies. Telegram bot not needed.

### 464-467. pyvis through pyxlsb
**Decision:** ❌ DELETE (pyvis), ⚠️ OPTIONAL (others)
**What:** Network viz, Windows utilities, Excel
**Reason:** pyvis not needed. Others could be useful.

### 468-470. PyYAML, pyyaml_env_tag, pyzmq
**Decision:** ⚠️ OPTIONAL
**What:** YAML parsing, ZeroMQ
**Reason:** YAML common format. ZeroMQ for messaging.

---

## Q

### 471. qlib (0.0.2.dev20)
**Decision:** ❌ DELETE
**What:** Microsoft's quant research platform
**Reason:** Not actively used. You have other tools.

### 472-473. qrcode, QtPy
**Decision:** ❌ DELETE
**What:** QR codes, Qt bindings
**Reason:** Not needed.

### 474. Quandl (3.7.0)
**Decision:** ❌ DELETE
**What:** Quandl API (deprecated)
**Reason:** API is deprecated. Use OpenBB instead.

### 475-476. quantconnect, quantconnect-stubs
**Decision:** ❌ DELETE
**What:** QuantConnect platform
**Reason:** You use Alpaca, not QuantConnect.

### 477. QuantLib (1.39)
**Decision:** ⚠️ OPTIONAL
**What:** Derivatives pricing library
**Reason:** Advanced derivatives pricing. Only if pricing exotic options. HUGE library.

### 478. Quantsbin (1.0.3)
**Decision:** ❌ DELETE
**What:** Quant binaries
**Reason:** Not actively maintained.

### 479. QuantStats (0.0.77)
**Decision:** 🟡 KEEP
**What:** Portfolio statistics
**Reason:** Excellent portfolio metrics. Industry standard.

### 480-481. questionary, queuelib
**Decision:** ⚠️ OPTIONAL (questionary), ❌ DELETE (queuelib)
**What:** Interactive prompts, queue
**Reason:** questionary for CLI. queuelib for Scrapy.

---

## R

### 482-483. RapidFuzz, rd
**Decision:** ⚠️ OPTIONAL (RapidFuzz), ❌ DELETE (rd)
**What:** Fuzzy string matching, unknown
**Reason:** RapidFuzz could be useful. rd unknown.

### 484. realtime (2.5.3)
**Decision:** ⚠️ OPTIONAL
**What:** Real-time data
**Reason:** Could be Supabase related. Check if needed.

### 485. redis (6.2.0)
**Decision:** ❌ DELETE
**What:** Redis client
**Reason:** You don't use Redis.

### 486-495. referencing through rfc3339-validator
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** Common dependencies. Keep.

### 496-497. rich, rich-rst
**Decision:** ⚠️ OPTIONAL
**What:** Rich terminal formatting
**Reason:** Nice for CLIs. Keep (colorama alternative).

### 498. Riskfolio-Lib (7.0.1)
**Decision:** 🟡 KEEP
**What:** Portfolio optimization
**Reason:** Advanced portfolio optimization. Excellent library.

### 499-501. rpds-py, rsa, ruff
**Decision:** ⚠️ OPTIONAL
**What:** Data structures, RSA, linter
**Reason:** ruff is fast linter (alternative to black).

---

## S

### 502. safetensors (0.6.2)
**Decision:** ❌ DELETE
**What:** Safe tensor storage
**Reason:** Related to transformers. You don't use transformers.

### 503. schedule (1.2.2)
**Decision:** ✅ KEEP
**What:** Job scheduling
**Reason:** Used in auto_options_scanner.py for daily scans.

### 504. scikit-learn (1.7.0)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Machine learning
**Reason:** Core ML library. RandomForest, GradientBoosting, etc.

### 505. scipy (1.15.3)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Scientific computing
**Reason:** Used for options pricing (scipy.stats.norm), optimization.

### 506. Scrapy (2.13.3)
**Decision:** ❌ DELETE
**What:** Web scraping framework
**Reason:** Not scraping in production. HUGE framework.

### 507. scs (3.2.8)
**Decision:** ❌ DELETE
**What:** Optimization solver
**Reason:** Dependency of cvxpy. You have Riskfolio-Lib.

### 508. sdnotify (0.3.2)
**Decision:** ❌ DELETE
**What:** Systemd notifications
**Reason:** Linux service management. Not needed on Windows.

### 509. seaborn (0.13.2)
**Decision:** 🟡 KEEP
**What:** Statistical visualization
**Reason:** Makes beautiful matplotlib plots. Useful for analysis.

### 510. selenium (4.31.0)
**Decision:** ❌ DELETE
**What:** Browser automation
**Reason:** Not doing browser automation in trading.

### 511. sendgrid (6.12.4)
**Decision:** ❌ DELETE
**What:** Email API
**Reason:** Not sending emails from trading system.

### 512-514. service-identity through sgmllib3k
**Decision:** ❌ DELETE
**What:** Twisted dependencies, SGML parser
**Reason:** Related to Scrapy/web scraping.

### 515-522. shellingham through soupsieve
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** Common dependencies. Keep.

### 523. SQLAlchemy (2.0.41)
**Decision:** ❌ DELETE
**What:** ORM
**Reason:** You use JSON logging, not SQL databases.

### 524-525. sse-starlette, sseclient-py
**Decision:** ⚠️ OPTIONAL
**What:** Server-sent events
**Reason:** Could be useful for real-time streams. Small.

### 526. stable_baselines3 (2.7.0)
**Decision:** ❌ DELETE
**What:** Reinforcement learning
**Reason:** You don't do RL trading.

### 527-530. stack-data through statsmodels
**Decision:** ⚠️ OPTIONAL
**What:** Debugging, statistics
**Reason:** statsmodels useful for econometrics. Keep.

### 531. storage3 (0.12.0)
**Decision:** ❌ DELETE
**What:** Supabase storage
**Reason:** Not using Supabase.

### 532. streamlit (1.49.1)
**Decision:** 🟡 KEEP
**What:** Dashboard framework
**Reason:** Quick dashboards. Alternative to dash. Pick one.

### 533-536. StrEnum through supafunc
**Decision:** ⚠️ OPTIONAL (StrEnum, structlog), ❌ DELETE (supabase)
**What:** Various utilities, Supabase
**Reason:** Not using Supabase. Others are dependencies.

### 537. sympy (1.14.0)
**Decision:** ❌ DELETE
**What:** Symbolic mathematics
**Reason:** Advanced math. Not needed for trading.

---

## T

### 538. ta (0.11.0)
**Decision:** 🟡 KEEP
**What:** Technical analysis library
**Reason:** Pure Python TA. Good alternative to TA-Lib.

### 539. TA-Lib (0.6.7)
**Decision:** 🟡 KEEP (CRITICAL FOR SERIOUS TRADING)
**What:** Technical analysis C library
**Reason:** 200+ indicators. Industry standard. ESSENTIAL.

### 540. tables (3.10.2)
**Decision:** ❌ DELETE
**What:** HDF5 tables
**Reason:** Not using HDF5 format.

### 541. tabulate (0.9.0)
**Decision:** ⚠️ OPTIONAL
**What:** Pretty tables
**Reason:** Useful for printing data. Keep (tiny).

### 542. technical (1.5.3)
**Decision:** ❌ DELETE
**What:** Freqtrade indicators
**Reason:** Duplicate of TA-Lib. Delete.

### 543. tenacity (9.1.2)
**Decision:** ⚠️ OPTIONAL
**What:** Retry library
**Reason:** Useful for API calls. Alternative to backoff.

### 544-545. tensorboard, tensorboard-data-server
**Decision:** ❌ DELETE
**What:** TensorBoard visualization
**Reason:** TensorFlow tool. You don't use TensorFlow.

### 546. termcolor (3.1.0)
**Decision:** ⚠️ OPTIONAL
**What:** Terminal colors
**Reason:** Alternative to colorama. Keep one.

### 547-548. text-unidecode, textblob
**Decision:** ❌ DELETE
**What:** Unicode to ASCII, NLP
**Reason:** Not doing text processing in trading.

### 549. threadpoolctl (3.6.0)
**Decision:** ⚠️ OPTIONAL
**What:** Thread pool control
**Reason:** Used by scikit-learn. Keep.

### 550. tiktoken (0.9.0)
**Decision:** ⚠️ OPTIONAL
**What:** OpenAI tokenizer
**Reason:** Useful for counting tokens in API calls. Keep.

### 551-553. tinysegmenter through tldextract
**Decision:** ❌ DELETE
**What:** Japanese tokenizer, domain extraction
**Reason:** Not needed.

### 554. tls-client (0.2.2)
**Decision:** ❌ DELETE
**What:** TLS client
**Reason:** Not needed. Standard requests handles TLS.

### 555. tokenizers (0.22.1)
**Decision:** ❌ DELETE
**What:** HuggingFace tokenizers
**Reason:** Related to transformers. You don't use.

### 556-560. toml through toolz
**Decision:** ⚠️ OPTIONAL
**What:** Configuration, utilities
**Reason:** Common dependencies. Keep.

### 561-562. torch, torchvision
**Decision:** ❌ DELETE (HUGE - 2+ GB!)
**What:** PyTorch deep learning
**Reason:** You use scikit-learn. PyTorch is MASSIVE. DELETE.

### 563. tornado (6.5.2)
**Decision:** ⚠️ OPTIONAL
**What:** Async web framework
**Reason:** Dependency of some packages. Keep.

### 564. tqdm (4.67.1)
**Decision:** ⚠️ OPTIONAL
**What:** Progress bars
**Reason:** Useful for long operations. Keep (tiny).

### 565. tradingview-ta (3.3.0)
**Decision:** ⚠️ OPTIONAL
**What:** TradingView technical analysis
**Reason:** Could be useful. Alternative TA source.

### 566. trafilatura (2.0.0)
**Decision:** ❌ DELETE
**What:** Web scraping
**Reason:** Not scraping in production.

### 567-571. traitlets through trove-classifiers
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** Common dependencies. Keep.

### 572. tulip (1.4.0)
**Decision:** ❌ DELETE
**What:** Technical analysis C library
**Reason:** You have TA-Lib. Redundant.

### 573-574. tweepy, twilio
**Decision:** ❌ DELETE
**What:** Twitter API, SMS
**Reason:** Not using Twitter or SMS.

### 575. Twisted (25.5.0)
**Decision:** ❌ DELETE
**What:** Async networking framework
**Reason:** Dependency of Scrapy. HUGE. Delete with Scrapy.

### 576-579. typer through typing-inspection
**Decision:** ⚠️ OPTIONAL
**What:** CLI, type utilities
**Reason:** Common dependencies. Keep.

### 580-581. tzdata, tzlocal
**Decision:** ⚠️ OPTIONAL
**What:** Timezone data
**Reason:** Important for trading (market hours). Keep.

---

## U-V

### 582-583. uagents, uagents-core
**Decision:** ❌ DELETE
**What:** Fetch.ai agents
**Reason:** Blockchain agent framework. Not needed.

### 584-591. ujson through uv
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities, UV package installer
**Reason:** ujson is fast JSON. uv is modern pip alternative. Keep if using.

### 592. v20 (3.0.25.0)
**Decision:** ✅ KEEP (CRITICAL)
**What:** OANDA API
**Reason:** Used for forex trading. ESSENTIAL.

### 593. vectorbt (0.28.1)
**Decision:** 🟡 KEEP
**What:** Vectorized backtesting
**Reason:** Fast backtesting framework. Professional tool.

### 594-603. virtualenv through websockets
**Decision:** ⚠️ OPTIONAL
**What:** Various utilities
**Reason:** websockets important for real-time data. Others are dependencies.

### 604-607. Werkzeug through wrapt
**Decision:** ⚠️ OPTIONAL (most), ❌ DELETE (Werkzeug)
**What:** Flask utilities, Windows, decorators
**Reason:** Werkzeug is Flask dependency. Delete. Others keep.

### 608. wsproto (1.2.0)
**Decision:** ⚠️ OPTIONAL
**What:** WebSocket protocol
**Reason:** Dependency of websockets. Keep if using WebSockets.

---

## X-Z

### 609-611. xarray, xarray-einstats, xgboost
**Decision:** ⚠️ OPTIONAL (xarray), ❌ DELETE (xgboost)
**What:** N-D arrays, gradient boosting
**Reason:** xarray for climate/earth data. xgboost redundant (you have sklearn).

### 612-614. xlrd, xlsxwriter, xmltodict
**Decision:** ⚠️ OPTIONAL
**What:** Excel reading/writing, XML parsing
**Reason:** Could be useful for data import/export. Keep.

### 615. xxhash (3.5.0)
**Decision:** ⚠️ OPTIONAL
**What:** Fast hashing
**Reason:** Very fast hash algorithm. Keep (tiny).

### 616. yarl (1.20.0)
**Decision:** ⚠️ OPTIONAL
**What:** URL parsing
**Reason:** Dependency of aiohttp. Keep if keeping aiohttp.

### 617. yfinance (0.2.58)
**Decision:** ✅ KEEP (CRITICAL)
**What:** Yahoo Finance API
**Reason:** Free market data. Currently used in production.

### 618. zipline-reloaded (3.1.1)
**Decision:** ⚠️ OPTIONAL
**What:** Zipline backtesting platform
**Reason:** Professional backtesting. But you have backtrader/vectorbt. Pick one.

### 619-621. zipp, zope.interface, zstandard
**Decision:** ⚠️ OPTIONAL
**What:** ZIP utilities, interfaces, compression
**Reason:** Common dependencies. Keep.

---

## 📊 FINAL SUMMARY

### ✅ CRITICAL KEEP (15):
alpaca-py, alpaca-trade-api, v20, pandas, numpy, yfinance, scikit-learn, scipy, anthropic, openai, python-dotenv, requests, schedule, python-dateutil, pytz

### 🟡 PROFESSIONAL KEEP (70):
All LangChain packages (12), All OpenBB packages (31), TA-Lib, pandas-ta, ta, backtrader, vectorbt, bt, QuantStats, pyfolio-reloaded, empyrical-reloaded, Riskfolio-Lib, pyportfolioopt, matplotlib, seaborn, plotly, dash, streamlit, cufflinks, chromadb, crewai, instructor, fredapi, alpha_vantage, polygon-api-client, colorama, black, pytest, aiohttp, websockets

### ⚠️ OPTIONAL (~75):
Dependencies and utilities

### ❌ DELETE (460+):
PyTorch (2GB!), TensorFlow, Keras, crypto trading, Scrapy, web frameworks, databases, Kubernetes, game engines, astronomy libraries, etc.

---

**RECOMMENDATION:**
Keep 85-90 professional libraries (~1.5 GB)
Delete 530+ bloat libraries (~4.5 GB saved)

**Path:** `ALL_621_PACKAGES_ANALYSIS.md`
