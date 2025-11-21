# HiveTrading System

## Overview
HiveTrading is a comprehensive quantitative trading platform designed for research, backtesting, and live trading. It features a modular architecture with support for multiple asset classes (Forex, Crypto, Futures) and advanced strategies.

## Prerequisites
- **Python 3.11+**
- **Poetry** (Dependency Management)
- **CUDA Toolkit** (Optional, for GPU acceleration)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd PC-HIVE-TRADING
    ```

2.  **Install dependencies**:
    We use [Poetry](https://python-poetry.org/) for dependency management.
    ```bash
    poetry install
    ```

3.  **Environment Setup**:
    Copy `.env.example` to `.env` and configure your API keys.
    ```bash
    cp .env.example .env
    ```

## Directory Structure

- **`bin/`**: Batch scripts for Windows automation and quick tasks.
- **`scripts/`**: Utility Python scripts for various tasks (backtesting, analysis, maintenance).
- **`strategies/`**: Core trading strategy implementations.
- **`backend/`**: FastAPI backend for the trading dashboard.
- **`frontend/`**: Web frontend for the dashboard.
- **`tests/`**: Comprehensive test suite.
- **`config/`**: Configuration files.
- **`data/`**: Data storage (logs, database, etc.).

## Usage

### Running Scripts
All utility scripts have been moved to the `scripts/` directory. Run them using Poetry to ensure dependencies are loaded correctly:

```bash
poetry run python scripts/BACKTEST_FOREX_STRATEGY.py
```

### Running the Dashboard
To start the backend server:

```bash
poetry run python backend/main.py
```

### Running Tests
To run the test suite:

```bash
poetry run pytest
```

## Development
- **Dependency Management**: Add new packages with `poetry add <package>`.
- **Code Style**: We use `black` and `flake8`. Run `poetry run black .` to format code.

## License
Proprietary software. All rights reserved.
