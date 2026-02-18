# Quant Lab: Institutional Strategy Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Status](https://img.shields.io/badge/status-stable-green)

**Quant Lab** is a professional-grade, portable research environment for developing, backtesting, and validating algorithmic trading strategies (specifically `NQ` and `ES` futures). 

Unlike standard backtesters, this engine features **Dynamic Volatility Slippage** and **Transaction Cost Modeling**, ensuring that strategies are validated against realistic, hostile market conditions rather than idealized assumptions.

---

## 🚀 Key Features

### 1. Institutional-Grade Backtester (`VectorEngine`)
*   **Dynamic Volatility Slippage**: Execution costs scale with market volatility (1% of Bar Range).
    *   *Calm Market*: Low slippage (~0.2 ticks).
    *   *Crash/Panic*: High slippage (~2-3 points).
*   **Precision Modeling**: Loads exact contract specs (Tick Size, Multiplier, Commission) from `config.json`.
*   **GPU Acceleration**: Supports Numba/cuDF for lightning-fast vector operations (with CPU fallback).

### 2. Truly Portable "Starter Kit"
*   **Included Data**: Compressed 1-minute and 5-minute tick data for NQ/ES (2020-2023) is included (`data/*.zip`).
*   **Auto-Hydration**: `setup_project.bat` automatically unzips data and installs dependencies.
*   **Docker / DevContainer**: One-click environment setup for VS Code.

### 3. Professional Tooling
*   **CLI Interface**: Run experiments without editing code.
    ```bash
    python run_experiments.py --symbol NQ --start 2023-01-01 --end 2023-12-31
    ```
*   **CI/CD**: GitHub Actions workflow (`test.yml`) ensures code stability.
*   **Dependency Locking**: `requirements.lock` for reproducible builds.

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+ or Docker
*   (Optional) NVIDIA GPU for acceleration

### Fast Setup (Windows)
1.  **Clone the Repo**
    ```powershell
    git clone https://github.com/bostonrobbie/Strategy-Lab.git
    cd Strategy-Lab
    ```
2.  **Run Setup Script**
    Double-click `setup_project.bat` or run:
    ```powershell
    .\setup_project.bat
    ```
    *This creates a virtual environment, installs libs, and unzips the data.*

3.  **Activate**
    ```powershell
    .venv\Scripts\activate
    ```

### Configuration
1.  Rename `.env.example` to `.env`.
2.  Add your API keys (Polygon, IBKR) if you plan to fetch *new* data.

---

## 🧪 Running Experiments

**Basic Run**:
```bash
python run_experiments.py --symbol NQ
```

**Custom Date Range**:
```bash
python run_experiments.py --symbol ES --start 2021-01-01 --end 2022-01-01 --capital 250000
```

---

## 📂 Project Structure
*   `StrategyPipeline/`: Core source code.
    *   `src/backtesting/`: The engine (Vector, Execution, Data).
    *   `strategies/`: Strategy logic (ORB, Mean Reversion).
*   `data/`: Local data storage (Git-ignored, auto-populated by setup).
*   `outputs/`: Backtest results and reports.

## 🔮 Future Roadmap
See [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for planned features:
1.  Cloud-Native Hyperparameter Optimization (Ray Tune)
2.  Event-Driven Engine (Zipline style)
3.  Automated Regime Detection
4.  Live Trading Bridge (IBKR)
5.  Strategy Dashboard (Streamlit)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
