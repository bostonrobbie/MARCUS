# AI Agent Setup Manual for Quant_Lab

## Overview
**To the AI Agent reading this:**
You have been tasked with setting up the `Quant_Lab` on a new machine. This repository is a "portable shell". It contains all the logic (`StrategyPipeline`) but is missing **massive data**, **API keys**, and **environment binaries** to keep it lightweight.

Follow these instructions to "hydrate" the system into a fully functional Quantitative Trading Environment.

---

## Phase 1: Environment & Logic (The Easy Part)

### 1.1 Run the Auto-Setup Script
*   **Action**: Execute `setup_project.bat` in the root directory.
*   **What it does**:
    *   Checks for Python.
    *   Creates a fresh virtual environment (`.venv`).
    *   Installs all Python dependencies from `requirements.txt`.
    *   Recreates the empty directory structure (`data/`, `logs/`, `outputs/`).

### 1.2 Verify Dependencies
*   **Action**: Activate the environment (`.venv\Scripts\activate`) and run:
    ```bash
    python -c "import pandas; import numpy; print('Environment OK')"
    ```

---

## Phase 2: Configuration (The Keys)

### 2.1 API Keys
*   **Context**: The system needs data from providers (like Polygon.io) or brokers (IBKR).
*   **Action**:
    1.  Locate `.env.example`.
    2.  Copy it to a new file named `.env`.
    3.  **CRITICAL**: Ask the USER for their `POLYGON_API_KEY` and `IB_ACCOUNT` details. Do not guess these.
    4.  Fill in the `.env` file.

---

## Phase 3: Data Hydration (The Heavy Lift)

### 3.1 Download Tick Data
*   **Context**: The strategy relies on high-resolution tick data (CSVs) which were too big for GitHub.
*   **Option A (Fastest)**: If the user has a cloud backup (S3/Google Drive), ask for the link and download the `data/` folder content.
*   **Option B (Re-Generate)**: Use the included scripts to fetch fresh data involved in the `.env` is set up.
    *   **Command**: `python StrategyPipeline/src/backtesting/generate_data.py --symbol NQ --days 30`
    *   *Note: This may take hours depending on the API limits.*

### 3.2 Verification
*   **Action**: Ensure `data/` contains files like `NQ_tick_data.csv` or similar before running backtests.

---

## Phase 4: Hardware Acceleration (Optional but Recommended)

### 4.1 CUDA Drivers
*   **Context**: Some strategies use GPU acceleration.
*   **Action**: Check if the machine has an NVIDIA GPU.
    *   Run `nvidia-smi`.
    *   If present, ensure drivers are installed. If `torch.cuda.is_available()` returns `False`, you may need to guide the user to install the specific Torch-CUDA version manually (standard `pip install` often defaults to CPU-only).

---

## Phase 5: Go Live
1.  **Run a Test Experiment**:
    ```bash
    python run_experiments.py --test-mode
    ```
2.  If it completes without errors, the `Quant_Lab` is fully operational.
