# MARCUS - Multi-Asset Research and Compute Unified System

Automated quantitative trading research platform for NQ and ES futures.

## Overview

MARCUS is an end-to-end quantitative research pipeline that discovers,
backtests, optimizes, validates, and deploys systematic trading strategies.

## Quick Start

    pip install -r requirements.txt
    python main.py
    pytest tests/

## Structure

- main.py - Pipeline entry point
- config.json - Pipeline configuration
- src/backtesting/ - Core backtesting engine (vectorized + GPU)
- strategies/ - Strategy definitions (ORB, overnight drift, mean reversion, etc)
- scripts/ - Operational scripts (runner, autopilot, benchmarks, QA)
- research/ - 16 strategy discovery research modules
- pine/ - 9 TradingView Pine Script strategies
- utils/ - Analysis, benchmarking, and debugging utilities
- tests/ - Full test suite (unit, integration, e2e, parity)
- dashboard/ - Web dashboard (OpenClaw integration)
- data/ - Market data archives (NQ + ES, 1m/5m/15m)
- docs/ - Documentation
- legacy/ - Original TypeScript agentic hedge fund orchestrator

## Key Strategies

- NQ ORB Enhanced: Opening Range Breakout with regime filters (5m/15m)
- Overnight Drift: Session momentum capture (Daily)
- LinReg Mean Reversion: Linear regression-based mean reversion (5m)
- EOD Fade MOC: End-of-day fade with market-on-close (15m)
- GhostRider: Multi-timeframe composite strategy

## Architecture

- Vectorized Engine: NumPy-based backtesting (100x faster than event-driven)
- GPU Acceleration: Optional CuPy/CUDA for parameter sweeps
- Walk-Forward Optimization: Rolling in-sample/out-of-sample windows
- Regime Detection: VIX-based regime filters for adaptive behavior
- Statistical Validation: Sharpe, MaxDD, Profit Factor gates
- Pine Script Generation: Automatic TradingView code from Python

## Legacy System

The legacy/ directory contains the original TypeScript MARCUS agentic
hedge fund orchestrator, preserved for reference.
