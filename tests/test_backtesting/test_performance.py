"""
Tests for performance metrics and tear sheet generation.

These tests verify metric calculations like Sharpe ratio, drawdowns, etc.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Should calculate positive Sharpe for positive returns."""
        from backtesting.performance import create_sharpe_ratio

        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.025])
        sharpe = create_sharpe_ratio(returns, periods=252)

        assert sharpe > 0

    def test_negative_sharpe(self):
        """Should calculate negative Sharpe for negative returns."""
        from backtesting.performance import create_sharpe_ratio

        returns = pd.Series([-0.01, -0.02, -0.015, -0.01, -0.025])
        sharpe = create_sharpe_ratio(returns, periods=252)

        assert sharpe < 0

    def test_annualization(self):
        """Should annualize based on periods."""
        from backtesting.performance import create_sharpe_ratio

        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe_daily = create_sharpe_ratio(returns, periods=252)
        sharpe_monthly = create_sharpe_ratio(returns, periods=12)

        # Daily annualization should produce higher Sharpe
        assert sharpe_daily > sharpe_monthly


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_positive_sortino(self):
        """Should calculate positive Sortino for positive mean returns."""
        from backtesting.performance import create_sortino_ratio

        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
        sortino = create_sortino_ratio(returns, periods=252)

        assert sortino > 0

    def test_handles_no_negative_returns(self):
        """Should return 0 when no negative returns."""
        from backtesting.performance import create_sortino_ratio

        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.025])
        sortino = create_sortino_ratio(returns, periods=252)

        assert sortino == 0.0


class TestDrawdowns:
    """Tests for drawdown calculations."""

    def test_max_drawdown(self):
        """Should calculate maximum drawdown."""
        from backtesting.performance import create_drawdowns

        equity_df = pd.DataFrame({
            'equity': [100, 110, 105, 90, 95, 100, 108]
        })

        dd_series, max_dd, max_duration = create_drawdowns(equity_df)

        # Max drawdown from 110 to 90 = (90-110)/110 = -18.18%
        assert max_dd == pytest.approx(-0.1818, abs=0.01)

    def test_no_drawdown(self):
        """Should handle strictly increasing equity."""
        from backtesting.performance import create_drawdowns

        equity_df = pd.DataFrame({
            'equity': [100, 105, 110, 115, 120]
        })

        dd_series, max_dd, max_duration = create_drawdowns(equity_df)

        assert max_dd == 0.0


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_var_95(self):
        """Should calculate 95% VaR."""
        from backtesting.performance import calculate_var

        # Create returns with known distribution
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = calculate_var(returns, confidence_level=0.95)

        # 5th percentile of normal(0.001, 0.02) should be around -0.032
        assert var < 0  # VaR should be negative (loss)

    def test_var_99(self):
        """Should calculate 99% VaR."""
        from backtesting.performance import calculate_var

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var_95 = calculate_var(returns, confidence_level=0.95)
        var_99 = calculate_var(returns, confidence_level=0.99)

        # 99% VaR should be more extreme than 95%
        assert var_99 < var_95

    def test_var_insufficient_data(self):
        """Should return 0 with insufficient data."""
        from backtesting.performance import calculate_var

        returns = [0.01, 0.02]  # Only 2 data points
        var = calculate_var(returns, confidence_level=0.95)

        assert var == 0.0


class TestTailRatio:
    """Tests for tail ratio calculation."""

    def test_positive_tail_ratio(self):
        """Should calculate positive tail ratio."""
        from backtesting.performance import calculate_tail_ratio

        # More upside than downside
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, -0.01])
        tail_ratio = calculate_tail_ratio(returns)

        assert tail_ratio > 1.0

    def test_handles_zero_downside(self):
        """Should handle zero 5th percentile."""
        from backtesting.performance import calculate_tail_ratio

        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        tail_ratio = calculate_tail_ratio(returns)

        # With all positive, 5th percentile is still positive, so ratio is positive
        assert tail_ratio >= 0


class TestTearSheet:
    """Tests for TearSheet class."""

    def test_initialization(self, sample_ohlcv_dict):
        """Should initialize from portfolio."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        # Add some equity curve data
        portfolio.equity_curve = [
            {'datetime': datetime(2024, 1, 1), 'cash': 100000, 'equity': 100000, 'total_market_value': 0},
            {'datetime': datetime(2024, 1, 2), 'cash': 100000, 'equity': 101000, 'total_market_value': 0},
        ]

        tear_sheet = TearSheet(portfolio)

        assert tear_sheet.portfolio is portfolio

    def test_analyze_empty_portfolio(self, sample_ohlcv_dict, capsys):
        """Should handle empty equity curve."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events)

        tear_sheet = TearSheet(portfolio)
        result = tear_sheet.analyze()

        assert result == {}
        captured = capsys.readouterr()
        assert "No trades" in captured.out

    def test_analyze_returns_stats(self, sample_ohlcv_dict):
        """Should return comprehensive statistics."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        # Create equity curve spanning multiple days
        dates = pd.date_range('2024-01-01', periods=30, freq='1D')
        portfolio.equity_curve = [
            {
                'datetime': d,
                'cash': 100000 + i * 100,
                'equity': 100000 + i * 100,
                'total_market_value': 0
            }
            for i, d in enumerate(dates)
        ]

        with patch.object(TearSheet, 'create_html_report'):
            tear_sheet = TearSheet(portfolio)
            stats = tear_sheet.analyze()

        assert 'Total Return' in stats
        assert 'CAGR' in stats
        assert 'Sharpe Ratio' in stats
        assert 'Sortino Ratio' in stats
        assert 'Max Drawdown' in stats
        assert 'Annual Volatility' in stats
        assert 'Ending Equity' in stats

    def test_analyze_with_trade_log(self, sample_ohlcv_dict):
        """Should include profit factor from trade log."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        dates = pd.date_range('2024-01-01', periods=30, freq='1D')
        portfolio.equity_curve = [
            {'datetime': d, 'cash': 100000, 'equity': 100000 + i * 50, 'total_market_value': 0}
            for i, d in enumerate(dates)
        ]
        portfolio.trade_log = [
            {'datetime': dates[0], 'symbol': 'NQ', 'side': 'BUY', 'quantity': 1, 'price': 17000, 'commission': 2.05, 'multiplier': 1.0, 'realized_pnl': 0},
            {'datetime': dates[5], 'symbol': 'NQ', 'side': 'SELL', 'quantity': 1, 'price': 17100, 'commission': 2.05, 'multiplier': 1.0, 'realized_pnl': 100},
        ]

        with patch.object(TearSheet, 'create_html_report'):
            tear_sheet = TearSheet(portfolio)
            stats = tear_sheet.analyze()

        assert 'Profit Factor' in stats

    @patch('backtesting.performance.GPU_AVAILABLE', False)
    def test_monte_carlo_cpu(self, sample_ohlcv_dict):
        """Should run Monte Carlo on CPU."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        dates = pd.date_range('2024-01-01', periods=100, freq='1D')
        portfolio.equity_curve = [
            {'datetime': d, 'cash': 100000, 'equity': 100000 + i * 10, 'total_market_value': 0}
            for i, d in enumerate(dates)
        ]

        tear_sheet = TearSheet(portfolio)
        mc_stats, sim_curves = tear_sheet.run_monte_carlo(n_sims=100)

        assert 'MC_Max' in mc_stats
        assert 'MC_Median' in mc_stats
        assert 'MC_Min' in mc_stats
        assert 'Risk_of_Ruin' in mc_stats
        assert len(sim_curves) <= 50

    def test_monte_carlo_insufficient_data(self, sample_ohlcv_dict):
        """Should return empty with insufficient data."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        # Only 5 data points (less than 10)
        portfolio.equity_curve = [
            {'datetime': datetime(2024, 1, i+1), 'cash': 100000, 'equity': 100000 + i * 10, 'total_market_value': 0}
            for i in range(5)
        ]

        tear_sheet = TearSheet(portfolio)
        mc_stats, sim_curves = tear_sheet.run_monte_carlo()

        assert mc_stats == {}
        assert sim_curves is None


class TestSafeIlocInPerformance:
    """Tests verifying safe_iloc is used correctly in performance.py."""

    def test_analyze_with_numpy_equity(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """Should handle numpy array equity curves via safe_iloc."""
        from backtesting.performance import TearSheet
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        dates = pd.date_range('2024-01-01', periods=30, freq='1D')
        portfolio.equity_curve = [
            {'datetime': d, 'cash': 100000, 'equity': 100000 + i * 100, 'total_market_value': 0}
            for i, d in enumerate(dates)
        ]

        with patch.object(TearSheet, 'create_html_report'):
            tear_sheet = TearSheet(portfolio)
            # This should not raise AttributeError
            stats = tear_sheet.analyze()

        assert 'Total Return' in stats
        assert 'Ending Equity' in stats
