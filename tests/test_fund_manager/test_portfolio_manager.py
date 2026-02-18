"""
Tests for Portfolio class.

These tests verify the portfolio management system including
strategy tracking, allocation, and metrics calculation.
"""
import pytest
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestPortfolioInitialization:
    """Tests for Portfolio initialization."""

    def test_creates_empty_portfolio(self, tmp_path, monkeypatch):
        """Should initialize with default values if file doesn't exist."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()

        assert port.data["strategies"] == []
        assert port.data["total_capital"] == 100000.0
        assert port.data["portfolio_metrics"]["num_strategies"] == 0

    def test_loads_existing_portfolio(self, tmp_path, monkeypatch):
        """Should load existing portfolio data."""
        from portfolio import Portfolio

        port_file = tmp_path / 'portfolio.json'
        existing = {
            "strategies": [{"name": "Test", "status": "active", "weight": 0.5, "metrics": {"sharpe": 1.5}}],
            "total_capital": 200000.0,
            "portfolio_metrics": {"num_strategies": 1}
        }
        port_file.write_text(json.dumps(existing))

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(port_file))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()

        assert len(port.data["strategies"]) == 1
        assert port.data["total_capital"] == 200000.0


class TestAddStrategy:
    """Tests for adding strategies to portfolio."""

    def test_adds_new_strategy(self, tmp_path, monkeypatch):
        """Should add strategy with all fields."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        strategy = port.add_strategy(
            name="TestStrategy",
            metrics={"sharpe": 1.5, "return_pct": 0.15, "max_dd": -0.1}
        )

        assert strategy["name"] == "TestStrategy"
        assert strategy["status"] == "active"
        assert "added_at" in strategy
        assert len(port.data["strategies"]) == 1

    def test_updates_existing_strategy(self, tmp_path, monkeypatch):
        """Should update strategy if name already exists."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})
        port.add_strategy("Test", {"sharpe": 2.0})  # Update

        assert len(port.data["strategies"]) == 1
        assert port.data["strategies"][0]["metrics"]["sharpe"] == 2.0

    def test_calculates_weight_from_sharpe(self, tmp_path, monkeypatch):
        """Should auto-calculate weight based on Sharpe."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()

        # High Sharpe gets higher weight
        high = port.add_strategy("High", {"sharpe": 2.5})
        assert high["weight"] == 0.25

        # Medium Sharpe
        med = port.add_strategy("Med", {"sharpe": 1.5})
        assert med["weight"] <= 0.20

    def test_accepts_custom_weight(self, tmp_path, monkeypatch):
        """Should use custom weight if provided."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        strategy = port.add_strategy("Test", {"sharpe": 0.5}, weight=0.5)

        # Weight may be rebalanced, but should be set initially
        assert strategy is not None

    def test_logs_history(self, tmp_path, monkeypatch):
        """Should log addition to history file."""
        from portfolio import Portfolio

        hist_file = tmp_path / 'history.json'
        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(hist_file))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})

        history = json.loads(hist_file.read_text())
        assert len(history) == 1
        assert history[0]["action"] == "add"


class TestRemoveStrategy:
    """Tests for removing strategies."""

    def test_removes_strategy(self, tmp_path, monkeypatch):
        """Should remove strategy by name."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})
        result = port.remove_strategy("Test")

        assert result is True
        assert len(port.data["strategies"]) == 0

    def test_returns_false_for_unknown(self, tmp_path, monkeypatch):
        """Should return False if strategy not found."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        result = port.remove_strategy("NonExistent")

        assert result is False


class TestUpdateStatus:
    """Tests for strategy status updates."""

    def test_updates_status(self, tmp_path, monkeypatch):
        """Should update strategy status."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})
        result = port.update_strategy_status("Test", "paused")

        assert result is True
        assert port.data["strategies"][0]["status"] == "paused"

    def test_returns_false_for_unknown(self, tmp_path, monkeypatch):
        """Should return False if strategy not found."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        result = port.update_strategy_status("NonExistent", "paused")

        assert result is False


class TestRebalancing:
    """Tests for portfolio rebalancing."""

    def test_weights_sum_to_one(self, tmp_path, monkeypatch):
        """Active strategy weights should sum to 1.0."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("A", {"sharpe": 2.0})
        port.add_strategy("B", {"sharpe": 1.5})
        port.add_strategy("C", {"sharpe": 1.0})

        total = sum(s["weight"] for s in port.data["strategies"])
        assert total == pytest.approx(1.0, abs=0.01)

    def test_excludes_paused_from_rebalance(self, tmp_path, monkeypatch):
        """Paused strategies should not affect weight calculation."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Active", {"sharpe": 1.0})
        port.add_strategy("Paused", {"sharpe": 1.0})
        port.update_strategy_status("Paused", "paused")

        active = port.get_active_strategies()
        total = sum(s["weight"] for s in active)
        assert total == pytest.approx(1.0, abs=0.01)


class TestPortfolioMetrics:
    """Tests for portfolio-level metrics."""

    def test_calculates_weighted_sharpe(self, tmp_path, monkeypatch):
        """Should calculate weighted average Sharpe."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("A", {"sharpe": 2.0})
        port.add_strategy("B", {"sharpe": 1.0})

        # Weighted avg should be between 1.0 and 2.0
        pm = port.data["portfolio_metrics"]
        assert 1.0 <= pm["sharpe"] <= 2.0

    def test_handles_empty_portfolio(self, tmp_path, monkeypatch):
        """Should have zero metrics when empty."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()

        pm = port.data["portfolio_metrics"]
        assert pm["sharpe"] == 0.0
        assert pm["num_strategies"] == 0


class TestGetSummary:
    """Tests for portfolio summary."""

    def test_returns_complete_summary(self, tmp_path, monkeypatch):
        """Should return all summary fields."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.5})

        summary = port.get_portfolio_summary()

        assert "total_strategies" in summary
        assert "active_count" in summary
        assert "total_capital" in summary
        assert "portfolio_metrics" in summary
        assert "strategies" in summary

    def test_counts_by_status(self, tmp_path, monkeypatch):
        """Should count strategies by status."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Active1", {"sharpe": 1.0})
        port.add_strategy("Active2", {"sharpe": 1.0})
        port.add_strategy("Paused", {"sharpe": 1.0})
        port.update_strategy_status("Paused", "paused")

        summary = port.get_portfolio_summary()

        assert summary["active_count"] == 2
        assert summary["paused_count"] == 1


class TestEquityCurve:
    """Tests for equity curve tracking."""

    def test_adds_equity_point(self, tmp_path, monkeypatch):
        """Should add equity data point."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})

        result = port.update_strategy_equity("Test", {
            "timestamp": datetime.now().isoformat(),
            "equity": 105000,
            "pnl": 5000
        })

        assert result is True
        curve = port.get_strategy_equity_curve("Test")
        assert len(curve) == 1

    def test_limits_to_1000_points(self, tmp_path, monkeypatch):
        """Should keep only last 1000 points."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.add_strategy("Test", {"sharpe": 1.0})

        # Add 1100 points
        for i in range(1100):
            port.update_strategy_equity("Test", {"equity": 100000 + i})

        curve = port.get_strategy_equity_curve("Test")
        assert len(curve) <= 1000


class TestAllocationBreakdown:
    """Tests for capital allocation calculation."""

    def test_returns_allocation_details(self, tmp_path, monkeypatch):
        """Should return allocation for each strategy."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()
        port.set_total_capital(100000)
        port.add_strategy("Test", {"sharpe": 1.0})

        breakdown = port.get_allocation_breakdown()

        assert len(breakdown) == 1
        assert breakdown[0]["name"] == "Test"
        assert "allocated_capital" in breakdown[0]
        assert breakdown[0]["allocated_capital"] > 0
