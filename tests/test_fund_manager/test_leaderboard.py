"""
Tests for Leaderboard class.

These tests verify the strategy ranking and tracking system.
"""
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestLeaderboardInitialization:
    """Tests for Leaderboard initialization."""

    def test_creates_empty_leaderboard(self, tmp_path, monkeypatch):
        """Should initialize with empty list if file doesn't exist."""
        from leaderboard import Leaderboard

        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(tmp_path / 'lb.json'))
        lb = Leaderboard()

        assert lb.leaders == []

    def test_loads_existing_leaderboard(self, tmp_path, monkeypatch):
        """Should load existing leaderboard."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        existing = [{"name": "Test", "sharpe": 2.0, "return": 0.15}]
        lb_file.write_text(json.dumps(existing))

        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))
        lb = Leaderboard()

        assert len(lb.leaders) == 1
        assert lb.leaders[0]["name"] == "Test"

    def test_handles_corrupted_file(self, tmp_path, monkeypatch):
        """Should return empty on corrupted file."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        lb_file.write_text("invalid json")

        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))
        lb = Leaderboard()

        assert lb.leaders == []


class TestLeaderboardUpdate:
    """Tests for updating the leaderboard."""

    def test_adds_strategy(self, tmp_path, monkeypatch):
        """Should add new strategy to leaderboard."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("TestStrategy", {"Sharpe Ratio": 1.5, "Total Return": "10%"})

        assert len(lb.leaders) == 1
        assert lb.leaders[0]["name"] == "TestStrategy"
        assert lb.leaders[0]["sharpe"] == 1.5

    def test_sorts_by_sharpe_descending(self, tmp_path, monkeypatch):
        """Should maintain leaderboard sorted by Sharpe."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("Low", {"Sharpe Ratio": 0.5})
        lb.update("High", {"Sharpe Ratio": 2.0})
        lb.update("Mid", {"Sharpe Ratio": 1.0})

        assert lb.leaders[0]["name"] == "High"
        assert lb.leaders[1]["name"] == "Mid"
        assert lb.leaders[2]["name"] == "Low"

    def test_keeps_top_5_only(self, tmp_path, monkeypatch):
        """Should only keep top 5 strategies."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        for i in range(10):
            lb.update(f"Strat{i}", {"Sharpe Ratio": float(i)})

        assert len(lb.leaders) == 5
        # Top 5 should be strategies 9, 8, 7, 6, 5
        assert lb.leaders[0]["sharpe"] == 9.0

    def test_parses_string_returns(self, tmp_path, monkeypatch):
        """Should parse percentage strings."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("Test", {"Sharpe Ratio": 1.0, "Total Return": "15.5%"})

        assert lb.leaders[0]["return"] == 15.5

    def test_stores_code_snippet(self, tmp_path, monkeypatch):
        """Should store code snippet."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        snippet = "def calculate_signals(): return 1"
        lb.update("Test", {"Sharpe Ratio": 1.0}, code_snippet=snippet)

        assert lb.leaders[0]["logic_summary"] == snippet

    def test_truncates_long_snippets(self, tmp_path, monkeypatch):
        """Should truncate snippets longer than 500 chars."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        long_snippet = "x" * 1000
        lb.update("Test", {"Sharpe Ratio": 1.0}, code_snippet=long_snippet)

        assert len(lb.leaders[0]["logic_summary"]) == 500

    def test_saves_to_file(self, tmp_path, monkeypatch):
        """Should persist to file after update."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("Test", {"Sharpe Ratio": 1.5})

        # Read file directly
        saved = json.loads(lb_file.read_text())
        assert len(saved) == 1

    def test_handles_missing_metrics(self, tmp_path, monkeypatch):
        """Should handle missing metric fields."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("Test", {})  # Empty metrics

        assert lb.leaders[0]["sharpe"] == 0
        assert lb.leaders[0]["return"] == 0


class TestGetChampion:
    """Tests for get_champion method."""

    def test_returns_top_strategy(self, tmp_path, monkeypatch):
        """Should return strategy with highest Sharpe."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        lb.update("Second", {"Sharpe Ratio": 1.0})
        lb.update("Champion", {"Sharpe Ratio": 2.5})
        lb.update("Third", {"Sharpe Ratio": 0.5})

        champion = lb.get_champion()

        assert champion["name"] == "Champion"
        assert champion["sharpe"] == 2.5

    def test_returns_none_when_empty(self, tmp_path, monkeypatch):
        """Should return None when leaderboard is empty."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'lb.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()
        champion = lb.get_champion()

        assert champion is None
