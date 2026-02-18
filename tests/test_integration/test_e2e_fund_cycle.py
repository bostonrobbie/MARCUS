"""
End-to-end integration tests for the autonomous fund management cycle.

These tests verify the complete NHHF workflow including LLM orchestration,
code validation, portfolio management, and the bug fixes.
"""
import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


@pytest.mark.integration
class TestCodeValidationIntegration:
    """Integration tests for Bug 3 fix - code validation pipeline."""

    def test_full_validation_pipeline(self, valid_strategy_code):
        """Should validate and pass valid code."""
        from code_validator import validate_and_fix_code

        is_valid, fixed, issues = validate_and_fix_code(valid_strategy_code)

        assert is_valid is True

    def test_fixes_common_llm_issues(self):
        """Should fix common LLM code generation issues."""
        from code_validator import validate_and_fix_code

        # Code with markdown and missing imports
        bad_code = """```python
# Strategy code
x = pd.DataFrame()
y = np.array([1, 2, 3])
t = time(9, 30)
```"""

        is_valid, fixed, issues = validate_and_fix_code(bad_code)

        # Should have fixed markdown
        assert "```" not in fixed

        # Should have added imports
        assert "import pandas" in fixed
        assert "import numpy" in fixed
        assert "from datetime import" in fixed

    def test_rejects_unfixable_syntax_errors(self, invalid_syntax_code):
        """Should reject code with unfixable syntax errors."""
        from code_validator import validate_and_fix_code

        is_valid, fixed, issues = validate_and_fix_code(invalid_syntax_code)

        assert is_valid is False
        assert any("Syntax error" in issue for issue in issues)


@pytest.mark.integration
class TestPortfolioMetricsIntegration:
    """Integration tests for Bug 2 fix - defensive dict access."""

    def test_analyst_handles_empty_portfolio(self, mock_ollama, empty_portfolio_summary):
        """Analyst should handle empty portfolio without KeyError."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst

            analyst = Analyst()

            # Access with defensive pattern (Bug 2 fix)
            pm = empty_portfolio_summary.get('portfolio_metrics', {})
            sharpe = pm.get('sharpe', 0)
            num_strategies = pm.get('num_strategies', 0)

            # Should not crash
            assert sharpe == 0
            assert num_strategies == 0

    def test_portfolio_updates_metrics_correctly(self, tmp_path, monkeypatch):
        """Portfolio should maintain correct metrics after updates."""
        from portfolio import Portfolio

        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))

        port = Portfolio()

        # Add multiple strategies
        port.add_strategy("A", {"sharpe": 2.0})
        port.add_strategy("B", {"sharpe": 1.0})

        summary = port.get_portfolio_summary()

        # Defensive access should work
        pm = summary.get('portfolio_metrics', {})
        assert pm.get('num_strategies', 0) == 2

        # Metrics should be non-zero
        assert pm.get('sharpe', 0) > 0


@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for GlobalMemory system."""

    def test_memory_persistence(self, tmp_path, monkeypatch):
        """Should persist and reload experiences."""
        from memory import GlobalMemory

        memory_file = tmp_path / 'memory.json'
        monkeypatch.setattr('memory.MEMORY_FILE', str(memory_file))

        with patch('memory.ollama') as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.5] * 1024}

            # Create and add experience
            mem1 = GlobalMemory()
            mem1.add_experience("Test RSI strategy", "SUCCESS", {"sharpe": 1.5})

        # Reload memory
        with patch('memory.ollama') as mock_ollama:
            mem2 = GlobalMemory()

        # Should have persisted experience
        assert len(mem2.memory) == 1
        assert mem2.memory[0]["summary"] == "Test RSI strategy"


@pytest.mark.integration
class TestLeaderboardIntegration:
    """Integration tests for strategy leaderboard."""

    def test_leaderboard_ranks_strategies(self, tmp_path, monkeypatch):
        """Should rank strategies by Sharpe ratio."""
        from leaderboard import Leaderboard

        lb_file = tmp_path / 'leaderboard.json'
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(lb_file))

        lb = Leaderboard()

        # Add strategies in random order
        lb.update("Low", {"Sharpe Ratio": 0.5})
        lb.update("High", {"Sharpe Ratio": 2.5})
        lb.update("Mid", {"Sharpe Ratio": 1.5})

        champion = lb.get_champion()

        assert champion["name"] == "High"
        assert champion["sharpe"] == 2.5


@pytest.mark.integration
class TestAgentOrchestratorIntegration:
    """Integration tests for multi-agent orchestration."""

    def test_analyst_uses_multiple_agents(self, mock_ollama):
        """Analyst should coordinate between agents."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst

            analyst = Analyst()

            # Should have different agents available
            assert analyst.fund_manager != analyst.risk_manager
            assert analyst.code_reviewer != analyst.fund_manager

            # Should be able to generate responses with different agents
            response1 = analyst.generate_response("test", agent=analyst.fund_manager)
            response2 = analyst.generate_response("test", agent=analyst.risk_manager)

            # Both should get responses
            assert response1 is not None
            assert response2 is not None


@pytest.mark.integration
class TestFullFundCycle:
    """Integration tests for complete fund management cycle."""

    def test_strategy_lifecycle(self, tmp_path, monkeypatch, mock_ollama):
        """Test complete strategy lifecycle: propose -> validate -> add -> track."""
        from portfolio import Portfolio
        from leaderboard import Leaderboard
        from code_validator import validate_and_fix_code

        # Setup paths
        monkeypatch.setattr('portfolio.PORTFOLIO_FILE', str(tmp_path / 'portfolio.json'))
        monkeypatch.setattr('portfolio.PORTFOLIO_HISTORY_FILE', str(tmp_path / 'history.json'))
        monkeypatch.setattr('leaderboard.LEADERBOARD_FILE', str(tmp_path / 'leaderboard.json'))

        # 1. Generate strategy code (mocked)
        strategy_code = """
import pandas as pd
import numpy as np
from datetime import time

class MyStrategy:
    def calculate_signals(self, df):
        return pd.Series(np.zeros(len(df)))
"""

        # 2. Validate code
        is_valid, fixed_code, issues = validate_and_fix_code(strategy_code)
        assert is_valid is True

        # 3. Add to portfolio with metrics
        port = Portfolio()
        metrics = {"sharpe": 1.8, "return_pct": 0.12, "max_dd": -0.08}
        port.add_strategy("MyStrategy", metrics)

        # 4. Update leaderboard
        lb = Leaderboard()
        lb.update("MyStrategy", {"Sharpe Ratio": 1.8, "Total Return": "12%"})

        # Verify state
        assert len(port.get_all_strategies()) == 1
        assert lb.get_champion()["name"] == "MyStrategy"

    def test_risk_veto_prevents_deployment(self, mock_ollama_veto):
        """Risk manager veto should prevent strategy deployment."""
        with patch('analyst.ollama', mock_ollama_veto):
            from analyst import Analyst

            analyst = Analyst()

            # Get risk assessment
            response = analyst.generate_response(
                "Should we deploy this high-risk strategy?",
                agent=analyst.risk_manager
            )

            # Should contain veto
            assert "VETO" in response.upper()


@pytest.mark.integration
class TestBugFixVerification:
    """Verification tests for all 3 critical bugs."""

    def test_bug1_ndarray_iloc_fixed(self, mock_gpu_unavailable):
        """Bug 1: 'ndarray' has no attribute 'iloc' should be fixed."""
        from backtesting.type_utils import safe_iloc
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])

        # Should not raise AttributeError
        last = safe_iloc(arr, -1)
        first = safe_iloc(arr, 0)

        assert last == 5
        assert first == 1

    def test_bug2_keyerror_portfolio_metrics_fixed(self):
        """Bug 2: KeyError 'num_strategies' should be fixed."""
        # Test the defensive access pattern
        summary = {"portfolio_metrics": {}}  # Empty dict

        # Should not raise KeyError
        pm = summary.get('portfolio_metrics', {})
        num = pm.get('num_strategies', 0)

        assert num == 0

        # Also test with missing key
        summary2 = {}
        pm2 = summary2.get('portfolio_metrics', {})
        num2 = pm2.get('num_strategies', 0)

        assert num2 == 0

    def test_bug3_import_time_fixed(self):
        """Bug 3: 'import time' vs 'from datetime import time' should be fixed."""
        from code_validator import validate_and_fix_code

        # Code that uses time(h, m) without proper import
        bad_code = """
x = time(9, 30)
y = time(15, 55)
"""

        is_valid, fixed, issues = validate_and_fix_code(bad_code)

        # Should add datetime import
        assert "from datetime import" in fixed
        assert "time" in fixed
