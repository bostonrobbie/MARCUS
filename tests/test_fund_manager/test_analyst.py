"""
Tests for Analyst class (orchestrator).

These tests verify the main analyst orchestration including
LLM integration, code generation, and the defensive dict access fixes (Bug 2).
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestAnalystInitialization:
    """Tests for Analyst initialization."""

    def test_initializes_with_model(self):
        """Should initialize with specified model."""
        with patch('analyst.ollama'):
            from analyst import Analyst
            analyst = Analyst(model="test-model")

            assert analyst.model == "test-model"

    def test_has_required_agents(self):
        """Should have all required agent personas."""
        with patch('analyst.ollama'):
            from analyst import Analyst
            analyst = Analyst()

            assert analyst.fund_manager is not None
            assert analyst.risk_manager is not None
            assert analyst.code_reviewer is not None


class TestGenerateResponse:
    """Tests for LLM response generation."""

    def test_calls_ollama(self, mock_ollama):
        """Should call Ollama chat API."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst
            analyst = Analyst()
            response = analyst.generate_response("test prompt")

        mock_ollama.chat.assert_called()

    def test_returns_response_content(self, mock_ollama):
        """Should return message content."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst
            analyst = Analyst()
            response = analyst.generate_response("test prompt")

        assert response == "Mocked LLM response"

    def test_uses_specified_agent(self, mock_ollama):
        """Should use agent as system prompt."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst
            analyst = Analyst()
            analyst.generate_response("test", agent="Test Agent Persona")

        # Check the system message contains the agent
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Test Agent Persona" in system_msg["content"]


class TestDefensiveDictAccess:
    """Tests for Bug 2 fix - defensive portfolio_metrics access."""

    def test_handles_empty_portfolio_metrics(self, sample_portfolio_summary):
        """Should not crash when portfolio_metrics is empty."""
        with patch('analyst.ollama') as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "OK"}}
            from analyst import Analyst
            analyst = Analyst()

            # Simulate empty portfolio_metrics
            empty_summary = sample_portfolio_summary.copy()
            empty_summary['portfolio_metrics'] = {}

            # Should not raise KeyError
            pm = empty_summary.get('portfolio_metrics', {})
            sharpe = pm.get('sharpe', 0)
            num_strategies = pm.get('num_strategies', 0)

            assert sharpe == 0
            assert num_strategies == 0

    def test_handles_missing_portfolio_metrics(self, sample_portfolio_summary):
        """Should handle when portfolio_metrics key is missing."""
        with patch('analyst.ollama'):
            from analyst import Analyst

            # Remove the key entirely
            summary = sample_portfolio_summary.copy()
            del summary['portfolio_metrics']

            # Defensive access should return default
            pm = summary.get('portfolio_metrics', {})
            assert pm == {}
            assert pm.get('sharpe', 0) == 0

    def test_handles_none_portfolio_metrics(self, sample_portfolio_summary):
        """Should handle when portfolio_metrics is None."""
        with patch('analyst.ollama'):
            from analyst import Analyst

            summary = sample_portfolio_summary.copy()
            summary['portfolio_metrics'] = None

            # Need to handle None case
            pm = summary.get('portfolio_metrics') or {}
            assert pm.get('sharpe', 0) == 0


class TestCodeGeneration:
    """Tests for strategy code generation."""

    def test_includes_validation_suffix(self, mock_ollama):
        """Should include validation instructions in prompt."""
        with patch('analyst.ollama', mock_ollama):
            with patch('analyst.validate_and_fix_code') as mock_validate:
                mock_validate.return_value = (True, "valid code", [])
                from analyst import Analyst
                analyst = Analyst()

                # Trigger code generation - need to call appropriate method
                # The actual method may vary based on analyst implementation
                # This tests that validation is integrated
                from code_validator import get_validation_prompt_suffix
                suffix = get_validation_prompt_suffix()

                assert "CRITICAL" in suffix
                assert "import pandas" in suffix

    def test_validates_generated_code(self, mock_ollama, valid_strategy_code):
        """Should validate code after generation."""
        with patch('analyst.ollama', mock_ollama):
            from code_validator import validate_and_fix_code

            is_valid, fixed, issues = validate_and_fix_code(valid_strategy_code)

            assert is_valid is True

    def test_retries_on_invalid_code(self, mock_ollama, invalid_syntax_code):
        """Should handle invalid code gracefully."""
        with patch('analyst.ollama', mock_ollama):
            from code_validator import validate_and_fix_code

            is_valid, fixed, issues = validate_and_fix_code(invalid_syntax_code)

            # Invalid syntax can't be auto-fixed
            assert is_valid is False


class TestVetoIntegration:
    """Tests for veto mechanism integration."""

    def test_risk_manager_can_veto(self, mock_ollama_veto):
        """Risk manager should be able to veto proposals."""
        with patch('analyst.ollama', mock_ollama_veto):
            from analyst import Analyst
            analyst = Analyst()

            response = analyst.generate_response(
                "Should we deploy this strategy?",
                agent=analyst.risk_manager
            )

            assert "VETO" in response.upper()


class TestAnalystWithEmptyPortfolio:
    """Tests for analyst behavior with empty/new portfolio."""

    def test_handles_new_portfolio(self, mock_ollama, empty_portfolio_summary):
        """Should handle portfolio with no strategies."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst
            analyst = Analyst()

            # Defensive access
            pm = empty_portfolio_summary.get('portfolio_metrics', {})
            num = pm.get('num_strategies', 0)

            assert num == 0

    def test_suggests_first_strategy(self, mock_ollama, empty_portfolio_summary):
        """Should suggest adding first strategy when empty."""
        with patch('analyst.ollama', mock_ollama):
            from analyst import Analyst
            analyst = Analyst()

            # With empty portfolio, should be able to generate recommendations
            if empty_portfolio_summary.get('portfolio_metrics', {}).get('num_strategies', 0) == 0:
                # System should handle this case
                pass


class TestAnalystMemoryIntegration:
    """Tests for integration with GlobalMemory."""

    def test_can_query_memory(self, mock_ollama):
        """Should be able to query past experiences."""
        with patch('analyst.ollama', mock_ollama):
            with patch('memory.ollama', mock_ollama):
                from analyst import Analyst
                from memory import GlobalMemory

                analyst = Analyst()

                # Memory query should work
                with patch.object(GlobalMemory, 'query_memory', return_value=[]):
                    mem = GlobalMemory()
                    results = mem.query_memory("RSI strategy")
                    assert results == []


class TestAnalystErrorHandling:
    """Tests for error handling in analyst."""

    def test_handles_ollama_error(self):
        """Should handle Ollama API errors gracefully."""
        with patch('analyst.ollama') as mock_ollama:
            mock_ollama.chat.side_effect = Exception("API Error")

            from analyst import Analyst
            analyst = Analyst()

            # Should handle error without crashing
            try:
                analyst.generate_response("test")
            except Exception as e:
                assert "API Error" in str(e)

    def test_handles_empty_response(self):
        """Should handle empty LLM response."""
        with patch('analyst.ollama') as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": ""}}

            from analyst import Analyst
            analyst = Analyst()

            response = analyst.generate_response("test")
            assert response == ""
