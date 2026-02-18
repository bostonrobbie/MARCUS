"""
Tests for agent classes.

These tests verify the 9 specialized AI agents used in the fund management system.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestAgentBase:
    """Tests for base agent functionality."""

    def test_agent_module_imports(self):
        """Should be able to import agents module."""
        import agents

        # Check key agent names are defined
        assert hasattr(agents, 'FUND_MANAGER')
        assert hasattr(agents, 'RISK_MANAGER')
        assert hasattr(agents, 'CODE_REVIEWER')


class TestFundManagerAgent:
    """Tests for Fund Manager agent."""

    def test_fund_manager_defined(self):
        """Should have Fund Manager persona defined."""
        from agents import FUND_MANAGER

        assert "Fund Manager" in FUND_MANAGER or "fund" in FUND_MANAGER.lower()

    def test_fund_manager_has_strategy_focus(self):
        """Fund Manager should focus on strategy coordination."""
        from agents import FUND_MANAGER

        # Should mention strategy, portfolio, or coordination
        fm_lower = FUND_MANAGER.lower()
        assert any(word in fm_lower for word in ['strategy', 'portfolio', 'coordinate', 'lead'])


class TestRiskManagerAgent:
    """Tests for Risk Manager agent."""

    def test_risk_manager_defined(self):
        """Should have Risk Manager persona defined."""
        from agents import RISK_MANAGER

        assert "Risk" in RISK_MANAGER or "risk" in RISK_MANAGER.lower()

    def test_risk_manager_has_risk_focus(self):
        """Risk Manager should focus on risk assessment."""
        from agents import RISK_MANAGER

        rm_lower = RISK_MANAGER.lower()
        assert any(word in rm_lower for word in ['risk', 'drawdown', 'exposure', 'protect'])


class TestCodeReviewerAgent:
    """Tests for Code Reviewer agent."""

    def test_code_reviewer_defined(self):
        """Should have Code Reviewer persona defined."""
        from agents import CODE_REVIEWER

        assert "Code" in CODE_REVIEWER or "code" in CODE_REVIEWER.lower()

    def test_code_reviewer_has_review_focus(self):
        """Code Reviewer should focus on code quality."""
        from agents import CODE_REVIEWER

        cr_lower = CODE_REVIEWER.lower()
        assert any(word in cr_lower for word in ['code', 'review', 'quality', 'bug', 'syntax'])


class TestQuantResearcherAgent:
    """Tests for Quant Researcher agent."""

    def test_quant_researcher_defined(self):
        """Should have Quant Researcher persona defined."""
        from agents import QUANT_RESEARCHER

        assert "Quant" in QUANT_RESEARCHER or "research" in QUANT_RESEARCHER.lower()

    def test_quant_researcher_has_research_focus(self):
        """Quant Researcher should focus on alpha generation."""
        from agents import QUANT_RESEARCHER

        qr_lower = QUANT_RESEARCHER.lower()
        assert any(word in qr_lower for word in ['alpha', 'signal', 'research', 'indicator'])


class TestBacktestEngineerAgent:
    """Tests for Backtest Engineer agent."""

    def test_backtest_engineer_defined(self):
        """Should have Backtest Engineer persona defined."""
        from agents import BACKTEST_ENGINEER

        assert "Backtest" in BACKTEST_ENGINEER or "backtest" in BACKTEST_ENGINEER.lower()


class TestPortfolioManagerAgent:
    """Tests for Portfolio Manager agent."""

    def test_portfolio_manager_defined(self):
        """Should have Portfolio Manager persona defined."""
        from agents import PORTFOLIO_MANAGER

        assert "Portfolio" in PORTFOLIO_MANAGER or "portfolio" in PORTFOLIO_MANAGER.lower()


class TestMarketRegimeAnalystAgent:
    """Tests for Market Regime Analyst agent."""

    def test_market_regime_analyst_defined(self):
        """Should have Market Regime Analyst persona defined."""
        from agents import MARKET_REGIME_ANALYST

        assert "Regime" in MARKET_REGIME_ANALYST or "market" in MARKET_REGIME_ANALYST.lower()


class TestMetaLearnerAgent:
    """Tests for Meta-Learner agent."""

    def test_meta_learner_defined(self):
        """Should have Meta-Learner persona defined."""
        from agents import META_LEARNER

        # Meta-learner should exist
        assert META_LEARNER is not None


class TestCodeGeneratorAgent:
    """Tests for Code Generator agent."""

    def test_code_generator_defined(self):
        """Should have Code Generator persona defined."""
        from agents import CODE_GENERATOR

        assert "Code" in CODE_GENERATOR or "code" in CODE_GENERATOR.lower()


class TestAgentIntegration:
    """Integration tests for agent usage."""

    def test_all_agents_are_strings(self):
        """All agent personas should be strings."""
        from agents import (
            FUND_MANAGER, RISK_MANAGER, CODE_REVIEWER,
            QUANT_RESEARCHER, BACKTEST_ENGINEER, PORTFOLIO_MANAGER,
            MARKET_REGIME_ANALYST, META_LEARNER, CODE_GENERATOR
        )

        agents = [
            FUND_MANAGER, RISK_MANAGER, CODE_REVIEWER,
            QUANT_RESEARCHER, BACKTEST_ENGINEER, PORTFOLIO_MANAGER,
            MARKET_REGIME_ANALYST, META_LEARNER, CODE_GENERATOR
        ]

        for agent in agents:
            assert isinstance(agent, str)
            assert len(agent) > 0

    def test_agents_are_distinct(self):
        """Each agent should have unique persona."""
        from agents import (
            FUND_MANAGER, RISK_MANAGER, CODE_REVIEWER,
            QUANT_RESEARCHER, BACKTEST_ENGINEER, PORTFOLIO_MANAGER,
            MARKET_REGIME_ANALYST, META_LEARNER, CODE_GENERATOR
        )

        agents = [
            FUND_MANAGER, RISK_MANAGER, CODE_REVIEWER,
            QUANT_RESEARCHER, BACKTEST_ENGINEER, PORTFOLIO_MANAGER,
            MARKET_REGIME_ANALYST, META_LEARNER, CODE_GENERATOR
        ]

        # All should be unique
        assert len(agents) == len(set(agents))
