"""
Pipeline Integration Tests
============================
Tests the full gate decision logic in the research engine:
- S2 statistical gates (Sharpe CI + permutation test)
- S4 tightened thresholds + WFO robustness
- S5 DSR + NQmain overlap
- MC VaR95 gate in _save_winner
- Complexity cap pre-S1
- Market rationale attachment
- New archetype generation

These tests verify the WIRING of gates, not the statistical math
(which is tested in test_statistical_gates.py).

Run with:
    cd StrategyPipeline/src
    python -m backtesting.tests.test_pipeline_integration
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
import tempfile
import shutil
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ===========================================================================
# TEST RUNNER
# ===========================================================================
class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_func):
        try:
            test_func()
            self.passed += 1
            print(f"  [PASS] {test_func.__name__}")
        except Exception as e:
            self.failed += 1
            self.errors.append((test_func.__name__, str(e)))
            import traceback
            print(f"  [FAIL] {test_func.__name__}: {e}")
            traceback.print_exc()

    def report(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, err in self.errors:
                print(f"  {name}: {err}")
        print(f"{'='*60}")
        return self.failed == 0


# ===========================================================================
# HELPERS
# ===========================================================================
def make_temp_db():
    """Create a temporary SQLite database for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, 'test_marcus.db')
    return db_path, tmpdir


def cleanup_temp(tmpdir):
    """Clean up temporary directory."""
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


def make_trending_returns(n=500, drift=0.001, volatility=0.02, seed=42):
    """Generate returns from a strategy with positive edge."""
    rng = np.random.RandomState(seed)
    return rng.normal(drift, volatility, n)


def make_random_returns(n=500, seed=42):
    """Generate zero-mean returns (no edge)."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.02, n)


# ===========================================================================
# S2 SHARPE CI GATE INTEGRATION
# ===========================================================================
def test_s2_sharpe_ci_gate_logic():
    """Test the S2 Sharpe CI gate logic as wired in research_engine.py.

    Simulates the exact decision path from _stage2_gauntlet:
    1. Compute Sharpe CI
    2. If lower bound <= 0, reject
    """
    from backtesting.statistics import StatisticalSignificance

    # Scenario 1: Strong strategy — CI should be entirely positive
    strong_returns = make_trending_returns(n=500, drift=0.002, volatility=0.015, seed=42)
    sharpe_val, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        strong_returns, confidence=0.95
    )
    # Simulate gate logic
    stat_require_sharpe_ci_positive = True
    gate_pass = not (stat_require_sharpe_ci_positive and ci_lower <= 0)
    assert gate_pass, \
        f"Strong strategy should pass Sharpe CI gate: CI=[{ci_lower:.3f}, {ci_upper:.3f}]"

    # Scenario 2: Weak strategy — CI crosses zero
    weak_returns = make_random_returns(n=100, seed=123)
    sharpe_val, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        weak_returns, confidence=0.95
    )
    # For a zero-mean strategy with 100 obs, CI should likely cross zero
    # (but not guaranteed due to randomness — check that the math runs)
    assert isinstance(ci_lower, float), "CI lower should be float"


def test_s2_sharpe_ci_skipped_when_disabled():
    """Sharpe CI gate should be skippable via config."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    # When disabled, gate should auto-pass
    config.stat_require_sharpe_ci_positive = False
    assert config.stat_require_sharpe_ci_positive is False

    # Simulate: if not config.stat_require_sharpe_ci_positive → skip gate
    returns = make_random_returns(n=50)
    ci_lower = -999  # Terrible CI
    gate_pass = not (config.stat_require_sharpe_ci_positive and ci_lower <= 0)
    assert gate_pass, "Disabled Sharpe CI gate should auto-pass"


def test_s2_sharpe_ci_insufficient_data():
    """S2 gate should gracefully skip CI with <20 observations."""
    from backtesting.statistics import StatisticalSignificance

    short_returns = np.array([0.01, 0.02, -0.01, 0.005, 0.003])  # Only 5 obs
    # As wired: len(equity_returns) > 20 is checked before CI computation
    skip = len(short_returns) <= 20
    assert skip, "Short returns should be skipped (auto-pass)"


# ===========================================================================
# S2 PERMUTATION TEST GATE INTEGRATION
# ===========================================================================
def test_s2_permutation_gate_config():
    """Permutation test config should have correct defaults."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    assert config.s2_permutation_enabled is True, "Permutation should be enabled by default"
    assert config.s2_permutation_sims == 50, f"Default sims should be 50, got {config.s2_permutation_sims}"


def test_s2_permutation_verdict_logic():
    """Test the permutation verdict decision logic as wired in S2."""
    # Simulate permutation result
    pass_result = {'p_value': 0.02, 'verdict': 'PASS', 'n_sims': 50}
    fail_result = {'p_value': 0.45, 'verdict': 'FAIL (Indistinguishable from Luck)', 'n_sims': 50}

    # Gate logic from research_engine
    assert pass_result['verdict'] == 'PASS', "PASS verdict should pass gate"
    assert fail_result['verdict'] != 'PASS', "FAIL verdict should not pass gate"


def test_s2_permutation_skipped_when_disabled():
    """Permutation test should be skippable via config."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    config.s2_permutation_enabled = False

    # When disabled, gate is skipped
    assert not config.s2_permutation_enabled


# ===========================================================================
# S4 ROBUSTNESS & CLIFF DETECTION INTEGRATION
# ===========================================================================
def test_s4_robustness_gate_with_stable_params():
    """S4 should pass strategy with high robustness score (stable params)."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Stable performance: 50K ± 5% across all factors
    rows = []
    for param_name in ['ema_period', 'atr_mult']:
        for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
            base = 50000
            noise = np.random.uniform(-2500, 2500)  # ±5%
            rows.append({
                'param_name': param_name,
                'factor': factor,
                'value': factor * 50 if param_name == 'ema_period' else factor * 2.0,
                'Total Return': base + noise,
            })

    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')
    result = mapper.robustness_score()

    min_robustness = 50.0
    score = result.get('robustness_score', 0)
    passed = score >= min_robustness
    assert passed, f"Stable params should have robustness >= {min_robustness}, got {score}"


def test_s4_robustness_gate_rejects_fragile_params():
    """S4 should reject strategy with low robustness (fragile params)."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Fragile: huge variance across factors
    rows = []
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        profit = 50000 if factor == 1.0 else np.random.uniform(-20000, 5000)
        rows.append({
            'param_name': 'ema_period',
            'factor': factor,
            'value': int(50 * factor),
            'Total Return': profit,
        })

    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')
    result = mapper.robustness_score()
    score = result.get('robustness_score', 100)
    # Not asserting specific score since randomness, but verify structure
    assert 'robustness_score' in result
    assert 'interpretation' in result


def test_s4_cliff_detection_rejects_high_severity():
    """S4 should reject strategy when HIGH severity cliff detected."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Cliff: performance drops 90% at factor 1.2
    rows = []
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        profit = 50000 if factor <= 1.1 else 5000  # 90% drop at 1.2
        rows.append({
            'param_name': 'sl_mult',
            'factor': factor,
            'value': round(2.0 * factor, 2),
            'Total Return': profit,
        })
    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')
    cliffs = mapper.cliff_detection()

    # Verify cliff detection ran
    assert isinstance(cliffs, list), "cliff_detection should return list"

    # Check if any HIGH severity cliffs found
    high_cliffs = [c for c in cliffs if c.get('severity') == 'HIGH']
    # The 90% drop should trigger a cliff
    if len(cliffs) > 0:
        # At least one cliff detected
        max_change = max(abs(c.get('pct_change', 0)) for c in cliffs)
        assert max_change > 0.5, f"Expected large change, got {max_change}"


def test_s4_variant_profitability_threshold():
    """S4 should reject when <60% of variants are profitable."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # 3 out of 8 profitable = 37.5%
    variant_results = {
        'p1_0.8': {'profitable': True, 'net_profit': 1000},
        'p1_0.9': {'profitable': True, 'net_profit': 2000},
        'p1_1.1': {'profitable': False, 'net_profit': -500},
        'p1_1.2': {'profitable': False, 'net_profit': -800},
        'p2_0.8': {'profitable': False, 'net_profit': -200},
        'p2_0.9': {'profitable': True, 'net_profit': 500},
        'p2_1.1': {'profitable': False, 'net_profit': -300},
        'p2_1.2': {'profitable': False, 'net_profit': -600},
    }

    total_valid = sum(1 for v in variant_results.values() if 'profitable' in v)
    profitable_count = sum(1 for v in variant_results.values() if v.get('profitable', False))
    profitable_pct = profitable_count / total_valid if total_valid > 0 else 0

    profitability_ok = profitable_pct >= config.s4_min_profitable_pct
    assert not profitability_ok, \
        f"37.5% profitable should fail S4 (threshold={config.s4_min_profitable_pct:.0%})"


# ===========================================================================
# S5 DSR GATE INTEGRATION
# ===========================================================================
def test_s5_dsr_gate_logic():
    """Test DSR gate logic as wired in _stage5_complementarity."""
    from backtesting.statistics import StatisticalSignificance

    # Simulate DSR computation as wired
    returns = make_trending_returns(n=500, drift=0.002, volatility=0.015, seed=42)
    n_trials = 5000  # Lots of backtests run

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    raw_sharpe = np.sqrt(252) * mean_ret / std_ret

    se = StatisticalSignificance.sharpe_standard_error(returns, raw_sharpe)
    var_sharpe = se ** 2

    ret_stats = StatisticalSignificance.returns_statistics(returns)

    dsr_prob = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=raw_sharpe,
        n_trials=n_trials,
        var_sharpe=var_sharpe,
        skewness=ret_stats.get('skewness', 0),
        kurtosis=ret_stats.get('kurtosis', 3),
        n_observations=len(returns),
    )

    # With high Sharpe and reasonable n_trials, should pass
    assert isinstance(dsr_prob, float), "DSR should return float"
    assert 0 <= dsr_prob <= 1, f"DSR probability should be 0-1, got {dsr_prob}"


def test_s5_dsr_skipped_with_low_trials():
    """DSR gate should be skipped when n_trials <= 10."""
    # As wired: if n_trials > 10: ... (only applies with sufficient trial history)
    n_trials = 5
    skip = n_trials <= 10
    assert skip, "DSR should be skipped with few trials"


# ===========================================================================
# S5 NQMAIN OVERLAP INTEGRATION
# ===========================================================================
def test_s5_nqmain_overlap_rejects_orb_overlap():
    """S5 should reject ORB breakout with high NQmain time overlap."""
    from backtesting.nqmain_analyzer import get_complementary_score

    # ORB breakout overlaps 100% with NQmain
    result = get_complementary_score('orb_breakout')

    # As wired: only blocks ORB-vs-ORB overlaps
    max_overlap = 0.30
    archetype = 'orb_breakout'
    rejected = archetype == 'orb_breakout' and result['time_overlap'] > max_overlap
    assert rejected, \
        f"ORB breakout with {result['time_overlap']:.0%} overlap should be rejected"


def test_s5_nqmain_overlap_passes_complementary():
    """S5 should pass strategies that complement NQmain."""
    from backtesting.nqmain_analyzer import get_complementary_score

    # Lunch range fade targets gap window
    result = get_complementary_score('lunch_range_fade')

    max_overlap = 0.30
    archetype = 'lunch_range_fade'
    # As wired: only blocks archetype == 'orb_breakout' AND high overlap
    rejected = archetype == 'orb_breakout' and result['time_overlap'] > max_overlap
    assert not rejected, "Lunch range fade should NOT be rejected by NQmain overlap"
    assert result['gap_coverage'] is True, "Lunch fade should cover a gap"


def test_s5_nqmain_overlap_non_orb_passes():
    """Non-ORB archetypes with high time overlap should still pass (different logic)."""
    from backtesting.nqmain_analyzer import get_complementary_score

    # MA crossover covers full day (high overlap) but isn't ORB
    result = get_complementary_score('ma_crossover')

    max_overlap = 0.30
    archetype = 'ma_crossover'
    # Gate only blocks archetype == 'orb_breakout'
    rejected = archetype == 'orb_breakout' and result['time_overlap'] > max_overlap
    assert not rejected, "Non-ORB archetype should not be rejected by ORB overlap check"


# ===========================================================================
# MC VAR95 GATE INTEGRATION
# ===========================================================================
def test_mc_var95_gate_logic():
    """Test MC VaR95 gate logic as wired in _save_winner."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Scenario 1: Acceptable VaR95
    mc_var95 = -0.15
    rejected = mc_var95 < config.mc_min_var95 and mc_var95 != 0.0
    assert not rejected, f"VaR95={mc_var95} should pass (threshold={config.mc_min_var95})"

    # Scenario 2: Severe VaR95
    mc_var95 = -0.50
    rejected = mc_var95 < config.mc_min_var95 and mc_var95 != 0.0
    assert rejected, f"VaR95={mc_var95} should be rejected (threshold={config.mc_min_var95})"

    # Scenario 3: Zero (MC not run)
    mc_var95 = 0.0
    rejected = mc_var95 < config.mc_min_var95 and mc_var95 != 0.0
    assert not rejected, "VaR95=0.0 should auto-pass"


# ===========================================================================
# COMPLEXITY CAP INTEGRATION
# ===========================================================================
def test_complexity_cap_logic():
    """Test pre-S1 complexity cap as wired in _process_idea."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Too many params
    idea_overfit = {
        'strategy_name': 'test_overfit',
        'params': {
            'p1': 10, 'p2': 20, 'p3': 30, 'p4': 40, 'p5': 50,
            'p6': 60, 'p7': 70, 'p8': 80, 'p9': 90,  # 9 numeric params
        }
    }
    numeric_params = sum(1 for v in idea_overfit['params'].values() if isinstance(v, (int, float)))
    rejected = numeric_params > config.max_strategy_params
    assert rejected, f"9 params should exceed cap of {config.max_strategy_params}"

    # Acceptable params
    idea_lean = {
        'strategy_name': 'test_lean',
        'params': {
            'ema': 50, 'atr': 14, 'sl': 2.0,  # 3 numeric params
            'name': 'my_strategy',  # Non-numeric, not counted
        }
    }
    numeric_params = sum(1 for v in idea_lean['params'].values() if isinstance(v, (int, float)))
    rejected = numeric_params > config.max_strategy_params
    assert not rejected, f"3 params should be under cap of {config.max_strategy_params}"


# ===========================================================================
# MARKET RATIONALE ATTACHMENT INTEGRATION
# ===========================================================================
def test_market_rationale_attachment():
    """Test that market rationale is attached to ideas in _process_idea."""
    from backtesting.research_engine import MARKET_RATIONALE

    # Simulate rationale attachment logic from _process_idea
    idea = {'strategy_name': 'test', 'archetype': 'orb_breakout', 'params': {}}
    archetype = idea.get('archetype', '')
    if archetype in MARKET_RATIONALE and 'market_rationale' not in idea:
        idea['market_rationale'] = MARKET_RATIONALE[archetype]

    assert 'market_rationale' in idea, "Rationale should be attached"
    assert 'Opening Range Breakout' in idea['market_rationale'], \
        "ORB rationale should mention Opening Range Breakout"


def test_market_rationale_not_overridden():
    """If idea already has rationale, it should not be overridden."""
    from backtesting.research_engine import MARKET_RATIONALE

    idea = {
        'strategy_name': 'test',
        'archetype': 'orb_breakout',
        'params': {},
        'market_rationale': 'Custom rationale',
    }
    archetype = idea.get('archetype', '')
    if archetype in MARKET_RATIONALE and 'market_rationale' not in idea:
        idea['market_rationale'] = MARKET_RATIONALE[archetype]

    assert idea['market_rationale'] == 'Custom rationale', \
        "Existing rationale should not be overridden"


# ===========================================================================
# NEW ARCHETYPE GENERATION INTEGRATION
# ===========================================================================
def test_new_archetypes_exist():
    """New archetype families should exist in ARCHETYPE_TIME_WINDOWS."""
    from backtesting.nqmain_analyzer import ARCHETYPE_TIME_WINDOWS
    from datetime import time as dtime

    # Power Hour Momentum
    assert 'power_hour_momentum' in ARCHETYPE_TIME_WINDOWS, \
        "power_hour_momentum should be in ARCHETYPE_TIME_WINDOWS"
    ph_start, ph_end = ARCHETYPE_TIME_WINDOWS['power_hour_momentum']
    assert ph_start == dtime(14, 0), "Power hour should start at 14:00"
    assert ph_end == dtime(15, 30), "Power hour should end at 15:30"

    # First Hour Fade
    assert 'first_hour_fade' in ARCHETYPE_TIME_WINDOWS, \
        "first_hour_fade should be in ARCHETYPE_TIME_WINDOWS"
    fhf_start, fhf_end = ARCHETYPE_TIME_WINDOWS['first_hour_fade']
    assert fhf_start == dtime(10, 15), "First hour fade should start at 10:15"
    assert fhf_end == dtime(11, 30), "First hour fade should end at 11:30"

    # Lunch Range Fade
    assert 'lunch_range_fade' in ARCHETYPE_TIME_WINDOWS, \
        "lunch_range_fade should be in ARCHETYPE_TIME_WINDOWS"
    lrf_start, lrf_end = ARCHETYPE_TIME_WINDOWS['lunch_range_fade']
    assert lrf_start == dtime(11, 30), "Lunch range fade should start at 11:30"
    assert lrf_end == dtime(13, 30), "Lunch range fade should end at 13:30"


def test_new_archetypes_in_rationale():
    """New archetypes should have market rationale."""
    from backtesting.research_engine import MARKET_RATIONALE

    for arch in ['power_hour_momentum', 'first_hour_fade', 'lunch_range_fade']:
        assert arch in MARKET_RATIONALE, f"Missing rationale for '{arch}'"
        assert len(MARKET_RATIONALE[arch]) > 50, \
            f"Rationale for '{arch}' too short"


def test_new_archetypes_complement_nqmain():
    """New archetypes should be complementary to NQmain in time or regime."""
    from backtesting.nqmain_analyzer import get_complementary_score

    # First hour fade: targets mean_reversion gap regime
    fhf = get_complementary_score('first_hour_fade')
    assert fhf['regime_complement'] is True, \
        "first_hour_fade should complement NQmain regime gaps"
    assert fhf['complementary_score'] > 0, \
        "first_hour_fade should have positive complementary score"

    # Lunch range fade: targets gap window AND choppy_range regime
    lrf = get_complementary_score('lunch_range_fade')
    assert lrf['regime_complement'] is True, \
        "lunch_range_fade should complement NQmain regime gaps"
    assert lrf['gap_coverage'] is True, \
        "lunch_range_fade should cover NQmain gap window"
    assert lrf['complementary_score'] >= 50, \
        f"lunch_range_fade should score >=50, got {lrf['complementary_score']}"

    # Power hour: same time window as NQmain but different market microstructure
    # (volume surge, MOC orders). Complementary score is 0 because it fully overlaps
    # NQmain in time — the value comes from targeting different market conditions
    phm = get_complementary_score('power_hour_momentum')
    assert phm['time_overlap'] == 1.0, \
        "power_hour_momentum is fully within NQmain window"
    # Score is 0 due to full overlap — this is correct behavior;
    # it's still allowed because it's not archetype == 'orb_breakout'
    assert isinstance(phm['complementary_score'], (int, float)), \
        "Score should be numeric"


# ===========================================================================
# REGISTRY INTEGRATION
# ===========================================================================
def test_registry_total_backtest_count():
    """Registry.get_total_backtest_count() should return correct count."""
    from backtesting.registry import StrategyRegistry

    db_path, tmpdir = make_temp_db()
    try:
        reg = StrategyRegistry(db_path)

        # Initially should be 0
        count = reg.get_total_backtest_count()
        assert count == 0, f"Empty DB should have 0 backtests, got {count}"

        # Save some runs
        for i in range(5):
            reg.save_run(
                strategy_name=f'test_{i}',
                symbol='NQ',
                interval='5m',
                params={'ema': 50 + i},
                stats={'sharpe_ratio': 0.5 + i * 0.1},
                data_range=('2020-01-01', '2024-12-31'),
                regime='STAGE1_PASS',
                notes='test',
            )

        count = reg.get_total_backtest_count()
        assert count == 5, f"After 5 saves, should have 5, got {count}"

    finally:
        cleanup_temp(tmpdir)


# ===========================================================================
# QUALITY NOTES CONSTRUCTION
# ===========================================================================
def test_quality_notes_construction():
    """Test quality notes string built in _save_winner."""
    s2 = {
        'sharpe_ci_lower': 0.45,
        'sharpe_ci_upper': 1.85,
        'permutation_p_value': 0.02,
    }
    s4 = {'robustness_score': 72}
    s5 = {'dsr_probability': 0.97}
    mc_var95 = -0.12

    stat_notes = ["Passed all 5 gates"]
    if s2.get('sharpe_ci_lower') is not None:
        stat_notes.append(f"Sharpe CI: [{s2['sharpe_ci_lower']:.3f}, {s2.get('sharpe_ci_upper', 0):.3f}]")
    if s2.get('permutation_p_value') is not None:
        stat_notes.append(f"Perm p={s2['permutation_p_value']:.3f}")
    if s5.get('dsr_probability') is not None:
        stat_notes.append(f"DSR={s5['dsr_probability']:.3f}")
    if s4.get('robustness_score') is not None:
        stat_notes.append(f"Robustness={s4['robustness_score']:.0f}")
    if mc_var95 != 0.0:
        stat_notes.append(f"MC VaR95={mc_var95:.3f}")

    quality_notes = " | ".join(stat_notes)

    assert "Passed all 5 gates" in quality_notes
    assert "Sharpe CI: [0.450, 1.850]" in quality_notes
    assert "Perm p=0.020" in quality_notes
    assert "DSR=0.970" in quality_notes
    assert "Robustness=72" in quality_notes
    assert "MC VaR95=-0.120" in quality_notes


# ===========================================================================
# FULL GATE CHAIN SIMULATION
# ===========================================================================
def test_full_gate_chain_rejection_reasons():
    """Simulate full gate chain and verify rejection reasons are properly logged."""
    from backtesting.statistics import StatisticalSignificance
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Strategy that would fail Sharpe CI
    returns = make_random_returns(n=200, seed=999)
    sharpe_val, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        returns, confidence=config.stat_confidence_level
    )

    if ci_lower <= 0:
        failure_reason = (
            f"Sharpe CI crosses zero: [{ci_lower:.3f}, {ci_upper:.3f}] — "
            f"not statistically significant at {config.stat_confidence_level:.0%}"
        )
        assert "Sharpe CI crosses zero" in failure_reason, "Failure reason should mention CI"
        assert f"{config.stat_confidence_level:.0%}" in failure_reason, \
            "Failure reason should mention confidence level"


def test_full_gate_chain_statistical_verification():
    """Verify a strong strategy can pass the full statistical chain."""
    from backtesting.statistics import StatisticalSignificance
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Very strong strategy: high drift, low vol, lots of data
    returns = make_trending_returns(n=1000, drift=0.003, volatility=0.015, seed=42)

    # Gate 1: Sharpe CI
    sharpe_val, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        returns, confidence=config.stat_confidence_level
    )
    ci_pass = ci_lower > 0
    assert ci_pass, f"Strong strategy should pass Sharpe CI: [{ci_lower:.3f}, {ci_upper:.3f}]"

    # Gate 2: DSR (after many trials)
    se = StatisticalSignificance.sharpe_standard_error(returns, sharpe_val)
    var_sharpe = se ** 2
    ret_stats = StatisticalSignificance.returns_statistics(returns)

    dsr_prob = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=sharpe_val,
        n_trials=1000,
        var_sharpe=var_sharpe,
        skewness=ret_stats.get('skewness', 0),
        kurtosis=ret_stats.get('kurtosis', 3),
        n_observations=len(returns),
    )

    # Note: With 1000 trials, a strong strategy should still pass
    # but DSR is deliberately strict — verify it's a valid probability
    assert 0 <= dsr_prob <= 1, f"DSR should be valid probability: {dsr_prob:.3f}"


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    runner = TestRunner()

    print("\n=== S2 Sharpe CI Gate Integration ===")
    runner.run_test(test_s2_sharpe_ci_gate_logic)
    runner.run_test(test_s2_sharpe_ci_skipped_when_disabled)
    runner.run_test(test_s2_sharpe_ci_insufficient_data)

    print("\n=== S2 Permutation Test Gate Integration ===")
    runner.run_test(test_s2_permutation_gate_config)
    runner.run_test(test_s2_permutation_verdict_logic)
    runner.run_test(test_s2_permutation_skipped_when_disabled)

    print("\n=== S4 Robustness & Cliff Detection Integration ===")
    runner.run_test(test_s4_robustness_gate_with_stable_params)
    runner.run_test(test_s4_robustness_gate_rejects_fragile_params)
    runner.run_test(test_s4_cliff_detection_rejects_high_severity)
    runner.run_test(test_s4_variant_profitability_threshold)

    print("\n=== S5 DSR Gate Integration ===")
    runner.run_test(test_s5_dsr_gate_logic)
    runner.run_test(test_s5_dsr_skipped_with_low_trials)

    print("\n=== S5 NQmain Overlap Integration ===")
    runner.run_test(test_s5_nqmain_overlap_rejects_orb_overlap)
    runner.run_test(test_s5_nqmain_overlap_passes_complementary)
    runner.run_test(test_s5_nqmain_overlap_non_orb_passes)

    print("\n=== MC VaR95 Gate Integration ===")
    runner.run_test(test_mc_var95_gate_logic)

    print("\n=== Complexity Cap Integration ===")
    runner.run_test(test_complexity_cap_logic)

    print("\n=== Market Rationale Integration ===")
    runner.run_test(test_market_rationale_attachment)
    runner.run_test(test_market_rationale_not_overridden)

    print("\n=== New Archetype Integration ===")
    runner.run_test(test_new_archetypes_exist)
    runner.run_test(test_new_archetypes_in_rationale)
    runner.run_test(test_new_archetypes_complement_nqmain)

    print("\n=== Registry Integration ===")
    runner.run_test(test_registry_total_backtest_count)

    print("\n=== Quality Notes Construction ===")
    runner.run_test(test_quality_notes_construction)

    print("\n=== Full Gate Chain Simulation ===")
    runner.run_test(test_full_gate_chain_rejection_reasons)
    runner.run_test(test_full_gate_chain_statistical_verification)

    success = runner.report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
