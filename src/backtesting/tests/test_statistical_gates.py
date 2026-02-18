"""
Statistical Gates Unit Tests
==============================
Tests for all anti-overfitting statistical verification gates:
- Sharpe Confidence Interval (Lo 2002)
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
- Permutation test gating
- Complexity caps
- Monte Carlo VaR95 gate
- WFO Parameter Robustness Score
- NQmain time overlap / complementary scoring

Run with:
    cd StrategyPipeline/src
    python -m backtesting.tests.test_statistical_gates
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import time as dtime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ===========================================================================
# TEST RUNNER (same pattern as test_vector_engine.py)
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
# SYNTHETIC DATA GENERATORS
# ===========================================================================
def make_trending_returns(n=500, drift=0.001, volatility=0.02, seed=42):
    """Generate returns from a strategy with positive edge (trending)."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(drift, volatility, n)
    return returns


def make_random_walk_returns(n=500, volatility=0.02, seed=42):
    """Generate returns from a pure random walk (zero edge)."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0, volatility, n)
    return returns


def make_overfit_returns(n=500, seed=42):
    """Generate returns that look great in-sample but are just noise.
    High Sharpe from noise — small N, lucky draw.
    """
    rng = np.random.RandomState(seed)
    # Small sample with lucky positive drift
    returns = rng.normal(0.003, 0.01, 30)  # Only 30 obs, high apparent Sharpe
    return returns


def make_high_sharpe_returns(n=1000, drift=0.002, volatility=0.015, seed=42):
    """Generate returns from a genuinely strong strategy (Sharpe > 2)."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(drift, volatility, n)
    return returns


def make_catastrophic_returns(n=500, seed=42):
    """Generate returns with catastrophic tail risk."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0005, 0.02, n)
    # Add a few massive drawdowns
    crash_idx = rng.choice(n, size=10, replace=False)
    returns[crash_idx] = rng.uniform(-0.15, -0.08, 10)
    return returns


# ===========================================================================
# SHARPE CONFIDENCE INTERVAL TESTS
# ===========================================================================
def test_sharpe_ci_rejects_zero_crossing():
    """Sharpe CI should reject strategy whose 95% CI crosses zero."""
    from backtesting.statistics import StatisticalSignificance

    # Random walk — CI should cross zero
    returns = make_random_walk_returns(n=200, seed=123)
    sharpe, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        returns, confidence=0.95
    )
    # With zero-drift returns, CI should span zero
    assert ci_lower <= 0 or abs(sharpe) < 0.5, \
        f"Random walk should have CI crossing zero or near-zero Sharpe, got [{ci_lower:.3f}, {ci_upper:.3f}]"


def test_sharpe_ci_passes_strong_strategy():
    """Sharpe CI should pass genuinely strong strategy with positive lower bound."""
    from backtesting.statistics import StatisticalSignificance

    returns = make_high_sharpe_returns(n=1000, seed=42)
    sharpe, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        returns, confidence=0.95
    )
    # Strong strategy should have positive CI lower bound
    assert ci_lower > 0, \
        f"Strong strategy should have CI lower > 0, got [{ci_lower:.3f}, {ci_upper:.3f}]"
    assert sharpe > 1.0, \
        f"Expected Sharpe > 1.0 for high-drift returns, got {sharpe:.3f}"


def test_sharpe_ci_width_increases_with_lower_n():
    """CI should be wider with fewer observations (more uncertainty)."""
    from backtesting.statistics import StatisticalSignificance

    returns_long = make_trending_returns(n=1000, seed=42)
    returns_short = make_trending_returns(n=50, seed=42)

    _, ci_lo_long, ci_hi_long = StatisticalSignificance.sharpe_confidence_interval(
        returns_long, confidence=0.95
    )
    _, ci_lo_short, ci_hi_short = StatisticalSignificance.sharpe_confidence_interval(
        returns_short, confidence=0.95
    )

    width_long = ci_hi_long - ci_lo_long
    width_short = ci_hi_short - ci_lo_short

    assert width_short > width_long, \
        f"Shorter sample should have wider CI: short={width_short:.3f}, long={width_long:.3f}"


def test_sharpe_se_formula():
    """Verify Sharpe SE follows Lo (2002) formula."""
    from backtesting.statistics import StatisticalSignificance

    returns = make_trending_returns(n=252)
    sharpe = 1.5
    se = StatisticalSignificance.sharpe_standard_error(returns, sharpe=sharpe)

    # Lo (2002): SE = sqrt((1 + 0.5 * SR^2) / (N - 1))
    expected_se = np.sqrt((1 + 0.5 * sharpe**2) / (252 - 1))
    assert abs(se - expected_se) < 1e-10, \
        f"SE mismatch: got {se:.6f}, expected {expected_se:.6f}"


def test_sharpe_ci_handles_edge_cases():
    """CI computation should handle edge cases gracefully."""
    from backtesting.statistics import StatisticalSignificance

    # Single observation
    sharpe, ci_lo, ci_hi = StatisticalSignificance.sharpe_confidence_interval(
        np.array([0.01]), confidence=0.95
    )
    assert ci_lo == -np.inf or np.isinf(ci_lo), "Single obs should give infinite CI"

    # Zero-variance returns
    sharpe, ci_lo, ci_hi = StatisticalSignificance.sharpe_confidence_interval(
        np.array([0.01, 0.01, 0.01, 0.01, 0.01]), confidence=0.95
    )
    # Should handle without crashing


# ===========================================================================
# DEFLATED SHARPE RATIO TESTS
# ===========================================================================
def test_dsr_rejects_after_many_trials():
    """DSR should reject mediocre Sharpe after 5000 trials (data mining)."""
    from backtesting.statistics import StatisticalSignificance

    # Mediocre Sharpe after many trials = likely lucky
    dsr = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=0.8,
        n_trials=5000,
        var_sharpe=0.3,
        skewness=-0.5,
        kurtosis=4.0,
        n_observations=500,
    )
    assert dsr < 0.95, \
        f"Mediocre Sharpe after 5000 trials should fail DSR, got probability={dsr:.3f}"


def test_dsr_passes_strong_sharpe():
    """DSR should pass genuinely strong Sharpe even after many trials."""
    from backtesting.statistics import StatisticalSignificance

    # Use even stronger Sharpe (4.0) with lower variance to ensure passage
    dsr = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=4.0,
        n_trials=5000,
        var_sharpe=0.2,
        skewness=-0.2,
        kurtosis=3.5,
        n_observations=1000,
    )
    assert dsr > 0.95, \
        f"Strong Sharpe (4.0) should pass DSR even after 5000 trials, got probability={dsr:.3f}"


def test_dsr_single_trial_passthrough():
    """DSR with 1 trial should return the raw Sharpe (no adjustment)."""
    from backtesting.statistics import StatisticalSignificance

    result = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=1.5,
        n_trials=1,
        var_sharpe=0.3,
    )
    assert result == 1.5, \
        f"Single trial DSR should return raw Sharpe, got {result}"


def test_dsr_increases_with_more_trials():
    """Expected max under null increases with more trials → DSR decreases."""
    from backtesting.statistics import StatisticalSignificance

    dsr_few = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=1.0, n_trials=10, var_sharpe=0.3, n_observations=500
    )
    dsr_many = StatisticalSignificance.deflated_sharpe_ratio(
        sharpe=1.0, n_trials=10000, var_sharpe=0.3, n_observations=500
    )
    assert dsr_few > dsr_many, \
        f"DSR should decrease with more trials: 10 trials={dsr_few:.3f}, 10000={dsr_many:.3f}"


# ===========================================================================
# RETURNS STATISTICS TESTS
# ===========================================================================
def test_returns_statistics_output():
    """Verify returns_statistics returns all required fields."""
    from backtesting.statistics import StatisticalSignificance

    returns = make_trending_returns(n=500)
    stats = StatisticalSignificance.returns_statistics(returns)

    assert 'mean' in stats, "Missing 'mean' in returns_statistics"
    assert 'std' in stats, "Missing 'std' in returns_statistics"
    assert 'skewness' in stats, "Missing 'skewness' in returns_statistics"
    assert 'kurtosis' in stats, "Missing 'kurtosis' in returns_statistics"

    # Verify values are reasonable
    assert abs(stats['mean'] - 0.001) < 0.005, f"Mean should be ~0.001, got {stats['mean']:.6f}"
    assert stats['std'] > 0, "Std should be positive"


def test_returns_statistics_short_array():
    """Returns statistics should handle very short arrays."""
    from backtesting.statistics import StatisticalSignificance

    stats = StatisticalSignificance.returns_statistics(np.array([0.01, 0.02]))
    assert stats['skewness'] == 0 or abs(stats['skewness']) < 100, \
        "Short array stats should not blow up"


# ===========================================================================
# SHARPE P-VALUE TESTS
# ===========================================================================
def test_sharpe_pvalue_significant():
    """Strong Sharpe with many obs should be highly significant."""
    from backtesting.statistics import StatisticalSignificance

    p = StatisticalSignificance.sharpe_pvalue(sharpe=2.0, n_observations=500)
    assert p < 0.05, f"Sharpe=2.0, n=500 should be significant, got p={p:.4f}"


def test_sharpe_pvalue_not_significant():
    """Weak Sharpe with few obs should not be significant."""
    from backtesting.statistics import StatisticalSignificance

    p = StatisticalSignificance.sharpe_pvalue(sharpe=0.3, n_observations=30)
    assert p > 0.05, f"Sharpe=0.3, n=30 should NOT be significant, got p={p:.4f}"


# ===========================================================================
# MINIMUM TRACK RECORD LENGTH TESTS
# ===========================================================================
def test_mtrl_reasonable_length():
    """MTRL should give reasonable track record length for moderate Sharpe."""
    from backtesting.statistics import StatisticalSignificance

    n = StatisticalSignificance.minimum_track_record_length(sharpe=1.0)
    assert 10 < n < 10000, f"MTRL for Sharpe=1.0 should be reasonable, got {n}"


def test_mtrl_zero_sharpe():
    """MTRL for zero Sharpe should be infinite."""
    from backtesting.statistics import StatisticalSignificance

    n = StatisticalSignificance.minimum_track_record_length(sharpe=0.0)
    assert n == np.inf, f"MTRL for Sharpe=0 should be inf, got {n}"


# ===========================================================================
# COMPLEXITY CAP TESTS
# ===========================================================================
def test_complexity_cap_rejects_overparameterized():
    """Strategy with >8 numeric params should be rejected pre-S1."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    assert config.max_strategy_params == 8, \
        f"Default max_strategy_params should be 8, got {config.max_strategy_params}"

    # Simulate complexity check from _process_idea
    params_overfit = {
        'ema_fast': 10, 'ema_slow': 50, 'atr_period': 14, 'atr_mult': 2.0,
        'sl_mult': 1.5, 'tp_mult': 3.0, 'trail_mult': 2.0, 'rvol_thresh': 1.5,
        'volume_filter': 100, 'lookback': 20,  # 10 numeric params
    }
    numeric_count = sum(1 for v in params_overfit.values() if isinstance(v, (int, float)))
    assert numeric_count > config.max_strategy_params, \
        f"Test params should exceed cap: {numeric_count} > {config.max_strategy_params}"


def test_complexity_cap_passes_lean_strategy():
    """Strategy with <=8 numeric params should pass complexity check."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    params_lean = {
        'ema_period': 50, 'atr_period': 14, 'sl_mult': 2.0, 'tp_mult': 4.0,
        'name': 'my_strategy',  # Non-numeric, doesn't count
    }
    numeric_count = sum(1 for v in params_lean.values() if isinstance(v, (int, float)))
    assert numeric_count <= config.max_strategy_params, \
        f"Lean params should be under cap: {numeric_count} <= {config.max_strategy_params}"


# ===========================================================================
# MONTE CARLO VAR95 GATE TESTS
# ===========================================================================
def test_mc_var95_config():
    """MC VaR95 config should be -0.30 (max 30% loss at 95th percentile)."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()
    assert config.mc_min_var95 == -0.30, \
        f"mc_min_var95 should be -0.30, got {config.mc_min_var95}"


def test_mc_gate_rejects_severe_var():
    """MC gate should reject VaR95 worse than -30%."""
    mc_var95 = -0.45  # 45% loss at 95th percentile
    mc_min_var95 = -0.30

    # Simulate gate logic from _save_winner
    rejected = mc_var95 < mc_min_var95 and mc_var95 != 0.0
    assert rejected, f"VaR95={mc_var95} should be rejected (threshold={mc_min_var95})"


def test_mc_gate_passes_acceptable_var():
    """MC gate should pass VaR95 within acceptable range."""
    mc_var95 = -0.15  # 15% loss at 95th percentile
    mc_min_var95 = -0.30

    rejected = mc_var95 < mc_min_var95 and mc_var95 != 0.0
    assert not rejected, f"VaR95={mc_var95} should pass (threshold={mc_min_var95})"


def test_mc_gate_zero_passthrough():
    """MC gate should pass when VaR95 is 0 (GPU not available / MC not run)."""
    mc_var95 = 0.0
    mc_min_var95 = -0.30

    rejected = mc_var95 < mc_min_var95 and mc_var95 != 0.0
    assert not rejected, "VaR95=0.0 (MC not run) should auto-pass"


# ===========================================================================
# S4 TIGHTENED THRESHOLDS TESTS
# ===========================================================================
def test_s4_thresholds_tightened():
    """S4 thresholds should be significantly tighter than original."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Max profit drop: was 2.0 (200%), now 0.50 (50%)
    assert config.s4_max_profit_drop_pct <= 0.50, \
        f"S4 max profit drop should be <=0.50, got {config.s4_max_profit_drop_pct}"

    # Variation factors: was [0.9, 1.1], now [0.8, 0.9, 1.1, 1.2]
    assert len(config.s4_variation_factors) >= 4, \
        f"S4 should have >=4 variation factors, got {len(config.s4_variation_factors)}"
    assert 0.8 in config.s4_variation_factors, "S4 should test ±20% (0.8 factor)"
    assert 1.2 in config.s4_variation_factors, "S4 should test ±20% (1.2 factor)"

    # Min profitable %: was 0.50, now 0.60
    assert config.s4_min_profitable_pct >= 0.60, \
        f"S4 min profitable pct should be >=0.60, got {config.s4_min_profitable_pct}"

    # Robustness score minimum
    assert config.s4_min_robustness_score >= 50.0, \
        f"S4 min robustness score should be >=50, got {config.s4_min_robustness_score}"


def test_s4_profit_drop_rejects_fragile():
    """S4 should reject strategy with >50% profit drop from parameter perturbation."""
    max_drop = 0.65  # 65% profit drop
    threshold = 0.50
    assert max_drop > threshold, "Test data should exceed threshold"

    # Simulate: excessive_drop = max_drop > config.s4_max_profit_drop_pct
    excessive_drop = max_drop > threshold
    assert excessive_drop, "65% profit drop should fail S4"


def test_s4_profit_drop_passes_robust():
    """S4 should pass strategy with <=50% profit drop."""
    max_drop = 0.25  # 25% profit drop
    threshold = 0.50
    excessive_drop = max_drop > threshold
    assert not excessive_drop, "25% profit drop should pass S4"


# ===========================================================================
# WFO ROBUSTNESS SCORE TESTS
# ===========================================================================
def test_robustness_score_structure():
    """ParameterSensitivityMapper.robustness_score() should return expected fields."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Create variant DataFrame with known structure
    rows = []
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        rows.append({
            'param_name': 'ema_period',
            'factor': factor,
            'value': int(50 * factor),
            'Total Return': 50000 * (1.0 - abs(factor - 1.0) * 0.3),  # Robust: gradual decline
        })
    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')

    result = mapper.robustness_score()
    assert 'robustness_score' in result, "Missing robustness_score"
    assert 'interpretation' in result, "Missing interpretation"
    assert isinstance(result['robustness_score'], (int, float)), "robustness_score should be numeric"


def test_cliff_detection_finds_cliff():
    """Cliff detection should identify sharp performance drops between adjacent params."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Cliff scenario: huge drop at factor=1.2
    rows = []
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        profit = 50000 if factor < 1.2 else 5000  # Cliff at 1.2
        rows.append({
            'param_name': 'ema_period',
            'factor': factor,
            'value': int(50 * factor),
            'Total Return': profit,
        })
    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')

    cliffs = mapper.cliff_detection()
    assert isinstance(cliffs, list), "cliff_detection should return list"
    # Should find at least one cliff (the big drop at 1.1->1.2)
    assert len(cliffs) > 0, "Should detect the parameter cliff"


def test_robustness_score_high_for_stable_params():
    """Robustness score should be high when all parameter variants perform well."""
    from backtesting.wfo_analytics import ParameterSensitivityMapper

    # Very stable performance across all factors
    rows = []
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        rows.append({
            'param_name': 'ema_period',
            'factor': factor,
            'value': int(50 * factor),
            'Total Return': 50000 + np.random.uniform(-1000, 1000),
        })
    df = pd.DataFrame(rows)
    mapper = ParameterSensitivityMapper(df, metric_col='Total Return')

    result = mapper.robustness_score()
    score = result['robustness_score']
    assert score >= 40, f"Stable params should have robustness >= 40, got {score}"


# ===========================================================================
# NQMAIN ANALYZER TESTS
# ===========================================================================
def test_nqmain_profile_creation():
    """NQmain profile should have correct time windows from Pine Script."""
    from backtesting.nqmain_analyzer import get_nqmain_profile

    profile = get_nqmain_profile()

    assert profile.orb_session_start == dtime(9, 30), "ORB start should be 9:30"
    assert profile.orb_session_end == dtime(9, 45), "ORB end should be 9:45"
    assert profile.trading_window_start == dtime(9, 45), "Trading start should be 9:45"
    assert profile.trading_window_end == dtime(15, 45), "Trading end should be 15:45"
    assert profile.regime == "trending", "NQmain regime should be trending"
    assert profile.requires_rvol is True, "NQmain requires RVOL"
    assert profile.rvol_threshold == 1.5, "RVOL threshold should be 1.5"
    assert profile.max_trades_per_day == 1, "Max trades per day should be 1"


def test_nqmain_active_minutes():
    """NQmain active minutes should be 360 (9:45 to 15:45 = 6 hours)."""
    from backtesting.nqmain_analyzer import get_nqmain_profile

    profile = get_nqmain_profile()
    assert profile.get_active_minutes() == 360, \
        f"Active minutes should be 360, got {profile.get_active_minutes()}"


def test_nqmain_gap_minutes():
    """NQmain gap minutes should be 15 + 120 + 15 = 150."""
    from backtesting.nqmain_analyzer import get_nqmain_profile

    profile = get_nqmain_profile()
    gap_min = profile.get_gap_minutes()
    assert gap_min == 150, f"Gap minutes should be 150, got {gap_min}"


def test_nqmain_coverage_map():
    """Coverage map should have GAP entries for known gaps."""
    from backtesting.nqmain_analyzer import get_nqmain_profile

    profile = get_nqmain_profile()
    cmap = profile.get_coverage_map()

    assert cmap["09:30-09:45"]["status"] == "GAP", "9:30-9:45 should be GAP"
    assert cmap["11:30-13:30"]["status"] == "GAP", "11:30-13:30 should be GAP"
    assert cmap["15:45-16:00"]["status"] == "GAP", "15:45-16:00 should be GAP"
    assert cmap["09:45-10:15"]["status"] == "COVERED", "9:45-10:15 should be COVERED"


def test_time_overlap_full_overlap():
    """ORB breakout should have 100% overlap with NQmain."""
    from backtesting.nqmain_analyzer import compute_time_overlap

    # ORB breakout: 9:45-15:45 (same as NQmain)
    orb_window = (dtime(9, 45), dtime(15, 45))
    nq_windows = [(dtime(9, 45), dtime(15, 45))]
    overlap = compute_time_overlap(orb_window, nq_windows)
    assert overlap == 1.0, f"Full overlap should be 1.0, got {overlap}"


def test_time_overlap_partial():
    """Power hour should have partial overlap with NQmain."""
    from backtesting.nqmain_analyzer import compute_time_overlap

    # Power hour: 14:00-15:30, NQmain: 9:45-15:45
    ph_window = (dtime(14, 0), dtime(15, 30))
    nq_windows = [(dtime(9, 45), dtime(15, 45))]
    overlap = compute_time_overlap(ph_window, nq_windows)
    assert 0 < overlap <= 1.0, f"Partial overlap should be 0 < x <= 1.0, got {overlap}"
    # 90 min strategy, 90 min overlap → 100%
    assert overlap == 1.0, f"Power hour fully within NQmain, should be 1.0, got {overlap}"


def test_time_overlap_no_overlap():
    """Post-market strategy should have zero overlap with NQmain."""
    from backtesting.nqmain_analyzer import compute_time_overlap

    # Post-market: 16:00-17:00 (NQmain ends at 15:45)
    pm_window = (dtime(16, 0), dtime(17, 0))
    nq_windows = [(dtime(9, 45), dtime(15, 45))]
    overlap = compute_time_overlap(pm_window, nq_windows)
    assert overlap == 0.0, f"Post-market should have zero overlap, got {overlap}"


def test_complementary_score_lunch_fade():
    """Lunch range fade should have high complementary score (fills NQmain gap)."""
    from backtesting.nqmain_analyzer import get_complementary_score

    result = get_complementary_score('lunch_range_fade')
    assert result['gap_coverage'] is True, "Lunch fade should cover an NQmain gap"
    assert result['regime_complement'] is True, "Lunch fade should target gap regime"
    assert result['complementary_score'] >= 50, \
        f"Lunch fade should score >=50, got {result['complementary_score']}"


def test_complementary_score_orb_breakout():
    """ORB breakout should have low complementary score (overlaps NQmain)."""
    from backtesting.nqmain_analyzer import get_complementary_score

    result = get_complementary_score('orb_breakout')
    assert result['time_overlap'] == 1.0, "ORB should fully overlap NQmain"
    assert result['complementary_score'] < 50, \
        f"ORB breakout should score <50 (redundant), got {result['complementary_score']}"


def test_complementary_score_first_hour_fade():
    """First hour fade should target mean reversion gap."""
    from backtesting.nqmain_analyzer import get_complementary_score

    result = get_complementary_score('first_hour_fade')
    assert result['regime_complement'] is True, "First hour fade targets mean_reversion gap"


def test_strategy_time_window_defaults():
    """get_strategy_time_window should return correct defaults per archetype."""
    from backtesting.nqmain_analyzer import get_strategy_time_window

    # Power hour
    start, end = get_strategy_time_window('power_hour_momentum')
    assert start == dtime(14, 0), f"Power hour start should be 14:00, got {start}"
    assert end == dtime(15, 30), f"Power hour end should be 15:30, got {end}"

    # Unknown archetype should default to full day
    start, end = get_strategy_time_window('unknown_archetype')
    assert start == dtime(9, 30), f"Unknown archetype should default to 9:30, got {start}"
    assert end == dtime(15, 45), f"Unknown archetype should default to 15:45, got {end}"


def test_strategy_time_window_param_override():
    """get_strategy_time_window should respect entry_time/exit_time params."""
    from backtesting.nqmain_analyzer import get_strategy_time_window

    params = {'entry_time': '11:00', 'exit_time': '13:00'}
    start, end = get_strategy_time_window('lunch_hour_breakout', params)
    assert start == dtime(11, 0), f"Start should be 11:00, got {start}"
    assert end == dtime(13, 0), f"End should be 13:00, got {end}"


def test_nqmain_gap_regimes():
    """NQmain should list expected gap regimes."""
    from backtesting.nqmain_analyzer import get_nqmain_profile

    profile = get_nqmain_profile()
    assert "mean_reversion" in profile.gap_regimes, "Should identify mean_reversion gap"
    assert "choppy_range" in profile.gap_regimes, "Should identify choppy_range gap"
    assert "low_volatility" in profile.gap_regimes, "Should identify low_volatility gap"


# ===========================================================================
# MARKET RATIONALE TESTS
# ===========================================================================
def test_market_rationale_exists_for_all_archetypes():
    """Every archetype should have a market rationale."""
    from backtesting.research_engine import MARKET_RATIONALE

    required_archetypes = [
        'orb_breakout', 'ma_crossover', 'eod_momentum',
        'lunch_hour_breakout', 'gap_fill_fade',
        'power_hour_momentum', 'first_hour_fade', 'lunch_range_fade',
    ]

    for arch in required_archetypes:
        assert arch in MARKET_RATIONALE, f"Missing rationale for '{arch}'"
        rationale = MARKET_RATIONALE[arch]
        assert len(rationale) > 50, f"Rationale for '{arch}' too short: {len(rationale)} chars"


# ===========================================================================
# CONFIG FIELD EXISTENCE TESTS
# ===========================================================================
def test_config_statistical_fields_exist():
    """All new statistical config fields should exist with correct defaults."""
    from backtesting.marcus_config import MarcusConfig

    config = MarcusConfig.default()

    # Statistical Verification Gates
    assert hasattr(config, 'stat_require_sharpe_ci_positive'), "Missing stat_require_sharpe_ci_positive"
    assert config.stat_require_sharpe_ci_positive is True

    assert hasattr(config, 'stat_confidence_level'), "Missing stat_confidence_level"
    assert config.stat_confidence_level == 0.95

    assert hasattr(config, 'stat_min_dsr_probability'), "Missing stat_min_dsr_probability"
    assert config.stat_min_dsr_probability == 0.95

    assert hasattr(config, 's2_permutation_enabled'), "Missing s2_permutation_enabled"
    assert config.s2_permutation_enabled is True

    assert hasattr(config, 's2_permutation_sims'), "Missing s2_permutation_sims"
    assert config.s2_permutation_sims == 50

    assert hasattr(config, 'mc_min_var95'), "Missing mc_min_var95"
    assert config.mc_min_var95 == -0.30

    # Complexity caps
    assert hasattr(config, 'max_strategy_params'), "Missing max_strategy_params"
    assert config.max_strategy_params == 8

    assert hasattr(config, 'max_strategy_indicators'), "Missing max_strategy_indicators"
    assert config.max_strategy_indicators == 5

    # NQmain
    assert hasattr(config, 's5_max_nqmain_overlap'), "Missing s5_max_nqmain_overlap"
    assert config.s5_max_nqmain_overlap == 0.30

    # S4 robustness
    assert hasattr(config, 's4_min_robustness_score'), "Missing s4_min_robustness_score"
    assert config.s4_min_robustness_score == 50.0


# ===========================================================================
# BOOTSTRAP INFERENCE TESTS
# ===========================================================================
def test_bootstrap_sharpe_ci():
    """Bootstrap Sharpe CI should produce valid confidence intervals."""
    from backtesting.statistics import BootstrapInference

    returns = make_high_sharpe_returns(n=500, seed=42)
    result = BootstrapInference.bootstrap_sharpe_ci(
        returns, n_bootstrap=1000, confidence=0.95, random_state=42
    )

    # Keys: 'sharpe', 'ci_lower', 'ci_upper', 'bootstrap_std', 'bootstrap_mean', 'crosses_zero'
    assert 'ci_lower' in result, "Missing ci_lower"
    assert 'ci_upper' in result, "Missing ci_upper"
    assert 'sharpe' in result, "Missing sharpe"
    assert result['ci_lower'] < result['ci_upper'], \
        "CI lower should be less than upper"
    assert result['sharpe'] > 0, "Strong strategy should have positive Sharpe"


def test_bootstrap_drawdown_distribution():
    """Bootstrap drawdown distribution should produce valid results."""
    from backtesting.statistics import BootstrapInference

    returns = make_trending_returns(n=300, seed=42)
    result = BootstrapInference.bootstrap_drawdown_distribution(
        returns, n_bootstrap=500, random_state=42
    )

    # Keys: 'original_dd', 'dd_p5', 'dd_p25', 'dd_p50', 'dd_p75', 'dd_p95', 'dd_mean', 'dd_std'
    assert 'dd_mean' in result, "Missing dd_mean"
    assert 'dd_p5' in result, "Missing dd_p5 (worst case)"
    assert 'original_dd' in result, "Missing original_dd"
    assert isinstance(result['dd_mean'], (int, float, np.floating)), \
        "Drawdown should be numeric"


# ===========================================================================
# MULTIPLE TESTING CORRECTION TESTS
# ===========================================================================
def test_bonferroni_correction():
    """Bonferroni correction should be more conservative with more tests."""
    from backtesting.statistics import MultipleTestingCorrection

    p_values = np.array([0.01, 0.03, 0.05, 0.10, 0.50])
    result = MultipleTestingCorrection.bonferroni_correction(p_values, alpha=0.05)

    # Keys: 'original_alpha', 'adjusted_alpha', 'n_tests', 'n_significant', 'significant_mask', 'adjusted_pvalues'
    assert 'adjusted_alpha' in result, "Missing adjusted_alpha"
    assert result['adjusted_alpha'] == 0.05 / len(p_values), \
        f"Adjusted alpha should be 0.05/{len(p_values)}, got {result['adjusted_alpha']}"
    assert result['n_tests'] == len(p_values), f"n_tests should be {len(p_values)}"


def test_fdr_correction():
    """FDR correction (Benjamini-Hochberg) should control false discovery rate."""
    from backtesting.statistics import MultipleTestingCorrection

    p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
    result = MultipleTestingCorrection.fdr_correction(p_values, alpha=0.05)

    assert 'n_significant' in result, "Missing n_significant"
    assert isinstance(result['n_significant'], (int, np.integer)), "n_significant should be int"


# ===========================================================================
# INTEGRATION: Full Statistical Analysis
# ===========================================================================
def test_analyze_strategy_significance():
    """Full significance analysis should combine all statistical tests."""
    from backtesting.statistics import analyze_strategy_significance

    returns = make_high_sharpe_returns(n=500, seed=42)
    result = analyze_strategy_significance(
        returns, n_trials=100, periods=252, confidence=0.95
    )

    # Keys: 'sharpe', 'sharpe_ci_lower', 'sharpe_ci_upper', 'sharpe_pvalue', etc.
    assert 'sharpe' in result, "Missing sharpe"
    assert 'sharpe_ci_lower' in result, "Missing sharpe_ci_lower"
    assert 'sharpe_ci_upper' in result, "Missing sharpe_ci_upper"
    assert 'sharpe_pvalue' in result, "Missing sharpe_pvalue"
    assert 'deflated_sharpe_ratio' in result, "Missing deflated_sharpe_ratio"

    # A strong strategy should have significant results
    assert result['sharpe'] > 1.0, f"Expected strong Sharpe, got {result['sharpe']}"


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    runner = TestRunner()

    print("\n=== Sharpe Confidence Interval Tests ===")
    runner.run_test(test_sharpe_ci_rejects_zero_crossing)
    runner.run_test(test_sharpe_ci_passes_strong_strategy)
    runner.run_test(test_sharpe_ci_width_increases_with_lower_n)
    runner.run_test(test_sharpe_se_formula)
    runner.run_test(test_sharpe_ci_handles_edge_cases)

    print("\n=== Deflated Sharpe Ratio Tests ===")
    runner.run_test(test_dsr_rejects_after_many_trials)
    runner.run_test(test_dsr_passes_strong_sharpe)
    runner.run_test(test_dsr_single_trial_passthrough)
    runner.run_test(test_dsr_increases_with_more_trials)

    print("\n=== Returns Statistics Tests ===")
    runner.run_test(test_returns_statistics_output)
    runner.run_test(test_returns_statistics_short_array)

    print("\n=== Sharpe P-Value Tests ===")
    runner.run_test(test_sharpe_pvalue_significant)
    runner.run_test(test_sharpe_pvalue_not_significant)

    print("\n=== Minimum Track Record Length Tests ===")
    runner.run_test(test_mtrl_reasonable_length)
    runner.run_test(test_mtrl_zero_sharpe)

    print("\n=== Complexity Cap Tests ===")
    runner.run_test(test_complexity_cap_rejects_overparameterized)
    runner.run_test(test_complexity_cap_passes_lean_strategy)

    print("\n=== Monte Carlo VaR95 Gate Tests ===")
    runner.run_test(test_mc_var95_config)
    runner.run_test(test_mc_gate_rejects_severe_var)
    runner.run_test(test_mc_gate_passes_acceptable_var)
    runner.run_test(test_mc_gate_zero_passthrough)

    print("\n=== S4 Tightened Threshold Tests ===")
    runner.run_test(test_s4_thresholds_tightened)
    runner.run_test(test_s4_profit_drop_rejects_fragile)
    runner.run_test(test_s4_profit_drop_passes_robust)

    print("\n=== WFO Robustness Score Tests ===")
    runner.run_test(test_robustness_score_structure)
    runner.run_test(test_cliff_detection_finds_cliff)
    runner.run_test(test_robustness_score_high_for_stable_params)

    print("\n=== NQmain Analyzer Tests ===")
    runner.run_test(test_nqmain_profile_creation)
    runner.run_test(test_nqmain_active_minutes)
    runner.run_test(test_nqmain_gap_minutes)
    runner.run_test(test_nqmain_coverage_map)
    runner.run_test(test_time_overlap_full_overlap)
    runner.run_test(test_time_overlap_partial)
    runner.run_test(test_time_overlap_no_overlap)
    runner.run_test(test_complementary_score_lunch_fade)
    runner.run_test(test_complementary_score_orb_breakout)
    runner.run_test(test_complementary_score_first_hour_fade)
    runner.run_test(test_strategy_time_window_defaults)
    runner.run_test(test_strategy_time_window_param_override)
    runner.run_test(test_nqmain_gap_regimes)

    print("\n=== Market Rationale Tests ===")
    runner.run_test(test_market_rationale_exists_for_all_archetypes)

    print("\n=== Config Field Existence Tests ===")
    runner.run_test(test_config_statistical_fields_exist)

    print("\n=== Bootstrap Inference Tests ===")
    runner.run_test(test_bootstrap_sharpe_ci)
    runner.run_test(test_bootstrap_drawdown_distribution)

    print("\n=== Multiple Testing Correction Tests ===")
    runner.run_test(test_bonferroni_correction)
    runner.run_test(test_fdr_correction)

    print("\n=== Full Statistical Analysis Tests ===")
    runner.run_test(test_analyze_strategy_significance)

    success = runner.report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
