"""
Statistical Significance Module for Strategy Validation.

Provides rigorous statistical testing to determine if strategy performance
is genuine or due to chance. Implements:
- Sharpe ratio standard error and confidence intervals
- Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
- Bootstrap inference for non-parametric confidence intervals
- Multiple testing corrections (Bonferroni, FDR)

References:
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy import stats as scipy_stats


class StatisticalSignificance:
    """
    Core statistical significance calculations for trading strategies.
    """

    @staticmethod
    def sharpe_standard_error(returns: np.ndarray, sharpe: float = None, periods: int = 252) -> float:
        """
        Calculate the standard error of the Sharpe ratio.

        Formula: SE = sqrt((1 + 0.5 * SR^2) / (N - 1))

        This accounts for the sampling uncertainty in estimating both
        mean and standard deviation from finite samples.

        Args:
            returns: Array of period returns
            sharpe: Pre-computed Sharpe ratio (optional, will compute if None)
            periods: Annualization factor (252 for daily, 52 for weekly)

        Returns:
            Standard error of the annualized Sharpe ratio
        """
        n = len(returns)
        if n < 2:
            return np.inf

        if sharpe is None:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            if std_ret == 0:
                return np.inf
            sharpe = np.sqrt(periods) * mean_ret / std_ret

        # Lo (2002) formula for Sharpe SE
        se = np.sqrt((1 + 0.5 * sharpe**2) / (n - 1))

        return se

    @staticmethod
    def sharpe_confidence_interval(
        returns: np.ndarray,
        confidence: float = 0.95,
        periods: int = 252
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for the Sharpe ratio.

        Args:
            returns: Array of period returns
            confidence: Confidence level (default 0.95 for 95% CI)
            periods: Annualization factor

        Returns:
            Tuple of (sharpe, lower_bound, upper_bound)
        """
        n = len(returns)
        if n < 2:
            return (0.0, -np.inf, np.inf)

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return (0.0, -np.inf, np.inf)

        sharpe = np.sqrt(periods) * mean_ret / std_ret
        se = StatisticalSignificance.sharpe_standard_error(returns, sharpe, periods)

        # Z-score for confidence level
        alpha = 1 - confidence
        z = scipy_stats.norm.ppf(1 - alpha / 2)

        lower = sharpe - z * se
        upper = sharpe + z * se

        return (sharpe, lower, upper)

    @staticmethod
    def sharpe_pvalue(sharpe: float, n_observations: int) -> float:
        """
        Calculate p-value for testing H0: True Sharpe = 0.

        Uses the asymptotic distribution of the Sharpe ratio estimator.

        Args:
            sharpe: Observed Sharpe ratio
            n_observations: Number of return observations

        Returns:
            Two-tailed p-value
        """
        if n_observations < 2:
            return 1.0

        se = np.sqrt((1 + 0.5 * sharpe**2) / (n_observations - 1))

        if se == 0 or np.isinf(se):
            return 1.0

        # Z-statistic
        z = sharpe / se

        # Two-tailed p-value
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

        return p_value

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe: float,
        n_trials: int,
        var_sharpe: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        n_observations: int = None
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio (DSR).

        Adjusts the Sharpe ratio for multiple testing / data snooping.
        Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"

        The DSR accounts for:
        1. Number of trials (strategies tested)
        2. Variance of Sharpe ratios across trials
        3. Non-normality of returns (skewness, kurtosis)

        Args:
            sharpe: Observed Sharpe ratio of the selected strategy
            n_trials: Number of strategies/parameter combinations tested
            var_sharpe: Variance of Sharpe ratios across all trials
            skewness: Skewness of returns (default 0)
            kurtosis: Kurtosis of returns (default 3 for normal)
            n_observations: Number of return observations

        Returns:
            Deflated Sharpe Ratio (probability that observed SR is skill, not luck)
        """
        if n_trials <= 1:
            return sharpe

        # Expected maximum Sharpe under null (all strategies have true SR=0)
        # E[max(SR)] ~ sqrt(2 * log(n_trials)) * sqrt(var_sharpe)
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) * np.sqrt(var_sharpe)

        # Euler-Mascheroni correction
        gamma = 0.5772156649
        expected_max_sr *= (1 - gamma / np.log(n_trials) + 0.5 * gamma**2 / np.log(n_trials)**2)

        # Adjust for non-normality (higher kurtosis = more extreme values expected by chance)
        if n_observations is not None and n_observations > 0:
            # Correction factor for non-normal returns
            correction = 1 + (kurtosis - 3) / (4 * n_observations) - skewness**2 / (6 * n_observations)
            expected_max_sr *= np.sqrt(correction)

        # DSR = probability that observed SR exceeds expected max under null
        if var_sharpe > 0:
            z = (sharpe - expected_max_sr) / np.sqrt(var_sharpe)
            dsr = scipy_stats.norm.cdf(z)
        else:
            dsr = 1.0 if sharpe > 0 else 0.0

        return dsr

    @staticmethod
    def minimum_track_record_length(
        sharpe: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        target_pvalue: float = 0.05,
        periods: int = 252
    ) -> int:
        """
        Calculate minimum track record length needed to confirm Sharpe is real.

        How many observations do we need before we can be confident
        (at the target p-value level) that the Sharpe ratio is positive?

        Args:
            sharpe: Target/observed annualized Sharpe ratio
            skewness: Return skewness
            kurtosis: Return kurtosis
            target_pvalue: Desired significance level
            periods: Annualization factor

        Returns:
            Minimum number of observations needed
        """
        if sharpe <= 0:
            return np.inf

        # De-annualize Sharpe to get per-period Sharpe
        sr_period = sharpe / np.sqrt(periods)

        # Z-score for target p-value (one-tailed, since we want SR > 0)
        z_target = scipy_stats.norm.ppf(1 - target_pvalue)

        # Solve for n: z_target = SR / SE(SR)
        # SE(SR) = sqrt((1 + 0.5*SR^2) / (n-1))
        # => n = (z_target / SR)^2 * (1 + 0.5*SR^2) + 1

        # Adjust for non-normality
        adjustment = 1 + (kurtosis - 3) / 4 - skewness * sr_period / 2

        n = (z_target / sr_period)**2 * (1 + 0.5 * sr_period**2) * adjustment + 1

        return int(np.ceil(n))

    @staticmethod
    def returns_statistics(returns: np.ndarray) -> Dict:
        """
        Calculate comprehensive return statistics including higher moments.

        Args:
            returns: Array of returns

        Returns:
            Dictionary with mean, std, skewness, kurtosis, and normality test
        """
        if len(returns) < 4:
            return {
                'mean': np.mean(returns) if len(returns) > 0 else 0,
                'std': np.std(returns) if len(returns) > 1 else 0,
                'skewness': 0,
                'kurtosis': 3,
                'is_normal': None,
                'normality_pvalue': None
            }

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        skewness = scipy_stats.skew(returns)
        kurtosis = scipy_stats.kurtosis(returns, fisher=False)  # Excess kurtosis

        # Jarque-Bera normality test
        if len(returns) >= 8:
            jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns)
            is_normal = jb_pvalue > 0.05
        else:
            jb_pvalue = None
            is_normal = None

        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': is_normal,
            'normality_pvalue': jb_pvalue
        }


class BootstrapInference:
    """
    Non-parametric bootstrap inference for strategy metrics.
    """

    @staticmethod
    def bootstrap_sharpe_ci(
        returns: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        periods: int = 252,
        random_state: int = None
    ) -> Dict:
        """
        Calculate bootstrap confidence interval for Sharpe ratio.

        Uses percentile bootstrap method which doesn't assume normality.

        Args:
            returns: Array of period returns
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            periods: Annualization factor
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with sharpe, ci_lower, ci_upper, and bootstrap distribution
        """
        if random_state is not None:
            np.random.seed(random_state)

        n = len(returns)
        if n < 5:
            return {
                'sharpe': 0,
                'ci_lower': -np.inf,
                'ci_upper': np.inf,
                'bootstrap_std': np.inf
            }

        # Original Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        original_sharpe = np.sqrt(periods) * mean_ret / std_ret if std_ret > 0 else 0

        # Bootstrap
        bootstrap_sharpes = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(returns, size=n, replace=True)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)

            if sample_std > 0:
                sr = np.sqrt(periods) * sample_mean / sample_std
                bootstrap_sharpes.append(sr)

        bootstrap_sharpes = np.array(bootstrap_sharpes)

        # Percentile confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

        return {
            'sharpe': original_sharpe,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_std': np.std(bootstrap_sharpes),
            'bootstrap_mean': np.mean(bootstrap_sharpes),
            'crosses_zero': ci_lower < 0 < ci_upper
        }

    @staticmethod
    def bootstrap_drawdown_distribution(
        returns: np.ndarray,
        n_bootstrap: int = 5000,
        random_state: int = None
    ) -> Dict:
        """
        Estimate the distribution of maximum drawdown via bootstrap.

        This helps answer: "Was my observed drawdown unusually good/bad?"

        Args:
            returns: Array of period returns
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed

        Returns:
            Dictionary with percentiles and original drawdown context
        """
        if random_state is not None:
            np.random.seed(random_state)

        n = len(returns)
        if n < 5:
            return {
                'original_dd': 0,
                'dd_p5': 0,
                'dd_p25': 0,
                'dd_p50': 0,
                'dd_p75': 0,
                'dd_p95': 0
            }

        def max_drawdown(rets):
            """Calculate max drawdown from returns array."""
            equity = np.cumprod(1 + rets)
            peak = np.maximum.accumulate(equity)
            safe_peak = np.where(peak == 0, np.nan, peak)
            dd = np.where(np.isnan(safe_peak), 0, (equity - peak) / safe_peak)
            return np.min(dd) if len(dd) > 0 else 0.0

        # Original drawdown
        original_dd = max_drawdown(returns)

        # Bootstrap drawdowns
        bootstrap_dds = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_dds.append(max_drawdown(sample))

        bootstrap_dds = np.array(bootstrap_dds)

        # Percentiles (note: drawdowns are negative, so p5 is worst)
        percentiles = {
            'original_dd': original_dd,
            'dd_p5': np.percentile(bootstrap_dds, 5),   # Worst case
            'dd_p25': np.percentile(bootstrap_dds, 25),
            'dd_p50': np.percentile(bootstrap_dds, 50),  # Median
            'dd_p75': np.percentile(bootstrap_dds, 75),
            'dd_p95': np.percentile(bootstrap_dds, 95),  # Best case
            'dd_mean': np.mean(bootstrap_dds),
            'dd_std': np.std(bootstrap_dds)
        }

        # Was original DD lucky or unlucky?
        rank = np.mean(bootstrap_dds <= original_dd)  # % of samples with worse DD
        percentiles['original_dd_percentile'] = rank * 100

        return percentiles

    @staticmethod
    def bootstrap_metric(
        returns: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
        random_state: int = None
    ) -> Dict:
        """
        Generic bootstrap CI for any metric function.

        Args:
            returns: Array of returns
            metric_func: Function that takes returns and returns a scalar
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            random_state: Random seed

        Returns:
            Dictionary with metric value and confidence interval
        """
        if random_state is not None:
            np.random.seed(random_state)

        n = len(returns)
        if n < 5:
            original = metric_func(returns)
            return {
                'value': original,
                'ci_lower': -np.inf,
                'ci_upper': np.inf
            }

        original = metric_func(returns)

        bootstrap_values = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            try:
                bootstrap_values.append(metric_func(sample))
            except Exception:
                pass

        if not bootstrap_values:
            return {'value': original, 'ci_lower': -np.inf, 'ci_upper': np.inf}

        bootstrap_values = np.array(bootstrap_values)

        alpha = 1 - confidence
        return {
            'value': original,
            'ci_lower': np.percentile(bootstrap_values, alpha / 2 * 100),
            'ci_upper': np.percentile(bootstrap_values, (1 - alpha / 2) * 100),
            'bootstrap_std': np.std(bootstrap_values)
        }


class MultipleTestingCorrection:
    """
    Corrections for multiple hypothesis testing (data snooping).
    """

    @staticmethod
    def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Bonferroni correction for multiple testing.

        Most conservative correction - controls family-wise error rate.

        Args:
            p_values: Array of p-values from multiple tests
            alpha: Desired significance level

        Returns:
            Dictionary with adjusted threshold and which tests pass
        """
        n_tests = len(p_values)
        adjusted_alpha = alpha / n_tests

        significant = p_values < adjusted_alpha

        return {
            'original_alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'n_tests': n_tests,
            'n_significant': np.sum(significant),
            'significant_mask': significant,
            'adjusted_pvalues': np.minimum(p_values * n_tests, 1.0)
        }

    @staticmethod
    def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Benjamini-Hochberg False Discovery Rate correction.

        Less conservative than Bonferroni - controls expected proportion
        of false discoveries.

        Args:
            p_values: Array of p-values
            alpha: Desired FDR level

        Returns:
            Dictionary with adjusted p-values and significance
        """
        n_tests = len(p_values)

        # Sort p-values
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH procedure
        ranks = np.arange(1, n_tests + 1)
        thresholds = ranks / n_tests * alpha

        # Find largest k where p_k <= threshold_k
        below_threshold = sorted_p <= thresholds

        if np.any(below_threshold):
            max_k = np.max(np.where(below_threshold)[0])
            significant_sorted = np.zeros(n_tests, dtype=bool)
            significant_sorted[:max_k + 1] = True
        else:
            significant_sorted = np.zeros(n_tests, dtype=bool)

        # Unsort
        significant = np.zeros(n_tests, dtype=bool)
        significant[sorted_idx] = significant_sorted

        # Adjusted p-values (BH adjusted)
        adjusted_p = np.zeros(n_tests)
        adjusted_p[sorted_idx] = np.minimum.accumulate(
            (sorted_p * n_tests / ranks)[::-1]
        )[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)

        return {
            'original_alpha': alpha,
            'n_tests': n_tests,
            'n_significant': np.sum(significant),
            'significant_mask': significant,
            'adjusted_pvalues': adjusted_p
        }

    @staticmethod
    def family_wise_adjusted_sharpe(
        sharpes: np.ndarray,
        n_observations: int
    ) -> Dict:
        """
        Adjust Sharpe ratios for family-wise error from multiple strategies.

        Args:
            sharpes: Array of Sharpe ratios from multiple strategies
            n_observations: Number of return observations per strategy

        Returns:
            Dictionary with original sharpes, p-values, and adjusted significance
        """
        n_strategies = len(sharpes)

        # Calculate p-value for each Sharpe
        p_values = np.array([
            StatisticalSignificance.sharpe_pvalue(sr, n_observations)
            for sr in sharpes
        ])

        # Apply FDR correction
        fdr_results = MultipleTestingCorrection.fdr_correction(p_values)

        # Apply Bonferroni for comparison
        bonf_results = MultipleTestingCorrection.bonferroni_correction(p_values)

        return {
            'sharpes': sharpes,
            'raw_pvalues': p_values,
            'fdr_adjusted_pvalues': fdr_results['adjusted_pvalues'],
            'fdr_significant': fdr_results['significant_mask'],
            'bonferroni_significant': bonf_results['significant_mask'],
            'best_sharpe_idx': np.argmax(sharpes),
            'best_sharpe': np.max(sharpes),
            'best_sharpe_significant_fdr': fdr_results['significant_mask'][np.argmax(sharpes)],
            'best_sharpe_significant_bonf': bonf_results['significant_mask'][np.argmax(sharpes)]
        }


def analyze_strategy_significance(
    returns: np.ndarray,
    n_trials: int = 1,
    periods: int = 252,
    confidence: float = 0.95
) -> Dict:
    """
    Comprehensive statistical significance analysis for a strategy.

    This is the main entry point for statistical validation.

    Args:
        returns: Array of strategy returns
        n_trials: Number of strategies/parameters tested (for deflation)
        periods: Annualization factor
        confidence: Confidence level for intervals

    Returns:
        Complete statistical analysis dictionary
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    n = len(returns)

    # Return statistics
    ret_stats = StatisticalSignificance.returns_statistics(returns)

    # Sharpe analysis
    sharpe, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
        returns, confidence, periods
    )
    sharpe_se = StatisticalSignificance.sharpe_standard_error(returns, sharpe, periods)
    sharpe_pval = StatisticalSignificance.sharpe_pvalue(sharpe, n)

    # Minimum track record
    min_track = StatisticalSignificance.minimum_track_record_length(
        sharpe, ret_stats['skewness'], ret_stats['kurtosis'], 0.05, periods
    )

    # Deflated Sharpe (if multiple trials)
    if n_trials > 1:
        # Estimate variance of Sharpe ratios (assume similar strategies have similar variance)
        var_sharpe = sharpe_se ** 2
        dsr = StatisticalSignificance.deflated_sharpe_ratio(
            sharpe, n_trials, var_sharpe,
            ret_stats['skewness'], ret_stats['kurtosis'], n
        )
    else:
        dsr = None

    # Bootstrap analysis
    bootstrap = BootstrapInference.bootstrap_sharpe_ci(returns, 5000, confidence, periods)
    dd_dist = BootstrapInference.bootstrap_drawdown_distribution(returns)

    return {
        # Basic stats
        'n_observations': n,
        'mean_return': ret_stats['mean'],
        'std_return': ret_stats['std'],
        'skewness': ret_stats['skewness'],
        'kurtosis': ret_stats['kurtosis'],
        'is_normal': ret_stats['is_normal'],

        # Sharpe analysis
        'sharpe': sharpe,
        'sharpe_se': sharpe_se,
        'sharpe_ci_lower': ci_lower,
        'sharpe_ci_upper': ci_upper,
        'sharpe_pvalue': sharpe_pval,
        'sharpe_significant': sharpe_pval < (1 - confidence),
        'sharpe_ci_crosses_zero': ci_lower < 0 < ci_upper,

        # Deflated (if applicable)
        'n_trials': n_trials,
        'deflated_sharpe_ratio': dsr,

        # Minimum track record
        'min_track_record_days': min_track,
        'has_sufficient_history': n >= min_track if min_track != np.inf else False,

        # Bootstrap
        'bootstrap_sharpe_ci_lower': bootstrap['ci_lower'],
        'bootstrap_sharpe_ci_upper': bootstrap['ci_upper'],
        'bootstrap_crosses_zero': bootstrap['crosses_zero'],

        # Drawdown distribution
        'max_drawdown': dd_dist['original_dd'],
        'dd_percentile': dd_dist['original_dd_percentile'],
        'dd_95_worst_case': dd_dist['dd_p5'],
        'dd_expected': dd_dist['dd_mean']
    }
