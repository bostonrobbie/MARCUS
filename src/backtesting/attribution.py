"""
Trade Attribution System for Strategy Analysis.

Breaks down WHY trades win or lose by analyzing:
- Entry quality: How close to optimal entry price?
- Exit quality: How close to optimal exit price?
- MAE/MFE: Maximum Adverse/Favorable Excursion during trade
- Timing: Time of day, day of week effects
- Regime: Market condition at entry

This helps answer: "Where is my edge coming from?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


@dataclass
class AttributedTrade:
    """A trade with full attribution analysis."""
    trade_id: int
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float

    # Quality metrics
    entry_quality: float  # Normalized: 0 = worst, 1 = perfect
    exit_quality: float   # Normalized: 0 = worst, 1 = perfect

    # MAE/MFE
    mae: float            # Maximum Adverse Excursion (as % of entry)
    mfe: float            # Maximum Favorable Excursion (as % of entry)
    mae_ratio: float      # MAE / risk taken
    mfe_ratio: float      # MFE / reward captured
    edge_ratio: float     # MFE / MAE

    # Context
    hold_bars: int
    regime: str
    hour_of_day: int
    day_of_week: int
    outcome: TradeOutcome


class TradeAttribution:
    """
    Analyzes trades to determine WHERE edge comes from.
    """

    def __init__(
        self,
        trade_log: List[Dict],
        price_data: pd.DataFrame,
        regime_series: pd.Series = None,
        atr_series: pd.Series = None
    ):
        """
        Args:
            trade_log: List of trade dictionaries from portfolio
            price_data: OHLCV DataFrame with datetime index
            regime_series: Optional series of regime labels aligned to price_data
            atr_series: Optional ATR series for normalization
        """
        self.trade_log = trade_log
        self.price_data = price_data.copy()
        self.regime_series = regime_series
        self.atr_series = atr_series

        # Ensure datetime index
        if not isinstance(self.price_data.index, pd.DatetimeIndex):
            if 'datetime' in self.price_data.columns:
                self.price_data.set_index('datetime', inplace=True)
            self.price_data.index = pd.to_datetime(self.price_data.index)

        self.attributed_trades: List[AttributedTrade] = []
        self.round_trips: List[Dict] = []

    def _build_round_trips(self) -> List[Dict]:
        """
        Convert sequential trade log into round-trip trades (entry + exit pairs).
        """
        if not self.trade_log:
            return []

        round_trips = []
        open_positions: Dict[str, Dict] = {}

        for trade in self.trade_log:
            symbol = trade['symbol']
            side = trade['side']
            qty = trade['quantity']
            price = trade['price']
            timestamp = pd.to_datetime(trade['datetime'])
            realized_pnl = trade.get('realized_pnl', 0)

            if symbol not in open_positions:
                open_positions[symbol] = {'qty': 0, 'entries': []}

            pos = open_positions[symbol]

            # Determine if this is opening or closing
            is_opening = (
                (side == 'BUY' and pos['qty'] >= 0) or
                (side == 'SELL' and pos['qty'] <= 0)
            )

            if is_opening:
                # Adding to position
                pos['entries'].append({
                    'time': timestamp,
                    'price': price,
                    'qty': qty,
                    'side': side
                })
                pos['qty'] += qty if side == 'BUY' else -qty
            else:
                # Closing position
                if pos['entries']:
                    entry = pos['entries'][0]  # FIFO
                    exit_qty = min(abs(pos['qty']), qty)

                    round_trips.append({
                        'symbol': symbol,
                        'side': entry['side'],
                        'entry_time': entry['time'],
                        'exit_time': timestamp,
                        'entry_price': entry['price'],
                        'exit_price': price,
                        'quantity': exit_qty,
                        'realized_pnl': realized_pnl
                    })

                    # Update position
                    if side == 'SELL':
                        pos['qty'] -= qty
                    else:
                        pos['qty'] += qty

                    if abs(pos['qty']) < entry['qty']:
                        pos['entries'].pop(0)

        self.round_trips = round_trips
        return round_trips

    def _calculate_mae_mfe(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate Maximum Adverse and Favorable Excursion during trade.

        MAE: Worst point against the trade
        MFE: Best point in favor of the trade
        """
        # Get price data during trade
        mask = (self.price_data.index >= entry_time) & (self.price_data.index <= exit_time)
        trade_bars = self.price_data[mask]

        if trade_bars.empty:
            return 0.0, 0.0

        if side == 'BUY':
            # Long trade: MAE is lowest low, MFE is highest high
            lowest = trade_bars['low'].min()
            highest = trade_bars['high'].max()

            mae = (lowest - entry_price) / entry_price  # Negative for adverse
            mfe = (highest - entry_price) / entry_price  # Positive for favorable
        else:
            # Short trade: MAE is highest high, MFE is lowest low
            highest = trade_bars['high'].max()
            lowest = trade_bars['low'].min()

            mae = (entry_price - highest) / entry_price  # Negative for adverse
            mfe = (entry_price - lowest) / entry_price   # Positive for favorable

        return mae, mfe

    def _calculate_entry_quality(
        self,
        entry_time: pd.Timestamp,
        entry_price: float,
        side: str
    ) -> float:
        """
        Calculate entry quality: How close to optimal entry?

        For LONG: Optimal entry = Low of entry bar
        For SHORT: Optimal entry = High of entry bar

        Returns: 0 to 1 scale (1 = perfect entry)
        """
        try:
            entry_bar = self.price_data.loc[entry_time]
            bar_range = entry_bar['high'] - entry_bar['low']

            if bar_range == 0:
                return 0.5  # No range, neutral

            if side == 'BUY':
                # Perfect entry = bought at low
                optimal = entry_bar['low']
                worst = entry_bar['high']
            else:
                # Perfect entry = sold at high
                optimal = entry_bar['high']
                worst = entry_bar['low']

            # Normalize: 1 if at optimal, 0 if at worst
            quality = 1 - abs(entry_price - optimal) / bar_range
            return max(0, min(1, quality))

        except (KeyError, IndexError):
            return 0.5

    def _calculate_exit_quality(
        self,
        exit_time: pd.Timestamp,
        exit_price: float,
        side: str
    ) -> float:
        """
        Calculate exit quality: How close to optimal exit?

        For LONG: Optimal exit = High of exit bar
        For SHORT: Optimal exit = Low of exit bar

        Returns: 0 to 1 scale (1 = perfect exit)
        """
        try:
            exit_bar = self.price_data.loc[exit_time]
            bar_range = exit_bar['high'] - exit_bar['low']

            if bar_range == 0:
                return 0.5

            if side == 'BUY':
                # Perfect exit = sold at high
                optimal = exit_bar['high']
                worst = exit_bar['low']
            else:
                # Perfect exit = covered at low
                optimal = exit_bar['low']
                worst = exit_bar['high']

            quality = 1 - abs(exit_price - optimal) / bar_range
            return max(0, min(1, quality))

        except (KeyError, IndexError):
            return 0.5

    def _get_regime_at_time(self, timestamp: pd.Timestamp) -> str:
        """Get market regime at specific timestamp."""
        if self.regime_series is None:
            return "UNKNOWN"

        try:
            # Find closest regime value
            idx = self.regime_series.index.get_indexer([timestamp], method='ffill')[0]
            if idx >= 0:
                regime = self.regime_series.iloc[idx]
                return regime.name if hasattr(regime, 'name') else str(regime)
        except Exception:
            pass

        return "UNKNOWN"

    def attribute_trades(self) -> pd.DataFrame:
        """
        Perform full attribution analysis on all trades.

        Returns:
            DataFrame with attributed trades
        """
        if not self.round_trips:
            self._build_round_trips()

        if not self.round_trips:
            return pd.DataFrame()

        self.attributed_trades = []

        for i, rt in enumerate(self.round_trips):
            entry_time = rt['entry_time']
            exit_time = rt['exit_time']
            entry_price = rt['entry_price']
            exit_price = rt['exit_price']
            side = rt['side']
            pnl = rt.get('realized_pnl', 0)

            # Calculate PnL percentage
            if side == 'BUY':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # Quality metrics
            entry_quality = self._calculate_entry_quality(entry_time, entry_price, side)
            exit_quality = self._calculate_exit_quality(exit_time, exit_price, side)

            # MAE/MFE
            mae, mfe = self._calculate_mae_mfe(entry_time, exit_time, entry_price, side)

            # MAE/MFE ratios
            if abs(mae) > 0:
                edge_ratio = abs(mfe) / abs(mae)
            else:
                edge_ratio = float('inf') if mfe > 0 else 0

            # Calculate how much of MFE was captured
            if mfe > 0:
                mfe_ratio = pnl_pct / mfe if mfe != 0 else 0
            else:
                mfe_ratio = 0

            # Calculate risk used (MAE as proxy)
            mae_ratio = abs(pnl_pct / mae) if mae != 0 else 0

            # Hold time
            try:
                mask = (self.price_data.index >= entry_time) & (self.price_data.index <= exit_time)
                hold_bars = mask.sum()
            except Exception:
                hold_bars = 0

            # Timing context
            hour = entry_time.hour if hasattr(entry_time, 'hour') else 0
            dow = entry_time.dayofweek if hasattr(entry_time, 'dayofweek') else 0

            # Regime
            regime = self._get_regime_at_time(entry_time)

            # Outcome
            if pnl_pct > 0.001:
                outcome = TradeOutcome.WIN
            elif pnl_pct < -0.001:
                outcome = TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.BREAKEVEN

            attributed = AttributedTrade(
                trade_id=i,
                symbol=rt['symbol'],
                side=side,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=rt['quantity'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_quality=entry_quality,
                exit_quality=exit_quality,
                mae=mae,
                mfe=mfe,
                mae_ratio=mae_ratio,
                mfe_ratio=mfe_ratio,
                edge_ratio=edge_ratio,
                hold_bars=hold_bars,
                regime=regime,
                hour_of_day=hour,
                day_of_week=dow,
                outcome=outcome
            )
            self.attributed_trades.append(attributed)

        # Convert to DataFrame
        df = pd.DataFrame([vars(t) for t in self.attributed_trades])
        df['outcome'] = df['outcome'].apply(lambda x: x.value)
        return df

    def calculate_edge_breakdown(self) -> Dict:
        """
        Calculate where the strategy's edge comes from.

        Returns breakdown of edge attribution:
        - Entry timing contribution
        - Exit timing contribution
        - Trade selection (regime filtering)
        - Overall metrics
        """
        if not self.attributed_trades:
            self.attribute_trades()

        if not self.attributed_trades:
            return {}

        df = pd.DataFrame([vars(t) for t in self.attributed_trades])

        total_pnl = df['pnl_pct'].sum()
        n_trades = len(df)
        winners = df[df['pnl_pct'] > 0]
        losers = df[df['pnl_pct'] < 0]

        # Entry quality correlation with returns
        entry_corr = df['entry_quality'].corr(df['pnl_pct'])

        # Exit quality correlation with returns
        exit_corr = df['exit_quality'].corr(df['pnl_pct'])

        # Calculate entry edge: difference in avg entry quality for winners vs losers
        if len(winners) > 0 and len(losers) > 0:
            entry_diff = winners['entry_quality'].mean() - losers['entry_quality'].mean()
            exit_diff = winners['exit_quality'].mean() - losers['exit_quality'].mean()
        else:
            entry_diff = 0
            exit_diff = 0

        # MAE/MFE analysis
        avg_mae = df['mae'].mean()
        avg_mfe = df['mfe'].mean()
        avg_edge_ratio = df['edge_ratio'].replace([np.inf, -np.inf], np.nan).mean()

        # MFE capture rate: how much of available profit was captured
        mfe_capture = df['mfe_ratio'].mean()

        # Regime breakdown
        regime_stats = df.groupby('regime').agg({
            'pnl_pct': ['mean', 'sum', 'count'],
            'entry_quality': 'mean',
            'exit_quality': 'mean'
        }).round(4)

        # Time breakdown
        hour_stats = df.groupby('hour_of_day')['pnl_pct'].agg(['mean', 'sum', 'count'])
        dow_stats = df.groupby('day_of_week')['pnl_pct'].agg(['mean', 'sum', 'count'])

        # Estimate contribution percentages
        # Simplified model: entry, exit, and selection each contribute
        total_corr = abs(entry_corr) + abs(exit_corr) + 0.01
        entry_contrib = abs(entry_corr) / total_corr * 100 if not np.isnan(entry_corr) else 33
        exit_contrib = abs(exit_corr) / total_corr * 100 if not np.isnan(exit_corr) else 33
        selection_contrib = 100 - entry_contrib - exit_contrib

        return {
            # Overall
            'total_pnl_pct': total_pnl,
            'n_trades': n_trades,
            'win_rate': len(winners) / n_trades if n_trades > 0 else 0,
            'avg_winner': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loser': losers['pnl_pct'].mean() if len(losers) > 0 else 0,

            # Quality metrics
            'avg_entry_quality': df['entry_quality'].mean(),
            'avg_exit_quality': df['exit_quality'].mean(),
            'entry_quality_correlation': entry_corr,
            'exit_quality_correlation': exit_corr,
            'winners_entry_quality': winners['entry_quality'].mean() if len(winners) > 0 else 0,
            'losers_entry_quality': losers['entry_quality'].mean() if len(losers) > 0 else 0,

            # MAE/MFE
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'avg_edge_ratio': avg_edge_ratio,
            'mfe_capture_rate': mfe_capture,

            # Edge breakdown (estimated %)
            'entry_contribution_pct': entry_contrib,
            'exit_contribution_pct': exit_contrib,
            'selection_contribution_pct': selection_contrib,

            # Detailed breakdowns
            'regime_stats': regime_stats.to_dict() if not regime_stats.empty else {},
            'hour_stats': hour_stats.to_dict() if not hour_stats.empty else {},
            'dow_stats': dow_stats.to_dict() if not dow_stats.empty else {}
        }


class MAE_MFE_Analysis:
    """
    Detailed MAE/MFE analysis for stop loss and take profit optimization.
    """

    def __init__(self, attributed_trades: List[AttributedTrade]):
        self.trades = attributed_trades

    def compute_distributions(self) -> Dict:
        """Compute MAE and MFE distributions."""
        if not self.trades:
            return {}

        maes = [t.mae for t in self.trades]
        mfes = [t.mfe for t in self.trades]

        return {
            'mae_percentiles': {
                'p5': np.percentile(maes, 5),
                'p25': np.percentile(maes, 25),
                'p50': np.percentile(maes, 50),
                'p75': np.percentile(maes, 75),
                'p95': np.percentile(maes, 95)
            },
            'mfe_percentiles': {
                'p5': np.percentile(mfes, 5),
                'p25': np.percentile(mfes, 25),
                'p50': np.percentile(mfes, 50),
                'p75': np.percentile(mfes, 75),
                'p95': np.percentile(mfes, 95)
            },
            'mae_mean': np.mean(maes),
            'mfe_mean': np.mean(mfes),
            'mae_std': np.std(maes),
            'mfe_std': np.std(mfes)
        }

    def optimal_stop_analysis(self) -> Dict:
        """
        Suggest optimal stop loss based on MAE distribution.

        Logic: Stop should be wide enough to not get stopped out on winners,
        but tight enough to limit losses on losers.
        """
        winners = [t for t in self.trades if t.outcome == TradeOutcome.WIN]
        losers = [t for t in self.trades if t.outcome == TradeOutcome.LOSS]

        if not winners or not losers:
            return {'recommendation': 'Insufficient data'}

        winner_maes = [abs(t.mae) for t in winners]
        loser_maes = [abs(t.mae) for t in losers]

        # Stop should be beyond 95% of winner MAEs (don't stop out winners)
        suggested_stop = np.percentile(winner_maes, 95)

        # How many losers would have been stopped earlier?
        stopped_early = sum(1 for mae in loser_maes if mae > suggested_stop)
        stopped_early_pct = (stopped_early / len(loser_maes) * 100) if len(loser_maes) > 0 else 0.0

        return {
            'suggested_stop_pct': suggested_stop,
            'winner_mae_p95': np.percentile(winner_maes, 95),
            'loser_mae_p50': np.percentile(loser_maes, 50),
            'losers_stopped_early_pct': stopped_early_pct,
            'recommendation': f"Stop at {suggested_stop:.2%} would preserve {100-stopped_early_pct:.0f}% of winners while stopping {stopped_early_pct:.0f}% of losers earlier"
        }

    def optimal_target_analysis(self) -> Dict:
        """
        Suggest optimal take profit based on MFE distribution.

        Logic: Target should capture most of available MFE on winners.
        """
        winners = [t for t in self.trades if t.outcome == TradeOutcome.WIN]

        if not winners:
            return {'recommendation': 'Insufficient data'}

        mfes = [t.mfe for t in winners]
        actual_captures = [t.pnl_pct for t in winners]

        # Average capture rate
        capture_rates = [ac / mfe if mfe > 0 else 0 for ac, mfe in zip(actual_captures, mfes)]
        avg_capture = np.mean(capture_rates)

        # Suggested target: p75 of MFEs (capture 75% of moves)
        suggested_target = np.percentile(mfes, 75)

        return {
            'suggested_target_pct': suggested_target,
            'current_capture_rate': avg_capture,
            'mfe_p50': np.percentile(mfes, 50),
            'mfe_p75': np.percentile(mfes, 75),
            'mfe_p90': np.percentile(mfes, 90),
            'recommendation': f"Target at {suggested_target:.2%} would capture 75% of favorable moves. Current capture rate: {avg_capture:.0%}"
        }


class WinLossDecomposition:
    """
    Decompose strategy performance by various factors.
    """

    def __init__(self, attributed_trades: List[AttributedTrade]):
        self.trades = attributed_trades

    def decompose_by_regime(self) -> Dict:
        """Break down performance by market regime."""
        if not self.trades:
            return {}

        regime_groups = {}
        for t in self.trades:
            if t.regime not in regime_groups:
                regime_groups[t.regime] = []
            regime_groups[t.regime].append(t)

        results = {}
        for regime, trades in regime_groups.items():
            pnls = [t.pnl_pct for t in trades]
            wins = [t for t in trades if t.outcome == TradeOutcome.WIN]

            results[regime] = {
                'n_trades': len(trades),
                'total_return': sum(pnls),
                'avg_return': np.mean(pnls),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'sharpe': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            }

        return results

    def decompose_by_time(self) -> Dict:
        """Break down performance by time factors."""
        if not self.trades:
            return {}

        # By hour
        hour_groups = {}
        for t in self.trades:
            h = t.hour_of_day
            if h not in hour_groups:
                hour_groups[h] = []
            hour_groups[h].append(t)

        hour_results = {}
        for hour, trades in hour_groups.items():
            pnls = [t.pnl_pct for t in trades]
            wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
            hour_results[hour] = {
                'n_trades': len(trades),
                'total_return': sum(pnls),
                'win_rate': len(wins) / len(trades) if trades else 0
            }

        # By day of week
        dow_groups = {}
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for t in self.trades:
            d = dow_names[t.day_of_week] if t.day_of_week < len(dow_names) else str(t.day_of_week)
            if d not in dow_groups:
                dow_groups[d] = []
            dow_groups[d].append(t)

        dow_results = {}
        for dow, trades in dow_groups.items():
            pnls = [t.pnl_pct for t in trades]
            wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
            dow_results[dow] = {
                'n_trades': len(trades),
                'total_return': sum(pnls),
                'win_rate': len(wins) / len(trades) if trades else 0
            }

        return {
            'by_hour': hour_results,
            'by_day_of_week': dow_results
        }

    def decompose_by_hold_time(self) -> Dict:
        """Break down performance by trade duration."""
        if not self.trades:
            return {}

        # Bucket by hold time
        buckets = {
            'very_short (1-5 bars)': [],
            'short (6-20 bars)': [],
            'medium (21-50 bars)': [],
            'long (51+ bars)': []
        }

        for t in self.trades:
            if t.hold_bars <= 5:
                buckets['very_short (1-5 bars)'].append(t)
            elif t.hold_bars <= 20:
                buckets['short (6-20 bars)'].append(t)
            elif t.hold_bars <= 50:
                buckets['medium (21-50 bars)'].append(t)
            else:
                buckets['long (51+ bars)'].append(t)

        results = {}
        for bucket, trades in buckets.items():
            if trades:
                pnls = [t.pnl_pct for t in trades]
                wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
                results[bucket] = {
                    'n_trades': len(trades),
                    'total_return': sum(pnls),
                    'avg_return': np.mean(pnls),
                    'win_rate': len(wins) / len(trades)
                }

        return results

    def generate_insights(self) -> List[str]:
        """Generate actionable insights from decomposition."""
        insights = []

        # Regime insights
        regime_data = self.decompose_by_regime()
        if regime_data:
            best_regime = max(regime_data.items(), key=lambda x: x[1]['avg_return'])
            worst_regime = min(regime_data.items(), key=lambda x: x[1]['avg_return'])

            if best_regime[1]['avg_return'] > 0:
                insights.append(
                    f"Best performance in {best_regime[0]} regime "
                    f"(+{best_regime[1]['avg_return']:.2%} avg, {best_regime[1]['win_rate']:.0%} win rate)"
                )

            if worst_regime[1]['avg_return'] < 0:
                insights.append(
                    f"Consider avoiding {worst_regime[0]} regime "
                    f"({worst_regime[1]['avg_return']:.2%} avg, {worst_regime[1]['n_trades']} trades)"
                )

        # Time insights
        time_data = self.decompose_by_time()
        if time_data.get('by_day_of_week'):
            dow = time_data['by_day_of_week']
            worst_day = min(dow.items(), key=lambda x: x[1]['total_return'])
            if worst_day[1]['total_return'] < 0 and worst_day[1]['n_trades'] >= 5:
                insights.append(
                    f"Consider avoiding {worst_day[0]} - "
                    f"negative edge ({worst_day[1]['total_return']:.2%} total)"
                )

        # Hold time insights
        hold_data = self.decompose_by_hold_time()
        if hold_data:
            for bucket, stats in hold_data.items():
                if stats['n_trades'] >= 5 and stats['avg_return'] < -0.01:
                    insights.append(
                        f"Poor performance on {bucket} trades - "
                        f"consider adjusting exit timing"
                    )

        return insights


def analyze_trades(
    trade_log: List[Dict],
    price_data: pd.DataFrame,
    regime_series: pd.Series = None
) -> Dict:
    """
    Main entry point for trade attribution analysis.

    Args:
        trade_log: Portfolio trade log
        price_data: OHLCV DataFrame
        regime_series: Optional regime labels

    Returns:
        Complete attribution analysis
    """
    # Build attribution
    attribution = TradeAttribution(trade_log, price_data, regime_series)
    trades_df = attribution.attribute_trades()

    if trades_df.empty:
        return {'error': 'No trades to analyze'}

    # Edge breakdown
    edge = attribution.calculate_edge_breakdown()

    # MAE/MFE analysis
    mae_mfe = MAE_MFE_Analysis(attribution.attributed_trades)
    mae_mfe_dist = mae_mfe.compute_distributions()
    stop_analysis = mae_mfe.optimal_stop_analysis()
    target_analysis = mae_mfe.optimal_target_analysis()

    # Decomposition
    decomp = WinLossDecomposition(attribution.attributed_trades)
    regime_decomp = decomp.decompose_by_regime()
    time_decomp = decomp.decompose_by_time()
    hold_decomp = decomp.decompose_by_hold_time()
    insights = decomp.generate_insights()

    return {
        'trades_df': trades_df,
        'edge_breakdown': edge,
        'mae_mfe_distribution': mae_mfe_dist,
        'optimal_stop': stop_analysis,
        'optimal_target': target_analysis,
        'regime_decomposition': regime_decomp,
        'time_decomposition': time_decomp,
        'hold_time_decomposition': hold_decomp,
        'insights': insights
    }
