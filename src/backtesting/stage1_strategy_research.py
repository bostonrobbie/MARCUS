"""
Stage 1: Strategy Research (Idea Generation)
=============================================
Uses LLM to generate novel trading strategy ideas, with deduplication
to prevent repeated research of the same concepts.

Fixes Issue #3 (LLM-to-code translation gap) by generating structured
JSON specs that map to known strategy patterns.
Fixes Issue #10 (Duplicate strategy generation) via hash-based dedup.
"""

import json
import logging
import hashlib
import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Known strategy archetypes that map to real implementations
STRATEGY_ARCHETYPES = {
    "orb_breakout": {
        "description": "Opening Range Breakout - trade breakouts from first N minutes",
        "params": ["orb_start", "orb_end", "ema_filter", "atr_filter", "sl_atr_mult", "tp_atr_mult", "atr_max_mult"],
        "variants": ["5min", "10min", "15min", "30min"],
    },
    "orb_vwap": {
        "description": "ORB with VWAP trend filter",
        "params": ["orb_start", "orb_end", "use_vwap"],
        "variants": ["standard"],
    },
    "orb_momentum": {
        "description": "ORB with RSI momentum filter",
        "params": ["orb_start", "orb_end", "rsi_period"],
        "variants": ["standard"],
    },
    "gap_fill_fade": {
        "description": "Fade overnight gaps expecting mean reversion",
        "params": ["gap_min_pct", "sl_atr_mult", "tp_atr_mult"],
        "variants": ["standard", "with_rvol"],
    },
    "es_gap_combo": {
        "description": "ES Opening Range + Gap Follow/Fade combo",
        "params": ["orb_period_min", "rvol_threshold", "hurst_threshold", "gap_min_follow", "gap_min_fade"],
        "variants": ["follow", "fade", "combo"],
    },
    "lunch_hour_breakout": {
        "description": "Breakout from lunch hour consolidation (11:30-13:00)",
        "params": ["range_start", "range_end", "breakout_mult"],
        "variants": ["standard"],
    },
    "eod_momentum": {
        "description": "End of day momentum capture after 14:00",
        "params": ["entry_time", "exit_time", "momentum_period"],
        "variants": ["standard", "with_vwap"],
    },
    "ma_crossover": {
        "description": "Moving average crossover trend following",
        "params": ["short_window", "long_window"],
        "variants": ["sma", "ema"],
    },
}


class StrategyIdeaGenerator:
    """
    Generates novel strategy ideas via LLM, constrained to known archetypes
    that have real backtest implementations.
    """

    def __init__(self, config: Dict[str, Any] = None, db_path: str = "backtests.db"):
        self.config = config or {}
        self.llm = LLMClient.from_config(self.config)
        self.db_path = db_path
        self._tested_hashes: Set[str] = set()
        self._load_tested_strategies()

    def _load_tested_strategies(self):
        """Load previously tested strategy hashes from DB (Issue #11: persistence)."""
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT hash_id FROM backtest_runs WHERE hash_id IS NOT NULL")
            for row in cursor.fetchall():
                self._tested_hashes.add(row[0])
            conn.close()
            logger.info(f"Loaded {len(self._tested_hashes)} previously tested strategies")
        except Exception as e:
            logger.warning(f"Could not load tested strategies: {e}")

    def _hash_idea(self, idea: Dict) -> str:
        """Generate a unique hash for a strategy idea to detect duplicates."""
        key_fields = {
            "archetype": idea.get("archetype", ""),
            "variant": idea.get("variant", ""),
            "params": json.dumps(idea.get("params", {}), sort_keys=True),
        }
        return hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()

    def generate_ideas(self, n_ideas: int = 5, context: str = "") -> List[Dict[str, Any]]:
        """
        Generate strategy ideas using LLM, constrained to implementable archetypes.
        Returns list of structured strategy specifications.
        """
        archetype_list = "\n".join(
            f"- {name}: {info['description']} (variants: {', '.join(info['variants'])})"
            for name, info in STRATEGY_ARCHETYPES.items()
        )

        already_tested = list(self._tested_hashes)[:20]
        dedup_note = ""
        if already_tested:
            dedup_note = f"\nIMPORTANT: We have already tested {len(self._tested_hashes)} strategies. Generate NOVEL variations, not repeats."

        system_prompt = (
            "You are a quantitative trading strategy researcher. "
            "You must generate strategy ideas that map to our existing backtest implementations.\n\n"
            f"Available strategy archetypes:\n{archetype_list}\n\n"
            "Each idea MUST specify:\n"
            "1. 'archetype' - one of the available archetypes above\n"
            "2. 'variant' - a variant of that archetype\n"
            "3. 'params' - specific parameter values to test\n"
            "4. 'hypothesis' - why this parameter combination might work\n"
            "5. 'strategy_name' - a unique descriptive name\n\n"
            "CRITICAL: Only use archetypes from the list above. Do NOT invent new strategy types "
            "that don't have implementations."
            f"{dedup_note}"
        )

        prompt = (
            f"Generate exactly {n_ideas} unique trading strategy ideas as a JSON array.\n"
            f"Market context: {context or 'NQ futures, 5-minute bars, US equity session'}\n\n"
            "Return ONLY a JSON array of strategy objects. Example:\n"
            '[\n'
            '  {\n'
            '    "strategy_name": "Tight_ORB_15m_High_ATR",\n'
            '    "archetype": "orb_breakout",\n'
            '    "variant": "15min",\n'
            '    "params": {"orb_start": "09:30", "orb_end": "09:45", "ema_filter": 50, "sl_atr_mult": 1.5, "tp_atr_mult": 3.0, "atr_max_mult": 2.0},\n'
            '    "hypothesis": "Tighter stops with higher R:R to capture trend continuation"\n'
            '  }\n'
            ']'
        )

        response = self.llm.call(prompt, system=system_prompt, response_format="json")

        if not response.success:
            logger.error(f"LLM idea generation failed: {response.error}")
            return self._fallback_ideas(n_ideas)

        ideas = response.parsed_json
        if not isinstance(ideas, list):
            # Try to extract array from response
            if isinstance(ideas, dict) and "strategies" in ideas:
                ideas = ideas["strategies"]
            else:
                logger.warning("LLM returned non-array response, using fallback ideas")
                return self._fallback_ideas(n_ideas)

        # Validate and deduplicate
        valid_ideas = []
        for idea in ideas:
            if not self._validate_idea(idea):
                continue

            idea_hash = self._hash_idea(idea)
            if idea_hash in self._tested_hashes:
                logger.info(f"Skipping duplicate: {idea.get('strategy_name', 'unknown')}")
                continue

            idea["_hash"] = idea_hash
            valid_ideas.append(idea)
            self._tested_hashes.add(idea_hash)

        if not valid_ideas:
            logger.warning("All LLM ideas were duplicates or invalid, generating fallback")
            return self._fallback_ideas(n_ideas)

        return valid_ideas

    def _validate_idea(self, idea: Dict) -> bool:
        """Validate that an idea maps to a known archetype."""
        archetype = idea.get("archetype", "")
        if archetype not in STRATEGY_ARCHETYPES:
            logger.warning(f"Unknown archetype '{archetype}' - skipping idea")
            return False

        if not idea.get("params"):
            logger.warning(f"Missing params for idea '{idea.get('strategy_name', '')}'")
            return False

        return True

    def _fallback_ideas(self, n_ideas: int) -> List[Dict[str, Any]]:
        """
        Generate deterministic fallback ideas when LLM is unavailable.
        Systematically explores parameter space of known archetypes.
        """
        import itertools

        ideas = []

        orb_periods = [("09:30", "09:35", "5min"), ("09:30", "09:40", "10min"), ("09:30", "09:45", "15min")]
        sl_values = [1.0, 1.5, 2.0, 2.5]
        tp_values = [2.0, 3.0, 4.0, 6.0]
        ema_values = [20, 50, 100]

        for (start, end, variant), sl, tp, ema in itertools.product(orb_periods, sl_values, tp_values, ema_values):
            if tp <= sl:
                continue  # Skip invalid R:R

            idea = {
                "strategy_name": f"ORB_{variant}_SL{sl}_TP{tp}_EMA{ema}",
                "archetype": "orb_breakout",
                "variant": variant,
                "params": {
                    "orb_start": start,
                    "orb_end": end,
                    "ema_filter": ema,
                    "atr_filter": 14,
                    "sl_atr_mult": sl,
                    "tp_atr_mult": tp,
                    "atr_max_mult": 2.5,
                },
                "hypothesis": f"ORB {variant} with {sl}x SL / {tp}x TP, EMA({ema}) filter",
            }

            idea_hash = self._hash_idea(idea)
            if idea_hash not in self._tested_hashes:
                idea["_hash"] = idea_hash
                ideas.append(idea)
                self._tested_hashes.add(idea_hash)

            if len(ideas) >= n_ideas:
                break

        return ideas[:n_ideas]

    def get_tested_count(self) -> int:
        """Return number of strategies already tested."""
        return len(self._tested_hashes)
