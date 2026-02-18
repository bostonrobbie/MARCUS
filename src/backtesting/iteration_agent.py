"""
Iteration Agent
===============
Orchestrates autonomous strategy improvement loops.
Analyzes results, generates code modifications, applies them safely,
and re-runs optimization until success or max iterations.
"""

import os
import ast
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import re

from .llm_client import LLMClient
from .decision_agent import Decision, DecisionResult, DecisionContext, ValidationDecisionAgent
from .logic_agent import LogicAgent, LogicSuggestion, StrategyHypothesis
from .param_agent import ParameterAgent, GridSuggestion

logger = logging.getLogger(__name__)


@dataclass
class CodeModification:
    """A proposed code modification."""
    file_path: str
    modification_type: str  # ADD_FILTER, MODIFY_ENTRY, MODIFY_EXIT, ADD_PARAMETER
    description: str
    original_code: Optional[str] = None
    new_code: Optional[str] = None
    insert_after: Optional[str] = None  # Pattern to find insertion point
    replace_pattern: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration_number: int
    decision: Decision
    decision_result: DecisionResult
    modifications_applied: List[CodeModification] = field(default_factory=list)
    param_changes: Optional[GridSuggestion] = None
    logic_suggestion: Optional[LogicSuggestion] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    improvement_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Return summary of iteration."""
        lines = [
            f"Iteration {self.iteration_number}",
            f"Decision: {self.decision.value}",
            f"Sharpe: {self.metrics_before.get('sharpe', 0):.2f} -> {self.metrics_after.get('sharpe', 0):.2f}",
            f"Improvement: {self.improvement_pct:+.1%}",
        ]
        if self.modifications_applied:
            lines.append(f"Modifications: {len(self.modifications_applied)}")
        return "\n".join(lines)


@dataclass
class IterationHistory:
    """Complete history of iteration loop."""
    strategy_name: str
    symbol: str
    iterations: List[IterationResult] = field(default_factory=list)
    final_decision: Optional[Decision] = None
    total_improvement_pct: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'iteration_count': self.iteration_count,
            'final_decision': self.final_decision.value if self.final_decision else None,
            'total_improvement_pct': self.total_improvement_pct,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'iterations': [
                {
                    'number': it.iteration_number,
                    'decision': it.decision.value,
                    'improvement_pct': it.improvement_pct,
                    'modifications': len(it.modifications_applied),
                }
                for it in self.iterations
            ],
        }

    def summary(self) -> str:
        """Return full history summary."""
        lines = [
            "=" * 50,
            f"ITERATION HISTORY: {self.strategy_name}",
            "=" * 50,
            f"Symbol: {self.symbol}",
            f"Total Iterations: {self.iteration_count}",
            f"Final Decision: {self.final_decision.value if self.final_decision else 'N/A'}",
            f"Total Improvement: {self.total_improvement_pct:+.1%}",
            "",
            "Iteration Timeline:",
        ]

        for it in self.iterations:
            lines.append(f"  {it.summary()}")
            lines.append("")

        return "\n".join(lines)


class CodeModifier:
    """Safely applies code modifications to strategy files."""

    def __init__(self, backup_enabled: bool = True):
        self.backup_enabled = backup_enabled
        self.backups = {}  # file_path -> backup_path

    def create_backup(self, file_path: str) -> str:
        """Create backup of file before modification."""
        if not self.backup_enabled:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"

        shutil.copy2(file_path, backup_path)
        self.backups[file_path] = backup_path

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    def restore_backup(self, file_path: str) -> bool:
        """Restore file from backup."""
        backup_path = self.backups.get(file_path)
        if not backup_path or not os.path.exists(backup_path):
            logger.error(f"No backup found for {file_path}")
            return False

        shutil.copy2(backup_path, file_path)
        logger.info(f"Restored from backup: {backup_path}")
        return True

    def apply_modification(self, mod: CodeModification) -> CodeModification:
        """
        Apply a code modification safely.

        Args:
            mod: CodeModification to apply

        Returns:
            Updated CodeModification with success/error status
        """
        if not os.path.exists(mod.file_path):
            mod.error = f"File not found: {mod.file_path}"
            return mod

        try:
            # Read current content
            with open(mod.file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            mod.original_code = original_content

            # Create backup
            self.create_backup(mod.file_path)

            # Apply modification based on type
            if mod.modification_type == "ADD_FILTER" and mod.insert_after:
                new_content = self._insert_after_pattern(
                    original_content, mod.insert_after, mod.new_code
                )
            elif mod.modification_type == "REPLACE" and mod.replace_pattern:
                new_content = re.sub(mod.replace_pattern, mod.new_code, original_content)
            elif mod.new_code:
                # Direct replacement at end of class/file
                new_content = self._append_to_class(original_content, mod.new_code)
            else:
                mod.error = "No modification action specified"
                return mod

            # Validate syntax
            if not self._validate_syntax(new_content):
                mod.error = "Modified code has syntax errors"
                self.restore_backup(mod.file_path)
                return mod

            # Write modified content
            with open(mod.file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            mod.success = True
            logger.info(f"Applied modification: {mod.description}")

        except Exception as e:
            mod.error = str(e)
            logger.error(f"Failed to apply modification: {e}")
            self.restore_backup(mod.file_path)

        return mod

    def _insert_after_pattern(self, content: str, pattern: str, new_code: str) -> str:
        """Insert code after a pattern."""
        match = re.search(pattern, content, re.MULTILINE)
        if not match:
            raise ValueError(f"Pattern not found: {pattern}")

        insert_pos = match.end()
        return content[:insert_pos] + "\n" + new_code + content[insert_pos:]

    def _append_to_class(self, content: str, new_code: str) -> str:
        """Append code to the main class in the file."""
        # Find last class definition
        class_matches = list(re.finditer(r'^class\s+\w+.*?:', content, re.MULTILINE))
        if not class_matches:
            # No class, append to end
            return content + "\n\n" + new_code

        # Find the indentation level
        lines = content.split('\n')
        insert_line = len(lines) - 1

        # Find last non-empty line before end
        while insert_line > 0 and not lines[insert_line].strip():
            insert_line -= 1

        # Insert with proper indentation
        indent = "    "  # Standard Python indent
        indented_code = "\n".join(indent + line for line in new_code.split('\n'))

        lines.insert(insert_line + 1, "\n" + indented_code)
        return '\n'.join(lines)

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return False


class IterationAgent:
    """
    Orchestrates autonomous strategy improvement.

    Workflow:
    1. Run optimization and get results
    2. Decision Agent evaluates
    3. If ITERATE:
        a. Logic Agent identifies weaknesses
        b. Generate code modifications
        c. Apply modifications safely
        d. Re-run optimization
        e. Loop until DEPLOY/ABANDON or max iterations
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_config = self.config.get('iteration_agent', {})
        self.max_iterations = self.agent_config.get('max_iterations', 5)
        self.auto_apply = self.agent_config.get('auto_apply_changes', False)
        self.backup_files = self.agent_config.get('backup_strategy_files', True)

        # Initialize sub-agents
        self.decision_agent = ValidationDecisionAgent(config)
        self.logic_agent = LogicAgent(config)
        self.param_agent = ParameterAgent(config)
        self.llm_client = LLMClient.from_config(config)
        self.code_modifier = CodeModifier(backup_enabled=self.backup_files)

    def run_iteration_loop(
        self,
        strategy_name: str,
        symbol: str,
        strategy_file: str,
        run_optimization: Callable[[], Dict[str, Any]],
        initial_stats: Optional[Dict[str, Any]] = None,
    ) -> IterationHistory:
        """
        Run the full iteration loop.

        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            strategy_file: Path to strategy source file
            run_optimization: Callback function to run optimization, returns stats dict
            initial_stats: Optional initial stats to skip first optimization

        Returns:
            IterationHistory with all iteration results
        """
        history = IterationHistory(
            strategy_name=strategy_name,
            symbol=symbol,
        )

        # Get initial stats
        if initial_stats:
            current_stats = initial_stats
        else:
            logger.info("Running initial optimization...")
            current_stats = run_optimization()

        initial_sharpe = current_stats.get('Sharpe', current_stats.get('sharpe_ratio', 0))

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*50}")

            # Build context and evaluate
            context = DecisionContext.from_stats(
                current_stats, strategy_name, symbol,
                current_stats.get('params', {})
            )
            decision_result = self.decision_agent.evaluate(context)

            # Record metrics
            iteration_result = IterationResult(
                iteration_number=iteration,
                decision=decision_result.decision,
                decision_result=decision_result,
                metrics_before={
                    'sharpe': current_stats.get('Sharpe', current_stats.get('sharpe_ratio', 0)),
                    'return': current_stats.get('Total Return', current_stats.get('total_return', 0)),
                    'drawdown': current_stats.get('Max Drawdown', current_stats.get('max_drawdown', 0)),
                },
            )

            logger.info(f"Decision: {decision_result.decision.value}")
            logger.info(f"Reasoning: {decision_result.reasoning}")

            # Check termination conditions
            if decision_result.decision == Decision.DEPLOY:
                logger.info("Strategy DEPLOYED - optimization successful!")
                iteration_result.metrics_after = iteration_result.metrics_before
                history.iterations.append(iteration_result)
                history.final_decision = Decision.DEPLOY
                break

            if decision_result.decision == Decision.ABANDON:
                logger.info("Strategy ABANDONED - not viable for improvement")
                iteration_result.metrics_after = iteration_result.metrics_before
                history.iterations.append(iteration_result)
                history.final_decision = Decision.ABANDON
                break

            # ITERATE: Apply improvements
            logger.info("Generating improvements...")

            # Get logic suggestions
            logic_suggestion = self.logic_agent.generate_full_suggestion(
                current_stats,
                trade_analysis=current_stats.get('trade_analysis'),
                strategy_code=self._read_strategy_code(strategy_file),
            )
            iteration_result.logic_suggestion = logic_suggestion

            # Generate and apply modifications
            if logic_suggestion.priority_hypothesis and self.auto_apply:
                modification = self._generate_modification(
                    strategy_file,
                    logic_suggestion.priority_hypothesis,
                )
                if modification:
                    applied = self.code_modifier.apply_modification(modification)
                    iteration_result.modifications_applied.append(applied)

                    if not applied.success:
                        logger.warning(f"Modification failed: {applied.error}")
            else:
                logger.info("Auto-apply disabled or no hypothesis - showing suggestions only")
                if logic_suggestion.priority_hypothesis:
                    logger.info(f"Suggested: {logic_suggestion.priority_hypothesis.name}")
                    logger.info(f"  {logic_suggestion.priority_hypothesis.description}")

            # Re-run optimization
            logger.info("Re-running optimization...")
            try:
                new_stats = run_optimization()
                current_stats = new_stats
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                # Restore backup and continue
                if iteration_result.modifications_applied:
                    self.code_modifier.restore_backup(strategy_file)
                current_stats = iteration_result.metrics_before
                iteration_result.metrics_after = iteration_result.metrics_before
                history.iterations.append(iteration_result)
                continue

            # Record results
            new_sharpe = new_stats.get('Sharpe', new_stats.get('sharpe_ratio', 0))
            iteration_result.metrics_after = {
                'sharpe': new_sharpe,
                'return': new_stats.get('Total Return', new_stats.get('total_return', 0)),
                'drawdown': new_stats.get('Max Drawdown', new_stats.get('max_drawdown', 0)),
            }

            old_sharpe = iteration_result.metrics_before.get('sharpe', 0)
            if old_sharpe != 0:
                iteration_result.improvement_pct = (new_sharpe - old_sharpe) / abs(old_sharpe)
            else:
                iteration_result.improvement_pct = 0

            history.iterations.append(iteration_result)

            logger.info(f"Sharpe: {old_sharpe:.2f} -> {new_sharpe:.2f} ({iteration_result.improvement_pct:+.1%})")

        # Finalize history
        history.completed_at = datetime.now()

        if not history.final_decision:
            # Hit max iterations without DEPLOY/ABANDON
            last_decision = history.iterations[-1].decision if history.iterations else Decision.ITERATE
            history.final_decision = last_decision

        # Calculate total improvement
        if history.iterations:
            final_sharpe = history.iterations[-1].metrics_after.get('sharpe', 0)
            if initial_sharpe != 0:
                history.total_improvement_pct = (final_sharpe - initial_sharpe) / abs(initial_sharpe)

        return history

    def _read_strategy_code(self, file_path: str) -> Optional[str]:
        """Read strategy source code."""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def _generate_modification(self, strategy_file: str,
                               hypothesis: StrategyHypothesis) -> Optional[CodeModification]:
        """Generate code modification from hypothesis."""
        if not hypothesis.implementation_code:
            return None

        return CodeModification(
            file_path=strategy_file,
            modification_type="ADD_FILTER" if hypothesis.category == "FILTER" else "REPLACE",
            description=hypothesis.name,
            new_code=hypothesis.implementation_code,
            insert_after=r"def\s+calculate_signals\s*\([^)]*\)\s*:",  # After signal generation
        )

    def single_iteration(
        self,
        stats: Dict[str, Any],
        strategy_name: str,
        symbol: str,
        strategy_file: Optional[str] = None,
    ) -> IterationResult:
        """
        Run a single iteration (for manual control).

        Args:
            stats: Current backtest statistics
            strategy_name: Strategy name
            symbol: Trading symbol
            strategy_file: Optional path to strategy file

        Returns:
            IterationResult with decision and suggestions
        """
        context = DecisionContext.from_stats(stats, strategy_name, symbol)
        decision_result = self.decision_agent.evaluate(context)

        result = IterationResult(
            iteration_number=1,
            decision=decision_result.decision,
            decision_result=decision_result,
            metrics_before={
                'sharpe': stats.get('Sharpe', stats.get('sharpe_ratio', 0)),
                'return': stats.get('Total Return', stats.get('total_return', 0)),
            },
        )

        if decision_result.decision == Decision.ITERATE:
            code = self._read_strategy_code(strategy_file) if strategy_file else None
            result.logic_suggestion = self.logic_agent.generate_full_suggestion(
                stats, strategy_code=code
            )

        return result


def run_autonomous_optimization(
    strategy_name: str,
    symbol: str,
    strategy_file: str,
    optimization_callback: Callable[[], Dict[str, Any]],
    config: Dict[str, Any] = None,
    initial_stats: Optional[Dict[str, Any]] = None,
) -> IterationHistory:
    """
    Convenience function to run autonomous optimization loop.

    Args:
        strategy_name: Name of the strategy
        symbol: Trading symbol
        strategy_file: Path to strategy source file
        optimization_callback: Function that runs optimization and returns stats
        config: Application configuration
        initial_stats: Optional initial stats

    Returns:
        IterationHistory with complete results
    """
    agent = IterationAgent(config)
    return agent.run_iteration_loop(
        strategy_name=strategy_name,
        symbol=symbol,
        strategy_file=strategy_file,
        run_optimization=optimization_callback,
        initial_stats=initial_stats,
    )
