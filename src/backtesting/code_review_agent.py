"""
Code Review Agent
=================
LLM-powered agent that reviews trading strategy code for common issues
like look-ahead bias, improper data handling, and other pitfalls.
Runs before optimization to catch problems early.
"""

import os
import re
import ast
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for code issues."""
    CRITICAL = "CRITICAL"  # Must fix, blocks deployment
    WARNING = "WARNING"    # Should fix, may cause problems
    INFO = "INFO"          # Suggestion for improvement


@dataclass
class CodeIssue:
    """A detected issue in strategy code."""
    severity: IssueSeverity
    category: str
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        return f"[{self.severity.value}] {self.category}{loc}: {self.message}"


@dataclass
class CodeReviewResult:
    """Result of code review."""
    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    passed: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    review_time_ms: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def summary(self) -> str:
        """Return summary of review results."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Code Review: {status}",
            f"File: {self.file_path}",
            f"Critical Issues: {self.critical_count}",
            f"Warnings: {self.warning_count}",
        ]
        if self.issues:
            lines.append("\nIssues Found:")
            for issue in self.issues:
                lines.append(f"  {issue}")
        return "\n".join(lines)


class CodeReviewAgent:
    """
    Reviews trading strategy code for common issues.

    Checks:
    - Look-ahead bias (using future data)
    - Trading hours handling
    - Commission/slippage assumptions
    - Vector vs event-driven consistency
    - Data snooping patterns
    """

    # Patterns that indicate potential look-ahead bias
    LOOKAHEAD_PATTERNS = [
        (r'\.shift\s*\(\s*-', "Negative shift may use future data"),
        (r'\.iloc\s*\[\s*\d+\s*:\s*\]', "Forward-looking iloc slice"),
        (r'\.loc\s*\[\s*[^:]+\s*:\s*\]', "Check loc slice direction"),
        (r'future|tomorrow|next_day', "Variable name suggests future data"),
        (r'\.rolling\(.*\)\..*\.shift\(-', "Rolling with negative shift"),
    ]

    # Patterns for proper trading hours handling
    TRADING_HOURS_PATTERNS = [
        (r'9:30|09:30|market_open', "trading_hours_check"),
        (r'16:00|4:00\s*pm|market_close', "trading_hours_check"),
        (r'is_market_hours|trading_session', "trading_hours_check"),
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_config = self.config.get('code_review_agent', {})
        self.enabled_checks = self.agent_config.get('checks', [
            'lookahead_bias',
            'trading_hours',
            'commission_handling',
            'vector_event_consistency'
        ])
        self.block_on_critical = self.agent_config.get('block_on_critical', True)
        self.llm_client = LLMClient.from_config(self.config)

    def review_strategy(self, file_path: str) -> CodeReviewResult:
        """
        Review a strategy file for issues.

        Args:
            file_path: Path to the strategy Python file

        Returns:
            CodeReviewResult with all detected issues
        """
        start_time = datetime.now()

        if not os.path.exists(file_path):
            return CodeReviewResult(
                file_path=file_path,
                issues=[CodeIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="FILE_NOT_FOUND",
                    message=f"Strategy file not found: {file_path}"
                )],
                passed=False,
            )

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return CodeReviewResult(
                file_path=file_path,
                issues=[CodeIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="FILE_READ_ERROR",
                    message=str(e)
                )],
                passed=False,
            )

        issues = []

        # Run static analysis checks
        if 'lookahead_bias' in self.enabled_checks:
            issues.extend(self._check_lookahead_bias(code))

        if 'trading_hours' in self.enabled_checks:
            issues.extend(self._check_trading_hours(code))

        if 'commission_handling' in self.enabled_checks:
            issues.extend(self._check_commission_handling(code))

        if 'vector_event_consistency' in self.enabled_checks:
            issues.extend(self._check_vector_event_consistency(code))

        # Run LLM-based deep analysis if available
        if self.llm_client.is_available():
            llm_issues = self._llm_review(code, file_path)
            issues.extend(llm_issues)
            model_used = self.config.get('llm', {}).get('model', 'unknown')
        else:
            model_used = "STATIC_ANALYSIS_ONLY"

        # Deduplicate issues
        issues = self._deduplicate_issues(issues)

        # Determine pass/fail
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        passed = len(critical_issues) == 0 if self.block_on_critical else True

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        return CodeReviewResult(
            file_path=file_path,
            issues=issues,
            passed=passed,
            timestamp=start_time,
            model_used=model_used,
            review_time_ms=elapsed,
        )

    def _check_lookahead_bias(self, code: str) -> List[CodeIssue]:
        """Check for potential look-ahead bias patterns."""
        issues = []
        lines = code.split('\n')

        for pattern, description in self.LOOKAHEAD_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's a comment
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue

                    issues.append(CodeIssue(
                        severity=IssueSeverity.CRITICAL,
                        category="LOOKAHEAD_BIAS",
                        message=description,
                        line_number=i,
                        code_snippet=line.strip()[:100],
                        suggestion="Ensure data shifts are non-negative (shift(1) not shift(-1))"
                    ))

        # AST-based check for shift() calls
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'shift':
                            # Check for negative argument
                            if node.args and isinstance(node.args[0], ast.UnaryOp):
                                if isinstance(node.args[0].op, ast.USub):
                                    issues.append(CodeIssue(
                                        severity=IssueSeverity.CRITICAL,
                                        category="LOOKAHEAD_BIAS",
                                        message="Negative shift detected - uses future data",
                                        line_number=node.lineno,
                                        suggestion="Use positive shift values only"
                                    ))
        except SyntaxError:
            pass  # Skip AST analysis if code has syntax errors

        return issues

    def _check_trading_hours(self, code: str) -> List[CodeIssue]:
        """Check for trading hours handling."""
        issues = []

        # Check if strategy handles market hours at all
        has_hours_check = any(
            re.search(pattern, code, re.IGNORECASE)
            for pattern, _ in self.TRADING_HOURS_PATTERNS
        )

        # Check for time-based operations
        has_time_ops = bool(re.search(r'\.hour|\.minute|\.time\(\)', code))

        if has_time_ops and not has_hours_check:
            issues.append(CodeIssue(
                severity=IssueSeverity.WARNING,
                category="TRADING_HOURS",
                message="Strategy uses time operations but may not properly filter trading hours",
                suggestion="Add explicit market hours filtering (e.g., 9:30-16:00 for US equities)"
            ))

        # Check for overnight position handling
        if 'overnight' not in code.lower() and 'close_all' not in code.lower():
            if 'intraday' in code.lower() or 'day_trade' in code.lower():
                issues.append(CodeIssue(
                    severity=IssueSeverity.INFO,
                    category="TRADING_HOURS",
                    message="Intraday strategy may benefit from explicit end-of-day position closing",
                    suggestion="Consider adding logic to close positions before market close"
                ))

        return issues

    def _check_commission_handling(self, code: str) -> List[CodeIssue]:
        """Check for proper commission and slippage handling."""
        issues = []

        # Check if commission is mentioned
        has_commission = bool(re.search(r'commission|fee|cost', code, re.IGNORECASE))

        # Check if slippage is mentioned
        has_slippage = bool(re.search(r'slippage|slip|market_impact', code, re.IGNORECASE))

        if not has_commission and not has_slippage:
            issues.append(CodeIssue(
                severity=IssueSeverity.WARNING,
                category="TRANSACTION_COSTS",
                message="Strategy does not appear to account for commission or slippage",
                suggestion="Ensure backtesting engine applies realistic transaction costs"
            ))

        # Check for hardcoded cost values that might be unrealistic
        cost_match = re.search(r'commission\s*[=:]\s*(\d+\.?\d*)', code, re.IGNORECASE)
        if cost_match:
            cost = float(cost_match.group(1))
            if cost == 0:
                issues.append(CodeIssue(
                    severity=IssueSeverity.WARNING,
                    category="TRANSACTION_COSTS",
                    message="Commission set to zero - unrealistic for live trading",
                    suggestion="Use realistic commission: ~$2 for futures, ~$0.005/share for equities"
                ))

        return issues

    def _check_vector_event_consistency(self, code: str) -> List[CodeIssue]:
        """Check for potential inconsistencies between vectorized and event-driven logic."""
        issues = []

        # Check if both paradigms are used
        has_vectorized = bool(re.search(r'\.apply\(|\.rolling\(|vectorize|np\.where', code))
        has_event = bool(re.search(r'def\s+on_bar|def\s+calculate_signals|for.*bar\s+in', code))

        if has_vectorized and has_event:
            issues.append(CodeIssue(
                severity=IssueSeverity.INFO,
                category="CONSISTENCY",
                message="Strategy uses both vectorized and event-driven patterns",
                suggestion="Ensure logic is identical in both modes for consistent backtest results"
            ))

        # Check for state that might not be properly reset
        if has_event:
            # Look for class-level state without reset
            state_vars = re.findall(r'self\.(\w+)\s*=\s*(?!None)', code)
            if state_vars and 'reset' not in code.lower():
                issues.append(CodeIssue(
                    severity=IssueSeverity.WARNING,
                    category="STATE_MANAGEMENT",
                    message=f"Strategy has state variables ({', '.join(state_vars[:3])}) but no reset method",
                    suggestion="Add reset() method to initialize state between backtest runs"
                ))

        return issues

    def _llm_review(self, code: str, file_path: str) -> List[CodeIssue]:
        """Use LLM for deeper code analysis."""
        prompt = f"""Review this trading strategy code for potential issues.

File: {os.path.basename(file_path)}

```python
{code[:8000]}  # Truncate for token limits
```

Analyze for:
1. Look-ahead bias (using future data in calculations)
2. Data snooping / overfitting risks
3. Edge cases not handled (gaps, missing data, market closures)
4. Logic errors in entry/exit conditions
5. Risk management issues

Respond with ONLY valid JSON:
{{
    "issues": [
        {{
            "severity": "CRITICAL" | "WARNING" | "INFO",
            "category": "string",
            "message": "description of the issue",
            "line_hint": "code pattern or approximate location",
            "suggestion": "how to fix"
        }}
    ]
}}

Only include real issues found. Empty array if code looks correct."""

        response = self.llm_client.call(
            prompt=prompt,
            system="You are an expert quantitative developer reviewing trading strategy code. Be thorough but avoid false positives.",
            response_format="json",
        )

        issues = []
        if response.success and response.parsed_json:
            data = response.parsed_json
            for item in data.get('issues', []):
                try:
                    severity = IssueSeverity(item.get('severity', 'INFO'))
                except ValueError:
                    severity = IssueSeverity.INFO

                issues.append(CodeIssue(
                    severity=severity,
                    category=item.get('category', 'LLM_DETECTED'),
                    message=item.get('message', ''),
                    code_snippet=item.get('line_hint'),
                    suggestion=item.get('suggestion'),
                ))

        return issues

    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues based on message similarity."""
        seen = set()
        unique = []

        for issue in issues:
            key = (issue.category, issue.message[:50], issue.line_number)
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        # Sort by severity (CRITICAL first)
        severity_order = {IssueSeverity.CRITICAL: 0, IssueSeverity.WARNING: 1, IssueSeverity.INFO: 2}
        unique.sort(key=lambda x: severity_order.get(x.severity, 3))

        return unique

    def review_multiple(self, file_paths: List[str]) -> Dict[str, CodeReviewResult]:
        """Review multiple strategy files."""
        results = {}
        for path in file_paths:
            results[path] = self.review_strategy(path)
        return results


def review_strategy_file(file_path: str, config: Dict[str, Any] = None) -> CodeReviewResult:
    """
    Convenience function to review a single strategy file.

    Args:
        file_path: Path to the strategy Python file
        config: Application configuration

    Returns:
        CodeReviewResult with all detected issues
    """
    agent = CodeReviewAgent(config)
    return agent.review_strategy(file_path)
