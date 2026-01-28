"""
Failure Explainer for Agentic CI

Analyzes CI failures to provide:
- Root cause identification
- Human-readable explanations
- Fix suggestions
- Pattern recognition for recurring issues
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import asyncio
from loguru import logger

from .llm_client import LLMClient
from .pattern_store import PatternStore


class FailureType(str, Enum):
    """Categories of CI failures."""
    # Build failures
    COMPILATION_ERROR = "compilation_error"
    DEPENDENCY_ERROR = "dependency_error"
    SYNTAX_ERROR = "syntax_error"

    # Test failures
    ASSERTION_FAILURE = "assertion_failure"
    TIMEOUT = "timeout"
    SEGFAULT = "segfault"
    MEMORY_ERROR = "memory_error"

    # Infrastructure failures
    NETWORK_ERROR = "network_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PERMISSION_DENIED = "permission_denied"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Code quality failures
    LINT_ERROR = "lint_error"
    TYPE_ERROR = "type_error"
    SECURITY_VIOLATION = "security_violation"

    # Other
    FLAKY_TEST = "flaky_test"
    UNKNOWN = "unknown"


@dataclass
class FailureExplanation:
    """Detailed explanation of a CI failure."""
    failure_type: FailureType
    root_cause: str
    explanation: str

    # Code references
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None

    # Fix suggestions
    fix_suggestions: List[str] = field(default_factory=list)
    fix_code: Optional[str] = None

    # Confidence and metadata
    confidence: float = 0.5
    is_flaky: bool = False
    similar_failures: List[Dict] = field(default_factory=list)

    # Links
    documentation_links: List[str] = field(default_factory=list)
    related_commits: List[str] = field(default_factory=list)


class FailureExplainer:
    """
    Explains CI failures with LLM-powered analysis.
    Learns from historical patterns to improve over time.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pattern_store: Optional[PatternStore] = None
    ):
        """
        Initialize failure explainer.

        Args:
            llm_client: LLM client for analysis
            pattern_store: Historical pattern storage
        """
        self.llm = llm_client or LLMClient()
        self.pattern_store = pattern_store or PatternStore()

        # Failure type detection patterns
        self.failure_patterns = {
            FailureType.COMPILATION_ERROR: [
                r"error:.*compilation",
                r"cannot find symbol",
                r"undefined reference",
                r"error\[E\d+\]",  # Rust errors
                r"error: aborting due to",
            ],
            FailureType.DEPENDENCY_ERROR: [
                r"Could not resolve dependencies",
                r"ModuleNotFoundError",
                r"Cannot find module",
                r"package .* not found",
                r"No matching version",
                r"ENOENT.*node_modules",
            ],
            FailureType.SYNTAX_ERROR: [
                r"SyntaxError:",
                r"Parse error:",
                r"Unexpected token",
                r"invalid syntax",
            ],
            FailureType.ASSERTION_FAILURE: [
                r"AssertionError",
                r"assert.*failed",
                r"Expected .* but got",
                r"FAILED.*assert",
                r"expect\(.*\)\.to",
            ],
            FailureType.TIMEOUT: [
                r"TimeoutError",
                r"timed out",
                r"exceeded.*timeout",
                r"deadline exceeded",
            ],
            FailureType.MEMORY_ERROR: [
                r"OutOfMemoryError",
                r"MemoryError",
                r"heap.*exhausted",
                r"allocation failed",
                r"OOM",
            ],
            FailureType.NETWORK_ERROR: [
                r"ConnectionError",
                r"ECONNREFUSED",
                r"getaddrinfo.*failed",
                r"network.*unreachable",
                r"connection reset",
            ],
            FailureType.PERMISSION_DENIED: [
                r"PermissionError",
                r"EACCES",
                r"permission denied",
                r"Access is denied",
            ],
            FailureType.LINT_ERROR: [
                r"Linting error",
                r"eslint.*error",
                r"flake8.*error",
                r"rubocop.*offense",
                r"clippy.*error",
            ],
            FailureType.TYPE_ERROR: [
                r"TypeError:",
                r"type.*mismatch",
                r"TS\d+:",  # TypeScript errors
                r"mypy.*error",
            ],
            FailureType.SECURITY_VIOLATION: [
                r"security.*violation",
                r"vulnerability.*found",
                r"CVE-\d+",
                r"SNYK-.*",
            ],
        }

        logger.info("FailureExplainer initialized")

    async def explain(
        self,
        failure_log: str,
        code_context: Optional[str] = None,
        diff: Optional[str] = None,
        commit_info: Optional[Dict] = None
    ) -> FailureExplanation:
        """
        Explain a CI failure.

        Args:
            failure_log: The failure output/logs
            code_context: Relevant code around the failure
            diff: Recent changes that may have caused failure
            commit_info: Commit metadata

        Returns:
            FailureExplanation with root cause and suggestions
        """
        # Classify the failure type
        failure_type = self._classify_failure(failure_log)

        # Extract file/line information
        file_path, line_number, code_snippet = self._extract_location(failure_log)

        # Find similar past failures
        similar_failures = await self.pattern_store.find_similar_failures(
            failure_log=failure_log,
            failure_type=failure_type.value,
            limit=3
        )

        # Get LLM explanation
        llm_result = await self.llm.explain_failure(
            failure_log=failure_log,
            failure_type=failure_type.value,
            code_context=code_context,
            diff=diff,
            similar_failures=similar_failures
        )

        # Check if this looks like a flaky test
        is_flaky = self._detect_flakiness(
            failure_log=failure_log,
            failure_type=failure_type,
            similar_failures=similar_failures
        )

        # Build explanation
        explanation = FailureExplanation(
            failure_type=failure_type,
            root_cause=llm_result.get("root_cause", "Unable to determine"),
            explanation=llm_result.get("explanation", "Analysis failed"),
            file_path=file_path,
            line_number=line_number,
            code_snippet=code_snippet,
            fix_suggestions=llm_result.get("fix_suggestions", []),
            confidence=llm_result.get("confidence", 0.5),
            is_flaky=is_flaky,
            similar_failures=similar_failures,
        )

        # Add documentation links based on failure type
        explanation.documentation_links = self._get_doc_links(failure_type)

        # Record for learning
        await self.pattern_store.record_failure_pattern(
            failure_type=failure_type.value,
            error_log=failure_log,
            root_cause=explanation.root_cause,
            affected_files=[file_path] if file_path else []
        )

        logger.info(
            f"Explained failure: {failure_type.value}, "
            f"confidence={explanation.confidence:.2f}, "
            f"flaky={is_flaky}"
        )

        return explanation

    async def explain_batch(
        self,
        failures: List[Dict]
    ) -> List[FailureExplanation]:
        """
        Explain multiple failures in parallel.

        Args:
            failures: List of failure dicts with log, context, etc.

        Returns:
            List of FailureExplanation objects
        """
        tasks = [
            self.explain(
                failure_log=f.get("log", ""),
                code_context=f.get("context"),
                diff=f.get("diff"),
                commit_info=f.get("commit_info")
            )
            for f in failures
        ]

        return await asyncio.gather(*tasks)

    async def suggest_fix(
        self,
        explanation: FailureExplanation,
        full_code: str
    ) -> Dict[str, Any]:
        """
        Generate a code fix for a failure.

        Args:
            explanation: Failure explanation
            full_code: Full source code of the failing file

        Returns:
            Dict with fix_code, explanation, and changes
        """
        return await self.llm.suggest_fix(
            failure_type=explanation.failure_type.value,
            root_cause=explanation.root_cause,
            code_context=full_code
        )

    def _classify_failure(self, failure_log: str) -> FailureType:
        """Classify failure type from log content."""
        log_lower = failure_log.lower()

        for failure_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, failure_log, re.IGNORECASE):
                    return failure_type

        return FailureType.UNKNOWN

    def _extract_location(
        self,
        failure_log: str
    ) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Extract file path, line number, and code snippet from log."""
        # Common patterns for file:line references
        location_patterns = [
            # Python: File "path", line N
            r'File "([^"]+)", line (\d+)',
            # JavaScript/TypeScript: at path:line:col
            r'at .*\(([^:]+):(\d+):\d+\)',
            # Go: path:line:col
            r'([^\s:]+\.go):(\d+):\d+',
            # Rust: --> path:line:col
            r'--> ([^:]+):(\d+):\d+',
            # Generic: path:line
            r'([^\s:]+\.[a-z]+):(\d+)',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, failure_log)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))

                # Try to extract code snippet
                code_snippet = self._extract_snippet(failure_log, file_path)

                return file_path, line_number, code_snippet

        return None, None, None

    def _extract_snippet(
        self,
        failure_log: str,
        file_path: str
    ) -> Optional[str]:
        """Extract code snippet from failure log."""
        # Look for code block patterns
        patterns = [
            # Indented code after file reference
            rf'{re.escape(file_path)}.*\n((?:\s+.+\n)+)',
            # Code with line numbers
            r'(\d+\s*\|.*(?:\n\d+\s*\|.*)*)',
            # Caret pointing to error
            r'(\s*\^+\s*\n.*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, failure_log)
            if match:
                return match.group(1).strip()

        return None

    def _detect_flakiness(
        self,
        failure_log: str,
        failure_type: FailureType,
        similar_failures: List[Dict]
    ) -> bool:
        """Detect if a failure is likely flaky."""
        # Flaky indicators in log
        flaky_indicators = [
            "flaky", "intermittent", "sporadic",
            "race condition", "timing", "eventually",
            "retry", "connection reset", "socket hang up"
        ]

        log_lower = failure_log.lower()
        if any(ind in log_lower for ind in flaky_indicators):
            return True

        # Timeout failures are often flaky
        if failure_type == FailureType.TIMEOUT:
            return True

        # Network errors are often flaky
        if failure_type == FailureType.NETWORK_ERROR:
            return True

        # Check if similar failures have been marked as flaky
        if similar_failures:
            flaky_count = sum(
                1 for f in similar_failures
                if f.get("is_flaky", False)
            )
            if flaky_count > len(similar_failures) / 2:
                return True

        return False

    def _get_doc_links(self, failure_type: FailureType) -> List[str]:
        """Get relevant documentation links for a failure type."""
        doc_links = {
            FailureType.DEPENDENCY_ERROR: [
                "https://docs.npmjs.com/cli/v8/commands/npm-ci",
                "https://pip.pypa.io/en/stable/topics/dependency-resolution/",
            ],
            FailureType.TYPE_ERROR: [
                "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
                "https://mypy.readthedocs.io/en/stable/",
            ],
            FailureType.MEMORY_ERROR: [
                "https://nodejs.org/api/cli.html#--max-old-space-sizesize-in-megabytes",
            ],
            FailureType.TIMEOUT: [
                "https://jestjs.io/docs/jest-object#jestsettimeouttimeout",
                "https://docs.pytest.org/en/stable/how-to/failures.html",
            ],
        }

        return doc_links.get(failure_type, [])

    def format_explanation(
        self,
        explanation: FailureExplanation,
        format: str = "markdown"
    ) -> str:
        """
        Format explanation for display.

        Args:
            explanation: The failure explanation
            format: Output format (markdown, plain, json)

        Returns:
            Formatted string
        """
        if format == "markdown":
            lines = [
                f"## {explanation.failure_type.value.replace('_', ' ').title()}",
                "",
                f"**Root Cause:** {explanation.root_cause}",
                "",
                explanation.explanation,
                "",
            ]

            if explanation.file_path:
                loc = f"`{explanation.file_path}`"
                if explanation.line_number:
                    loc += f" (line {explanation.line_number})"
                lines.append(f"**Location:** {loc}")
                lines.append("")

            if explanation.code_snippet:
                lines.append("**Code:**")
                lines.append("```")
                lines.append(explanation.code_snippet)
                lines.append("```")
                lines.append("")

            if explanation.fix_suggestions:
                lines.append("**Suggested Fixes:**")
                for i, fix in enumerate(explanation.fix_suggestions, 1):
                    lines.append(f"{i}. {fix}")
                lines.append("")

            if explanation.is_flaky:
                lines.append("*This failure appears to be flaky. Consider retry or investigation.*")
                lines.append("")

            lines.append(f"*Confidence: {explanation.confidence:.0%}*")

            return "\n".join(lines)

        elif format == "plain":
            lines = [
                f"Failure Type: {explanation.failure_type.value}",
                f"Root Cause: {explanation.root_cause}",
                f"Explanation: {explanation.explanation}",
            ]

            if explanation.fix_suggestions:
                lines.append("Fixes: " + "; ".join(explanation.fix_suggestions))

            return "\n".join(lines)

        else:
            import json
            return json.dumps({
                "failure_type": explanation.failure_type.value,
                "root_cause": explanation.root_cause,
                "explanation": explanation.explanation,
                "file_path": explanation.file_path,
                "line_number": explanation.line_number,
                "fix_suggestions": explanation.fix_suggestions,
                "confidence": explanation.confidence,
                "is_flaky": explanation.is_flaky,
            }, indent=2)
