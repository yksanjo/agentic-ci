"""
CI Optimizer for Agentic CI

Optimizes CI pipelines by:
- Detecting and managing flaky tests
- Optimizing test parallelization
- Suggesting pipeline improvements
- Tracking CI health metrics
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import defaultdict
from loguru import logger

from .pattern_store import PatternStore
from .llm_client import LLMClient


class OptimizationAction(str, Enum):
    """Types of optimization actions."""
    SKIP_TEST = "skip_test"
    QUARANTINE_TEST = "quarantine_test"
    PARALLELIZE = "parallelize"
    CACHE_DEPENDENCY = "cache_dependency"
    REDUCE_MATRIX = "reduce_matrix"
    MERGE_JOBS = "merge_jobs"
    SPLIT_JOB = "split_job"


@dataclass
class FlakyTest:
    """Represents a flaky test."""
    test_path: str
    failure_rate: float  # 0.0 to 1.0
    total_runs: int
    failures: int
    last_failure: datetime
    failure_patterns: List[str] = field(default_factory=list)
    quarantined: bool = False
    notes: str = ""


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""
    action: OptimizationAction
    target: str  # test path, job name, etc.
    reason: str
    expected_impact: str  # e.g., "Save ~2 minutes per run"
    confidence: float
    implementation: str = ""  # How to implement


@dataclass
class OptimizationReport:
    """Complete optimization report."""
    generated_at: datetime
    pipeline_health_score: float  # 0.0 to 1.0

    # Flaky tests
    flaky_tests: List[FlakyTest]
    quarantined_tests: List[str]

    # Performance metrics
    avg_duration_ms: int
    p95_duration_ms: int
    failure_rate: float

    # Suggestions
    suggestions: List[OptimizationSuggestion]

    # Trends
    duration_trend: str  # improving, stable, degrading
    reliability_trend: str


class CIOptimizer:
    """
    Optimizes CI pipelines for speed and reliability.
    Tracks metrics and provides actionable suggestions.
    """

    def __init__(
        self,
        pattern_store: Optional[PatternStore] = None,
        llm_client: Optional[LLMClient] = None,
        flaky_threshold: float = 0.1,
        quarantine_threshold: float = 0.3
    ):
        """
        Initialize CI optimizer.

        Args:
            pattern_store: Historical pattern storage
            llm_client: LLM client for analysis
            flaky_threshold: Failure rate to consider test flaky
            quarantine_threshold: Failure rate to quarantine test
        """
        self.pattern_store = pattern_store or PatternStore()
        self.llm = llm_client or LLMClient()
        self.flaky_threshold = flaky_threshold
        self.quarantine_threshold = quarantine_threshold

        # In-memory tracking (would be database in production)
        self.test_runs: Dict[str, List[Dict]] = defaultdict(list)
        self.job_durations: List[Dict] = []
        self.flaky_tests: Dict[str, FlakyTest] = {}
        self.quarantined_tests: Set[str] = set()

        logger.info("CIOptimizer initialized")

    async def record_test_run(
        self,
        test_path: str,
        passed: bool,
        duration_ms: int,
        error_message: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> None:
        """
        Record a test run result.

        Args:
            test_path: Path to the test
            passed: Whether the test passed
            duration_ms: Test duration in milliseconds
            error_message: Error message if failed
            run_id: Unique run identifier
        """
        run_data = {
            "timestamp": datetime.now(),
            "passed": passed,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "run_id": run_id,
        }

        self.test_runs[test_path].append(run_data)

        # Keep only last 100 runs per test
        if len(self.test_runs[test_path]) > 100:
            self.test_runs[test_path] = self.test_runs[test_path][-100:]

        # Update flaky test tracking
        await self._update_flaky_status(test_path)

    async def record_job_run(
        self,
        job_name: str,
        duration_ms: int,
        passed: bool,
        tests_run: int,
        tests_failed: int
    ) -> None:
        """
        Record a CI job run.

        Args:
            job_name: Name of the CI job
            duration_ms: Total job duration
            passed: Whether the job passed
            tests_run: Number of tests executed
            tests_failed: Number of tests that failed
        """
        self.job_durations.append({
            "timestamp": datetime.now(),
            "job_name": job_name,
            "duration_ms": duration_ms,
            "passed": passed,
            "tests_run": tests_run,
            "tests_failed": tests_failed,
        })

        # Keep only last 500 job runs
        if len(self.job_durations) > 500:
            self.job_durations = self.job_durations[-500:]

    async def generate_report(
        self,
        days: int = 7
    ) -> OptimizationReport:
        """
        Generate optimization report.

        Args:
            days: Number of days to analyze

        Returns:
            OptimizationReport with findings and suggestions
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Analyze flaky tests
        flaky_tests = list(self.flaky_tests.values())
        quarantined = list(self.quarantined_tests)

        # Calculate job metrics
        recent_jobs = [
            j for j in self.job_durations
            if j["timestamp"] > cutoff
        ]

        if recent_jobs:
            durations = [j["duration_ms"] for j in recent_jobs]
            avg_duration = int(sum(durations) / len(durations))
            sorted_durations = sorted(durations)
            p95_idx = int(len(sorted_durations) * 0.95)
            p95_duration = sorted_durations[min(p95_idx, len(sorted_durations) - 1)]

            failures = sum(1 for j in recent_jobs if not j["passed"])
            failure_rate = failures / len(recent_jobs)
        else:
            avg_duration = 0
            p95_duration = 0
            failure_rate = 0.0

        # Generate suggestions
        suggestions = await self._generate_suggestions(
            flaky_tests=flaky_tests,
            recent_jobs=recent_jobs
        )

        # Calculate health score
        health_score = self._calculate_health_score(
            failure_rate=failure_rate,
            flaky_count=len(flaky_tests),
            avg_duration=avg_duration
        )

        # Determine trends
        duration_trend = self._calculate_trend(
            [j["duration_ms"] for j in recent_jobs]
        )
        reliability_trend = self._calculate_trend(
            [1.0 if j["passed"] else 0.0 for j in recent_jobs]
        )

        return OptimizationReport(
            generated_at=datetime.now(),
            pipeline_health_score=health_score,
            flaky_tests=flaky_tests,
            quarantined_tests=quarantined,
            avg_duration_ms=avg_duration,
            p95_duration_ms=p95_duration,
            failure_rate=failure_rate,
            suggestions=suggestions,
            duration_trend=duration_trend,
            reliability_trend=reliability_trend,
        )

    async def get_flaky_tests(
        self,
        min_failure_rate: float = 0.0
    ) -> List[FlakyTest]:
        """
        Get list of flaky tests.

        Args:
            min_failure_rate: Minimum failure rate to include

        Returns:
            List of FlakyTest objects
        """
        return [
            ft for ft in self.flaky_tests.values()
            if ft.failure_rate >= min_failure_rate
        ]

    async def quarantine_test(
        self,
        test_path: str,
        reason: str = ""
    ) -> bool:
        """
        Quarantine a flaky test.

        Args:
            test_path: Path to the test
            reason: Reason for quarantine

        Returns:
            True if quarantined successfully
        """
        self.quarantined_tests.add(test_path)

        if test_path in self.flaky_tests:
            self.flaky_tests[test_path].quarantined = True
            self.flaky_tests[test_path].notes = reason

        logger.info(f"Quarantined test: {test_path} - {reason}")
        return True

    async def unquarantine_test(
        self,
        test_path: str
    ) -> bool:
        """
        Remove a test from quarantine.

        Args:
            test_path: Path to the test

        Returns:
            True if unquarantined successfully
        """
        if test_path in self.quarantined_tests:
            self.quarantined_tests.remove(test_path)

        if test_path in self.flaky_tests:
            self.flaky_tests[test_path].quarantined = False

        logger.info(f"Unquarantined test: {test_path}")
        return True

    def is_quarantined(self, test_path: str) -> bool:
        """Check if a test is quarantined."""
        return test_path in self.quarantined_tests

    async def suggest_parallelization(
        self,
        test_paths: List[str],
        target_duration_ms: int
    ) -> List[List[str]]:
        """
        Suggest test grouping for parallel execution.

        Args:
            test_paths: List of test paths
            target_duration_ms: Target duration per parallel group

        Returns:
            List of test groups for parallel execution
        """
        # Get durations for each test
        test_durations = {}
        for path in test_paths:
            if path in self.test_runs and self.test_runs[path]:
                durations = [r["duration_ms"] for r in self.test_runs[path][-10:]]
                test_durations[path] = sum(durations) / len(durations)
            else:
                test_durations[path] = 1000  # Default 1 second

        # Sort by duration (longest first)
        sorted_tests = sorted(
            test_paths,
            key=lambda p: test_durations.get(p, 1000),
            reverse=True
        )

        # Bin packing algorithm
        groups: List[List[str]] = []
        group_durations: List[int] = []

        for test in sorted_tests:
            duration = int(test_durations.get(test, 1000))

            # Find a group that can fit this test
            placed = False
            for i, group_dur in enumerate(group_durations):
                if group_dur + duration <= target_duration_ms:
                    groups[i].append(test)
                    group_durations[i] += duration
                    placed = True
                    break

            if not placed:
                groups.append([test])
                group_durations.append(duration)

        return groups

    async def _update_flaky_status(self, test_path: str) -> None:
        """Update flaky status for a test."""
        runs = self.test_runs.get(test_path, [])

        if len(runs) < 5:
            return  # Not enough data

        # Calculate failure rate from recent runs
        recent_runs = runs[-20:]  # Last 20 runs
        failures = sum(1 for r in recent_runs if not r["passed"])
        failure_rate = failures / len(recent_runs)

        if failure_rate >= self.flaky_threshold:
            # Get failure patterns
            failure_messages = [
                r.get("error_message", "")
                for r in recent_runs
                if not r["passed"] and r.get("error_message")
            ]

            # Find last failure
            last_failure = None
            for r in reversed(recent_runs):
                if not r["passed"]:
                    last_failure = r["timestamp"]
                    break

            self.flaky_tests[test_path] = FlakyTest(
                test_path=test_path,
                failure_rate=failure_rate,
                total_runs=len(recent_runs),
                failures=failures,
                last_failure=last_failure or datetime.now(),
                failure_patterns=list(set(failure_messages))[:5],
                quarantined=test_path in self.quarantined_tests,
            )

            # Auto-quarantine if threshold exceeded
            if failure_rate >= self.quarantine_threshold:
                await self.quarantine_test(
                    test_path,
                    f"Auto-quarantined: {failure_rate:.0%} failure rate"
                )

        elif test_path in self.flaky_tests:
            # No longer flaky
            del self.flaky_tests[test_path]

    async def _generate_suggestions(
        self,
        flaky_tests: List[FlakyTest],
        recent_jobs: List[Dict]
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions."""
        suggestions = []

        # Suggest quarantining highly flaky tests
        for ft in flaky_tests:
            if not ft.quarantined and ft.failure_rate >= 0.2:
                suggestions.append(OptimizationSuggestion(
                    action=OptimizationAction.QUARANTINE_TEST,
                    target=ft.test_path,
                    reason=f"Test has {ft.failure_rate:.0%} failure rate",
                    expected_impact=f"Reduce false failures by ~{ft.failures} per {ft.total_runs} runs",
                    confidence=0.9,
                    implementation=f"Add to quarantine list or mark with @flaky decorator"
                ))

        # Analyze job patterns
        if recent_jobs:
            # Find slow jobs
            avg_duration = sum(j["duration_ms"] for j in recent_jobs) / len(recent_jobs)
            slow_jobs = [
                j for j in recent_jobs
                if j["duration_ms"] > avg_duration * 1.5
            ]

            if len(slow_jobs) > len(recent_jobs) * 0.3:
                suggestions.append(OptimizationSuggestion(
                    action=OptimizationAction.PARALLELIZE,
                    target="test_suite",
                    reason=f"30%+ of runs are 50% slower than average",
                    expected_impact="Could reduce P95 duration by 20-40%",
                    confidence=0.7,
                    implementation="Split test suite into parallel jobs"
                ))

            # Check for dependency caching opportunities
            if avg_duration > 300000:  # > 5 minutes
                suggestions.append(OptimizationSuggestion(
                    action=OptimizationAction.CACHE_DEPENDENCY,
                    target="node_modules,vendor",
                    reason="Long average duration suggests dependency install overhead",
                    expected_impact="Save 1-3 minutes per run",
                    confidence=0.6,
                    implementation="Add dependency caching to CI config"
                ))

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    def _calculate_health_score(
        self,
        failure_rate: float,
        flaky_count: int,
        avg_duration: int
    ) -> float:
        """Calculate overall pipeline health score."""
        # Start with 100
        score = 1.0

        # Deduct for failure rate
        score -= failure_rate * 0.4

        # Deduct for flaky tests
        flaky_penalty = min(flaky_count * 0.02, 0.2)
        score -= flaky_penalty

        # Deduct for slow duration (> 10 minutes is concerning)
        if avg_duration > 600000:
            score -= 0.1
        elif avg_duration > 300000:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a series of values."""
        if len(values) < 5:
            return "stable"

        # Compare first half to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid
        second_half_avg = sum(values[mid:]) / (len(values) - mid)

        if first_half_avg == 0:
            return "stable"

        change = (second_half_avg - first_half_avg) / first_half_avg

        if change > 0.1:
            return "improving" if "duration" not in str(values) else "degrading"
        elif change < -0.1:
            return "degrading" if "duration" not in str(values) else "improving"
        else:
            return "stable"

    def format_report(
        self,
        report: OptimizationReport,
        format: str = "markdown"
    ) -> str:
        """
        Format optimization report for display.

        Args:
            report: The optimization report
            format: Output format (markdown, plain)

        Returns:
            Formatted string
        """
        if format == "markdown":
            lines = [
                "# CI Optimization Report",
                f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
                "",
                f"## Pipeline Health: {report.pipeline_health_score:.0%}",
                "",
                "### Metrics",
                f"- Average Duration: {report.avg_duration_ms / 1000:.1f}s",
                f"- P95 Duration: {report.p95_duration_ms / 1000:.1f}s",
                f"- Failure Rate: {report.failure_rate:.1%}",
                f"- Duration Trend: {report.duration_trend}",
                f"- Reliability Trend: {report.reliability_trend}",
                "",
            ]

            if report.flaky_tests:
                lines.append(f"### Flaky Tests ({len(report.flaky_tests)})")
                for ft in report.flaky_tests[:10]:
                    status = " (quarantined)" if ft.quarantined else ""
                    lines.append(
                        f"- `{ft.test_path}`: {ft.failure_rate:.0%} failure rate{status}"
                    )
                lines.append("")

            if report.suggestions:
                lines.append("### Suggestions")
                for i, s in enumerate(report.suggestions[:5], 1):
                    lines.append(f"{i}. **{s.action.value}** - {s.target}")
                    lines.append(f"   {s.reason}")
                    lines.append(f"   *Impact: {s.expected_impact}*")
                    lines.append("")

            return "\n".join(lines)

        else:
            lines = [
                f"CI Health: {report.pipeline_health_score:.0%}",
                f"Avg Duration: {report.avg_duration_ms / 1000:.1f}s",
                f"Failure Rate: {report.failure_rate:.1%}",
                f"Flaky Tests: {len(report.flaky_tests)}",
                f"Suggestions: {len(report.suggestions)}",
            ]
            return "\n".join(lines)
