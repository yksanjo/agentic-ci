"""
Risk Scorer for Agentic CI

Calculates risk scores for code changes based on multiple factors:
- File criticality (core paths, config, dependencies)
- Change complexity (lines changed, functions modified)
- Historical failure patterns
- LLM semantic analysis
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class RiskFactors:
    """Individual risk factors with weights."""
    file_criticality: float = 0.0
    change_complexity: float = 0.0
    historical_risk: float = 0.0
    semantic_risk: float = 0.0
    dependency_risk: float = 0.0


class RiskScorer:
    """
    Calculates risk scores for code changes.
    Higher scores indicate higher risk of test failures.
    """

    def __init__(
        self,
        critical_paths: Optional[List[str]] = None,
        high_risk_patterns: Optional[List[str]] = None
    ):
        """
        Initialize risk scorer.

        Args:
            critical_paths: Paths that are considered high-risk
            high_risk_patterns: File patterns that are high-risk
        """
        self.critical_paths = critical_paths or [
            "src/core/",
            "src/api/",
            "config/",
            "lib/",
            "database/",
            "migrations/",
            "security/",
            "auth/",
        ]

        self.high_risk_patterns = high_risk_patterns or [
            "*.sql",
            "*.env*",
            "*config*",
            "*secret*",
            "*auth*",
            "*security*",
            "package.json",
            "requirements.txt",
            "Cargo.toml",
            "go.mod",
        ]

        # Weights for combining risk factors
        self.weights = {
            "file_criticality": 0.30,
            "change_complexity": 0.25,
            "historical_risk": 0.20,
            "semantic_risk": 0.15,
            "dependency_risk": 0.10,
        }

        logger.info("RiskScorer initialized")

    def score(
        self,
        file_changes: List[Any],
        llm_analysis: Optional[Dict] = None,
        historical_patterns: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate overall risk score for changes.

        Args:
            file_changes: List of FileChange objects
            llm_analysis: LLM semantic analysis result
            historical_patterns: Historical failure patterns for files

        Returns:
            Risk score between 0.0 and 1.0
        """
        factors = RiskFactors()

        # Calculate file criticality
        factors.file_criticality = self._calculate_file_criticality(file_changes)

        # Calculate change complexity
        factors.change_complexity = self._calculate_change_complexity(file_changes)

        # Calculate historical risk
        if historical_patterns:
            factors.historical_risk = self._calculate_historical_risk(
                file_changes, historical_patterns
            )

        # Extract semantic risk from LLM analysis
        if llm_analysis:
            factors.semantic_risk = self._calculate_semantic_risk(llm_analysis)

        # Calculate dependency risk
        factors.dependency_risk = self._calculate_dependency_risk(file_changes)

        # Combine factors with weights
        total_score = (
            self.weights["file_criticality"] * factors.file_criticality +
            self.weights["change_complexity"] * factors.change_complexity +
            self.weights["historical_risk"] * factors.historical_risk +
            self.weights["semantic_risk"] * factors.semantic_risk +
            self.weights["dependency_risk"] * factors.dependency_risk
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, total_score))

        logger.debug(
            f"Risk score: {final_score:.3f} "
            f"(crit={factors.file_criticality:.2f}, "
            f"complex={factors.change_complexity:.2f}, "
            f"hist={factors.historical_risk:.2f}, "
            f"semantic={factors.semantic_risk:.2f}, "
            f"dep={factors.dependency_risk:.2f})"
        )

        return final_score

    def _calculate_file_criticality(self, file_changes: List[Any]) -> float:
        """Calculate risk based on file paths."""
        if not file_changes:
            return 0.0

        critical_count = 0
        high_risk_count = 0

        for change in file_changes:
            path = getattr(change, 'path', change.get('path', ''))

            # Check critical paths
            if any(crit in path for crit in self.critical_paths):
                critical_count += 1

            # Check high-risk patterns
            import fnmatch
            if any(fnmatch.fnmatch(path, pattern) for pattern in self.high_risk_patterns):
                high_risk_count += 1

        total = len(file_changes)
        criticality = (critical_count * 1.0 + high_risk_count * 0.7) / total

        return min(1.0, criticality)

    def _calculate_change_complexity(self, file_changes: List[Any]) -> float:
        """Calculate risk based on change size and complexity."""
        if not file_changes:
            return 0.0

        total_additions = 0
        total_deletions = 0
        functions_modified = 0

        for change in file_changes:
            total_additions += getattr(change, 'additions', change.get('additions', 0))
            total_deletions += getattr(change, 'deletions', change.get('deletions', 0))

            funcs = getattr(change, 'functions_modified', change.get('functions_modified', []))
            functions_modified += len(funcs) if funcs else 0

        total_lines = total_additions + total_deletions

        # Scoring thresholds
        if total_lines < 10:
            size_score = 0.1
        elif total_lines < 50:
            size_score = 0.3
        elif total_lines < 200:
            size_score = 0.5
        elif total_lines < 500:
            size_score = 0.7
        else:
            size_score = 0.9

        # Adjust for number of files
        file_count = len(file_changes)
        if file_count > 10:
            size_score = min(1.0, size_score + 0.2)

        # Adjust for function modifications
        if functions_modified > 5:
            size_score = min(1.0, size_score + 0.1)

        return size_score

    def _calculate_historical_risk(
        self,
        file_changes: List[Any],
        historical_patterns: List[Dict]
    ) -> float:
        """Calculate risk based on historical failure patterns."""
        if not historical_patterns:
            return 0.0

        changed_paths = {
            getattr(c, 'path', c.get('path', ''))
            for c in file_changes
        }

        # Check how many changed files have historical failures
        risky_files = 0
        total_failure_rate = 0.0

        for pattern in historical_patterns:
            affected = pattern.get('affected_files', [])
            for path in changed_paths:
                if path in affected:
                    risky_files += 1
                    total_failure_rate += pattern.get('occurrence_count', 1) / 100

        if not changed_paths:
            return 0.0

        return min(1.0, (risky_files / len(changed_paths)) * 0.5 + total_failure_rate * 0.5)

    def _calculate_semantic_risk(self, llm_analysis: Dict) -> float:
        """Extract semantic risk from LLM analysis."""
        risk_areas = llm_analysis.get('risk_areas', [])
        confidence = llm_analysis.get('confidence', 0.5)

        # More risk areas = higher risk
        risk_count = len(risk_areas)
        if risk_count == 0:
            base_risk = 0.1
        elif risk_count <= 2:
            base_risk = 0.3
        elif risk_count <= 5:
            base_risk = 0.6
        else:
            base_risk = 0.8

        # Adjust based on change type
        change_type = llm_analysis.get('change_type', 'unknown')
        type_multipliers = {
            'bugfix': 0.7,  # Bug fixes are often well-tested
            'refactor': 1.2,  # Refactors can introduce subtle bugs
            'feature': 1.0,  # New features have medium risk
            'config': 1.1,  # Config changes can have wide impact
            'dependency': 1.3,  # Dependency changes are risky
            'test': 0.3,  # Test changes are low risk
            'documentation': 0.1,  # Docs are very low risk
        }

        multiplier = type_multipliers.get(change_type, 1.0)

        return min(1.0, base_risk * multiplier * confidence)

    def _calculate_dependency_risk(self, file_changes: List[Any]) -> float:
        """Calculate risk from dependency file changes."""
        dependency_files = [
            'package.json', 'package-lock.json', 'yarn.lock',
            'requirements.txt', 'Pipfile', 'Pipfile.lock',
            'Cargo.toml', 'Cargo.lock',
            'go.mod', 'go.sum',
            'pom.xml', 'build.gradle',
            'Gemfile', 'Gemfile.lock',
        ]

        for change in file_changes:
            path = getattr(change, 'path', change.get('path', ''))
            filename = path.split('/')[-1]

            if filename in dependency_files:
                return 0.8  # High risk for any dependency change

        return 0.0

    def explain_risk(
        self,
        file_changes: List[Any],
        llm_analysis: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Provide human-readable risk explanation.

        Returns:
            Dict with risk_level, factors, and recommendations
        """
        score = self.score(file_changes, llm_analysis)

        if score < 0.3:
            level = "low"
            recommendations = ["Standard test suite should be sufficient"]
        elif score < 0.6:
            level = "medium"
            recommendations = [
                "Run relevant unit tests",
                "Consider integration tests for affected components"
            ]
        elif score < 0.8:
            level = "high"
            recommendations = [
                "Run full test suite",
                "Manual review recommended",
                "Consider staging deployment test"
            ]
        else:
            level = "critical"
            recommendations = [
                "Full test suite required",
                "Mandatory code review",
                "Staging deployment required",
                "Consider canary release"
            ]

        return {
            "score": score,
            "level": level,
            "recommendations": recommendations,
            "factors": {
                "files_changed": len(file_changes),
                "critical_paths_affected": sum(
                    1 for c in file_changes
                    if any(crit in getattr(c, 'path', c.get('path', ''))
                           for crit in self.critical_paths)
                ),
                "risk_areas": llm_analysis.get('risk_areas', []) if llm_analysis else []
            }
        }
