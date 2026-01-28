"""
Test Predictor for Agentic CI

Intelligently predicts which tests to run based on:
- Code change analysis
- Historical test-file correlations
- Dependency graph analysis
- LLM semantic understanding
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import asyncio
from pathlib import Path
from loguru import logger

from .llm_client import LLMClient
from .pattern_store import PatternStore


class TestPriority(str, Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"  # Must run, directly affected
    HIGH = "high"          # Strongly related
    MEDIUM = "medium"      # Potentially affected
    LOW = "low"            # Run if time permits


@dataclass
class TestPrediction:
    """Prediction for a single test."""
    test_path: str
    priority: TestPriority
    confidence: float
    reason: str
    related_files: List[str] = field(default_factory=list)
    estimated_duration_ms: int = 0
    is_flaky: bool = False
    last_failure: Optional[str] = None


@dataclass
class PredictionResult:
    """Complete test prediction result."""
    tests: List[TestPrediction]
    total_estimated_duration_ms: int
    coverage_confidence: float
    skipped_tests: List[str] = field(default_factory=list)
    skip_reasons: Dict[str, str] = field(default_factory=dict)


class TestPredictor:
    """
    Predicts which tests to run for code changes.
    Uses multiple signals for accurate, minimal test selection.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pattern_store: Optional[PatternStore] = None,
        test_directory: str = "tests",
        source_directory: str = "src"
    ):
        """
        Initialize test predictor.

        Args:
            llm_client: LLM client for semantic analysis
            pattern_store: Historical pattern storage
            test_directory: Default test directory
            source_directory: Default source directory
        """
        self.llm = llm_client or LLMClient()
        self.pattern_store = pattern_store or PatternStore()
        self.test_directory = test_directory
        self.source_directory = source_directory

        # Test naming conventions by language
        self.test_conventions = {
            "python": {
                "patterns": ["test_{name}.py", "{name}_test.py", "tests/test_{name}.py"],
                "markers": ["@pytest.mark", "def test_", "class Test"],
            },
            "javascript": {
                "patterns": ["{name}.test.js", "{name}.spec.js", "__tests__/{name}.test.js"],
                "markers": ["describe(", "it(", "test("],
            },
            "typescript": {
                "patterns": ["{name}.test.ts", "{name}.spec.ts", "__tests__/{name}.test.ts"],
                "markers": ["describe(", "it(", "test("],
            },
            "go": {
                "patterns": ["{name}_test.go"],
                "markers": ["func Test"],
            },
            "rust": {
                "patterns": ["{name}_test.rs", "tests/{name}.rs"],
                "markers": ["#[test]", "#[cfg(test)]"],
            },
            "ruby": {
                "patterns": ["{name}_spec.rb", "spec/{name}_spec.rb", "{name}_test.rb"],
                "markers": ["describe ", "it ", "RSpec.describe"],
            },
        }

        # Import/dependency patterns
        self.import_patterns = {
            "python": r"^(?:from|import)\s+([\w.]+)",
            "javascript": r"(?:import|require)\s*\(?['\"]([^'\"]+)",
            "typescript": r"(?:import|require)\s*\(?['\"]([^'\"]+)",
            "go": r"import\s+\"([^\"]+)\"",
            "rust": r"use\s+([\w:]+)",
        }

        logger.info("TestPredictor initialized")

    async def predict(
        self,
        changed_files: List[Dict],
        available_tests: Optional[List[str]] = None,
        max_duration_ms: Optional[int] = None,
        include_flaky: bool = True
    ) -> PredictionResult:
        """
        Predict tests to run for changed files.

        Args:
            changed_files: List of file change dicts with path, additions, etc.
            available_tests: List of available test files (auto-discovered if None)
            max_duration_ms: Maximum total test duration budget
            include_flaky: Whether to include known flaky tests

        Returns:
            PredictionResult with prioritized test list
        """
        predictions: List[TestPrediction] = []
        seen_tests: Set[str] = set()

        # Process each changed file
        for file_change in changed_files:
            file_path = file_change.get("path", "")

            # Skip if it's already a test file
            if self._is_test_file(file_path):
                if file_path not in seen_tests:
                    predictions.append(TestPrediction(
                        test_path=file_path,
                        priority=TestPriority.CRITICAL,
                        confidence=1.0,
                        reason="Directly modified test file",
                        related_files=[file_path],
                    ))
                    seen_tests.add(file_path)
                continue

            # Find related tests using multiple strategies
            related_tests = await self._find_related_tests(file_change)

            for test_info in related_tests:
                test_path = test_info["path"]
                if test_path in seen_tests:
                    continue

                predictions.append(TestPrediction(
                    test_path=test_path,
                    priority=self._determine_priority(test_info),
                    confidence=test_info.get("confidence", 0.5),
                    reason=test_info.get("reason", "Related to changed file"),
                    related_files=[file_path],
                    estimated_duration_ms=test_info.get("duration_ms", 1000),
                    is_flaky=test_info.get("is_flaky", False),
                ))
                seen_tests.add(test_path)

        # Filter flaky tests if requested
        if not include_flaky:
            predictions = [p for p in predictions if not p.is_flaky]

        # Sort by priority and confidence
        predictions.sort(
            key=lambda p: (
                self._priority_order(p.priority),
                -p.confidence
            )
        )

        # Apply duration budget if specified
        skipped_tests = []
        skip_reasons = {}

        if max_duration_ms:
            filtered_predictions = []
            total_duration = 0

            for pred in predictions:
                if total_duration + pred.estimated_duration_ms <= max_duration_ms:
                    filtered_predictions.append(pred)
                    total_duration += pred.estimated_duration_ms
                elif pred.priority == TestPriority.CRITICAL:
                    # Always include critical tests
                    filtered_predictions.append(pred)
                    total_duration += pred.estimated_duration_ms
                else:
                    skipped_tests.append(pred.test_path)
                    skip_reasons[pred.test_path] = "Duration budget exceeded"

            predictions = filtered_predictions

        # Calculate totals
        total_duration = sum(p.estimated_duration_ms for p in predictions)
        avg_confidence = (
            sum(p.confidence for p in predictions) / len(predictions)
            if predictions else 0.0
        )

        logger.info(
            f"Predicted {len(predictions)} tests, "
            f"estimated duration: {total_duration}ms, "
            f"confidence: {avg_confidence:.2f}"
        )

        return PredictionResult(
            tests=predictions,
            total_estimated_duration_ms=total_duration,
            coverage_confidence=avg_confidence,
            skipped_tests=skipped_tests,
            skip_reasons=skip_reasons,
        )

    async def predict_impact(
        self,
        planned_changes: List[str],
        change_description: str
    ) -> Dict[str, Any]:
        """
        Predict test impact for planned changes (pre-commit).

        Args:
            planned_changes: List of files planned to change
            change_description: Natural language description

        Returns:
            Dict with predicted tests and estimated impact
        """
        # Use LLM to predict impact
        impact = await self.llm.predict_change_impact(
            file_paths=planned_changes,
            description=change_description
        )

        # Enhance with pattern store knowledge
        for file_path in planned_changes:
            stored_tests = self.pattern_store.get_test_mapping(file_path)
            if stored_tests:
                impact["affected_tests"].extend(stored_tests)

        # Deduplicate
        impact["affected_tests"] = list(set(impact["affected_tests"]))

        return impact

    async def _find_related_tests(self, file_change: Dict) -> List[Dict]:
        """Find tests related to a changed file."""
        file_path = file_change.get("path", "")
        related_tests = []

        # Strategy 1: Convention-based mapping
        convention_tests = self._find_by_convention(file_path)
        related_tests.extend(convention_tests)

        # Strategy 2: Pattern store lookup
        stored_tests = self.pattern_store.get_test_mapping(file_path)
        if stored_tests:
            for test_path in stored_tests:
                related_tests.append({
                    "path": test_path,
                    "confidence": 0.8,
                    "reason": "Historical correlation",
                    "source": "pattern_store",
                })

        # Strategy 3: Import/dependency analysis
        if file_change.get("imports_changed"):
            import_tests = self._find_by_imports(
                file_path,
                file_change["imports_changed"]
            )
            related_tests.extend(import_tests)

        # Strategy 4: Function/class name matching
        functions = file_change.get("functions_modified", [])
        classes = file_change.get("classes_modified", [])

        if functions or classes:
            name_tests = self._find_by_names(functions + classes)
            related_tests.extend(name_tests)

        # Deduplicate by path, keeping highest confidence
        seen = {}
        for test in related_tests:
            path = test["path"]
            if path not in seen or test["confidence"] > seen[path]["confidence"]:
                seen[path] = test

        return list(seen.values())

    def _find_by_convention(self, file_path: str) -> List[Dict]:
        """Find tests using naming conventions."""
        tests = []

        # Detect language
        language = self._detect_language(file_path)
        if language not in self.test_conventions:
            return tests

        conventions = self.test_conventions[language]

        # Extract file name without extension
        path_obj = Path(file_path)
        name = path_obj.stem
        parent = str(path_obj.parent)

        for pattern in conventions["patterns"]:
            test_name = pattern.format(name=name)

            # Try different base paths
            possible_paths = [
                test_name,
                f"{self.test_directory}/{test_name}",
                f"{parent}/{test_name}",
                f"{parent}/../tests/{test_name}",
            ]

            for test_path in possible_paths:
                # Normalize path
                test_path = str(Path(test_path).as_posix())
                tests.append({
                    "path": test_path,
                    "confidence": 0.7,
                    "reason": f"Convention: {pattern}",
                    "source": "convention",
                })

        return tests

    def _find_by_imports(
        self,
        file_path: str,
        imports: List[str]
    ) -> List[Dict]:
        """Find tests that might test imported modules."""
        tests = []

        for imp in imports:
            # Convert import to potential test path
            imp_path = imp.replace(".", "/")

            # Look for tests of the imported module
            potential_tests = [
                f"tests/test_{imp_path}.py",
                f"tests/{imp_path}_test.py",
                f"__tests__/{imp_path}.test.ts",
            ]

            for test_path in potential_tests:
                tests.append({
                    "path": test_path,
                    "confidence": 0.5,
                    "reason": f"Tests imported module: {imp}",
                    "source": "import_analysis",
                })

        return tests

    def _find_by_names(self, names: List[str]) -> List[Dict]:
        """Find tests by function/class names."""
        tests = []

        for name in names:
            # Convert CamelCase to snake_case for test matching
            snake_name = self._to_snake_case(name)

            potential_tests = [
                f"tests/test_{snake_name}.py",
                f"tests/{snake_name}_test.py",
                f"__tests__/{name}.test.ts",
            ]

            for test_path in potential_tests:
                tests.append({
                    "path": test_path,
                    "confidence": 0.4,
                    "reason": f"Tests function/class: {name}",
                    "source": "name_matching",
                })

        return tests

    def _determine_priority(self, test_info: Dict) -> TestPriority:
        """Determine test priority based on confidence and source."""
        confidence = test_info.get("confidence", 0.5)
        source = test_info.get("source", "")

        if confidence >= 0.9:
            return TestPriority.CRITICAL
        elif confidence >= 0.7 or source == "pattern_store":
            return TestPriority.HIGH
        elif confidence >= 0.5:
            return TestPriority.MEDIUM
        else:
            return TestPriority.LOW

    def _priority_order(self, priority: TestPriority) -> int:
        """Convert priority to sort order (lower = higher priority)."""
        order = {
            TestPriority.CRITICAL: 0,
            TestPriority.HIGH: 1,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 3,
        }
        return order.get(priority, 4)

    def _is_test_file(self, path: str) -> bool:
        """Check if a file is a test file."""
        test_indicators = [
            '/test_', '/_test.', '/tests/', '/__tests__/',
            '_test.py', '_test.go', '_test.js', '_test.ts',
            '.test.js', '.test.ts', '.spec.js', '.spec.ts',
            '_spec.rb', '/spec/'
        ]
        return any(ind in path for ind in test_indicators)

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file path."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
        }

        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang
        return 'unknown'

    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
