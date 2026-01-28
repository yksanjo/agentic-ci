"""
Change Analyzer for Agentic CI

Analyzes code changes to understand:
- What changed (semantic understanding)
- Why it changed (commit context)
- What might break (risk assessment)
- What tests to run (intelligent selection)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import asyncio
from loguru import logger

from .llm_client import LLMClient
from .risk_scorer import RiskScorer
from .pattern_store import PatternStore


class ChangeType(str, Enum):
    """Types of code changes."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class FileChange:
    """Represents a single file change."""
    path: str
    change_type: str  # added, modified, deleted, renamed
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    language: str = ""
    functions_modified: List[str] = field(default_factory=list)
    classes_modified: List[str] = field(default_factory=list)
    imports_changed: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of change analysis."""
    # Basic info
    commit_sha: str
    branch: str
    author: str

    # Changes
    files_changed: List[FileChange]
    total_additions: int
    total_deletions: int

    # Semantic analysis
    change_type: ChangeType
    summary: str
    affected_components: List[str]

    # Risk assessment
    risk_score: float
    risk_level: str  # low, medium, high, critical
    risk_factors: List[str]

    # Test recommendations
    predicted_tests: List[str]
    test_confidence: float

    # Additional context
    similar_past_changes: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ChangeAnalyzer:
    """
    Analyzes code changes with LLM-powered semantic understanding.
    Core orchestrator for the Agentic CI system.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        risk_scorer: Optional[RiskScorer] = None,
        pattern_store: Optional[PatternStore] = None,
        test_patterns: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the change analyzer.

        Args:
            llm_client: LLM client for semantic analysis
            risk_scorer: Risk scoring component
            pattern_store: Historical pattern storage
            test_patterns: Mapping of file patterns to test patterns
        """
        self.llm = llm_client or LLMClient()
        self.risk_scorer = risk_scorer or RiskScorer()
        self.pattern_store = pattern_store or PatternStore()

        # Default test pattern mappings
        self.test_patterns = test_patterns or {
            "src/(.+)\\.py$": ["tests/test_\\1.py", "tests/\\1_test.py"],
            "src/(.+)\\.ts$": ["__tests__/\\1.test.ts", "\\1.spec.ts"],
            "src/(.+)\\.js$": ["__tests__/\\1.test.js", "\\1.spec.js"],
            "lib/(.+)\\.rb$": ["spec/\\1_spec.rb", "test/\\1_test.rb"],
            "(.+)\\.go$": ["\\1_test.go"],
        }

        logger.info("ChangeAnalyzer initialized")

    async def analyze(
        self,
        diff: str,
        commit_info: Optional[Dict] = None,
        repo_context: Optional[Dict] = None
    ) -> AnalysisResult:
        """
        Analyze code changes comprehensively.

        Args:
            diff: Git diff of changes
            commit_info: Commit metadata (sha, message, author, branch)
            repo_context: Repository context (structure, conventions)

        Returns:
            AnalysisResult with full analysis
        """
        commit_info = commit_info or {}
        repo_context = repo_context or {}

        # Parse the diff into structured file changes
        file_changes = self._parse_diff(diff)

        # Run analysis tasks in parallel
        llm_task = self.llm.analyze_code_changes(
            diff=diff,
            file_changes=[self._file_change_to_dict(f) for f in file_changes]
        )

        # Get historical patterns for changed files
        file_paths = [f.path for f in file_changes]
        historical_patterns = self.pattern_store.get_patterns_for_files(file_paths)

        # Wait for LLM analysis
        llm_analysis = await llm_task

        # Calculate risk score
        risk_score = self.risk_scorer.score(
            file_changes=file_changes,
            llm_analysis=llm_analysis,
            historical_patterns=historical_patterns
        )

        # Get risk explanation
        risk_explanation = self.risk_scorer.explain_risk(
            file_changes=file_changes,
            llm_analysis=llm_analysis
        )

        # Predict tests to run
        predicted_tests, test_confidence = await self._predict_tests(
            file_changes=file_changes,
            llm_analysis=llm_analysis
        )

        # Build result
        result = AnalysisResult(
            commit_sha=commit_info.get("sha", "unknown"),
            branch=commit_info.get("branch", "unknown"),
            author=commit_info.get("author", "unknown"),
            files_changed=file_changes,
            total_additions=sum(f.additions for f in file_changes),
            total_deletions=sum(f.deletions for f in file_changes),
            change_type=self._determine_change_type(llm_analysis),
            summary=llm_analysis.get("summary", "Unable to analyze"),
            affected_components=llm_analysis.get("affected_components", []),
            risk_score=risk_score,
            risk_level=risk_explanation["level"],
            risk_factors=llm_analysis.get("risk_areas", []),
            predicted_tests=predicted_tests,
            test_confidence=test_confidence,
            similar_past_changes=[],
            recommendations=risk_explanation.get("recommendations", [])
        )

        # Record for learning
        await self.pattern_store.record_analysis(result)

        logger.info(
            f"Analysis complete: {len(file_changes)} files, "
            f"risk={risk_score:.2f} ({result.risk_level}), "
            f"{len(predicted_tests)} tests predicted"
        )

        return result

    async def analyze_pr(
        self,
        pr_diff: str,
        pr_info: Dict,
        commits: List[Dict]
    ) -> AnalysisResult:
        """
        Analyze a pull request with all its commits.

        Args:
            pr_diff: Combined diff for the PR
            pr_info: PR metadata (number, title, author, base, head)
            commits: List of commit info dicts

        Returns:
            AnalysisResult for the entire PR
        """
        # Analyze the combined diff
        result = await self.analyze(
            diff=pr_diff,
            commit_info={
                "sha": pr_info.get("head_sha", "unknown"),
                "branch": pr_info.get("head_branch", "unknown"),
                "author": pr_info.get("author", "unknown"),
            }
        )

        # Enhance with PR-specific info
        result.recommendations.insert(
            0,
            f"PR #{pr_info.get('number', '?')}: {pr_info.get('title', 'No title')}"
        )

        return result

    def _parse_diff(self, diff: str) -> List[FileChange]:
        """Parse git diff into structured file changes."""
        file_changes = []
        current_file = None
        current_patch_lines = []

        lines = diff.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # New file header
            if line.startswith('diff --git'):
                # Save previous file
                if current_file:
                    current_file.patch = '\n'.join(current_patch_lines)
                    self._extract_code_elements(current_file)
                    file_changes.append(current_file)

                # Parse file path
                match = re.search(r'diff --git a/(.+) b/(.+)', line)
                if match:
                    path = match.group(2)
                    current_file = FileChange(
                        path=path,
                        change_type="modified",
                        language=self._detect_language(path)
                    )
                    current_patch_lines = []

            # File status
            elif line.startswith('new file'):
                if current_file:
                    current_file.change_type = "added"
            elif line.startswith('deleted file'):
                if current_file:
                    current_file.change_type = "deleted"
            elif line.startswith('rename from'):
                if current_file:
                    current_file.change_type = "renamed"

            # Count additions/deletions
            elif line.startswith('+') and not line.startswith('+++'):
                if current_file:
                    current_file.additions += 1
                    current_patch_lines.append(line)
            elif line.startswith('-') and not line.startswith('---'):
                if current_file:
                    current_file.deletions += 1
                    current_patch_lines.append(line)
            elif line.startswith(' ') or line.startswith('@@'):
                current_patch_lines.append(line)

            i += 1

        # Don't forget the last file
        if current_file:
            current_file.patch = '\n'.join(current_patch_lines)
            self._extract_code_elements(current_file)
            file_changes.append(current_file)

        return file_changes

    def _extract_code_elements(self, file_change: FileChange) -> None:
        """Extract functions, classes, and imports from patch."""
        patch = file_change.patch
        language = file_change.language

        # Python patterns
        if language == "python":
            # Functions
            functions = re.findall(r'^[+-]\s*def\s+(\w+)', patch, re.MULTILINE)
            file_change.functions_modified = list(set(functions))

            # Classes
            classes = re.findall(r'^[+-]\s*class\s+(\w+)', patch, re.MULTILINE)
            file_change.classes_modified = list(set(classes))

            # Imports
            imports = re.findall(r'^[+-]\s*(?:from|import)\s+(\S+)', patch, re.MULTILINE)
            file_change.imports_changed = list(set(imports))

        # JavaScript/TypeScript patterns
        elif language in ("javascript", "typescript"):
            # Functions
            functions = re.findall(
                r'^[+-]\s*(?:function\s+(\w+)|const\s+(\w+)\s*=.*=>|(\w+)\s*\([^)]*\)\s*\{)',
                patch, re.MULTILINE
            )
            file_change.functions_modified = list(set(
                f for group in functions for f in group if f
            ))

            # Classes
            classes = re.findall(r'^[+-]\s*class\s+(\w+)', patch, re.MULTILINE)
            file_change.classes_modified = list(set(classes))

            # Imports
            imports = re.findall(r'^[+-]\s*import\s+.*from\s+[\'"]([^\'"]+)', patch, re.MULTILINE)
            file_change.imports_changed = list(set(imports))

        # Go patterns
        elif language == "go":
            functions = re.findall(r'^[+-]\s*func\s+(?:\([^)]+\)\s+)?(\w+)', patch, re.MULTILINE)
            file_change.functions_modified = list(set(functions))

            # Structs as classes
            structs = re.findall(r'^[+-]\s*type\s+(\w+)\s+struct', patch, re.MULTILINE)
            file_change.classes_modified = list(set(structs))

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
            '.java': 'java',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.scala': 'scala',
            '.sql': 'sql',
            '.sh': 'bash',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.md': 'markdown',
        }

        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang
        return 'unknown'

    def _determine_change_type(self, llm_analysis: Dict) -> ChangeType:
        """Determine the overall change type."""
        type_str = llm_analysis.get("change_type", "unknown").lower()

        type_map = {
            "feature": ChangeType.FEATURE,
            "bugfix": ChangeType.BUGFIX,
            "fix": ChangeType.BUGFIX,
            "refactor": ChangeType.REFACTOR,
            "test": ChangeType.TEST,
            "config": ChangeType.CONFIG,
            "configuration": ChangeType.CONFIG,
            "documentation": ChangeType.DOCUMENTATION,
            "docs": ChangeType.DOCUMENTATION,
            "dependency": ChangeType.DEPENDENCY,
            "dependencies": ChangeType.DEPENDENCY,
        }

        return type_map.get(type_str, ChangeType.UNKNOWN)

    async def _predict_tests(
        self,
        file_changes: List[FileChange],
        llm_analysis: Dict
    ) -> Tuple[List[str], float]:
        """Predict which tests should run for these changes."""
        predicted_tests = set()
        confidence_scores = []

        for file_change in file_changes:
            # Skip test files themselves
            if self._is_test_file(file_change.path):
                predicted_tests.add(file_change.path)
                confidence_scores.append(1.0)
                continue

            # Try pattern-based mapping
            for src_pattern, test_patterns in self.test_patterns.items():
                match = re.match(src_pattern, file_change.path)
                if match:
                    for test_pattern in test_patterns:
                        try:
                            test_path = match.expand(test_pattern)
                            predicted_tests.add(test_path)
                            confidence_scores.append(0.8)
                        except Exception:
                            pass

            # Check pattern store for learned mappings
            stored_tests = self.pattern_store.get_test_mapping(file_change.path)
            if stored_tests:
                predicted_tests.update(stored_tests)
                confidence_scores.append(0.7)

        # Add LLM-suggested tests
        llm_tests = llm_analysis.get("test_coverage_concerns", [])
        for concern in llm_tests:
            if "/" in concern or concern.endswith(".py") or concern.endswith(".ts"):
                predicted_tests.add(concern)
                confidence_scores.append(0.6)

        # Calculate average confidence
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.5
        )

        return list(predicted_tests), avg_confidence

    def _is_test_file(self, path: str) -> bool:
        """Check if a file is a test file."""
        test_indicators = [
            '/test_', '/_test.', '/tests/', '/__tests__/',
            '_test.py', '_test.go', '_test.js', '_test.ts',
            '.test.js', '.test.ts', '.spec.js', '.spec.ts',
            '_spec.rb', '/spec/'
        ]
        return any(ind in path for ind in test_indicators)

    def _file_change_to_dict(self, fc: FileChange) -> Dict:
        """Convert FileChange to dict for LLM."""
        return {
            "path": fc.path,
            "change_type": fc.change_type,
            "additions": fc.additions,
            "deletions": fc.deletions,
            "language": fc.language,
            "functions_modified": fc.functions_modified,
            "classes_modified": fc.classes_modified,
        }
