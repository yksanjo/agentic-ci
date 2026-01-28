"""
Pattern Store for Agentic CI

Stores and retrieves learned patterns for:
- Failure signatures and root causes
- File → test mappings
- Historical test results
- Embeddings for similarity search
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
from loguru import logger


@dataclass
class FailurePattern:
    """Represents a learned failure pattern."""
    id: str
    failure_type: str
    error_signature: str
    root_cause: str
    typical_fix: str
    affected_files: List[str] = field(default_factory=list)
    occurrence_count: int = 1
    last_seen: datetime = field(default_factory=datetime.now)
    fix_success_rate: float = 0.0
    embedding: Optional[List[float]] = None


@dataclass
class TestMapping:
    """Mapping between source file and its tests."""
    source_file: str
    test_files: List[str]
    confidence: float = 0.5
    source: str = "static_analysis"  # static_analysis, historical, manual
    validated: bool = False


class PatternStore:
    """
    In-memory pattern store with optional file persistence.
    For production, replace with database storage.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_persist: bool = True
    ):
        """
        Initialize pattern store.

        Args:
            storage_path: Path to persist patterns (optional)
            auto_persist: Whether to auto-save on updates
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_persist = auto_persist

        # In-memory storage
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.test_mappings: Dict[str, TestMapping] = {}
        self.test_history: List[Dict] = []

        # Load from disk if available
        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

        logger.info(f"PatternStore initialized with {len(self.failure_patterns)} patterns")

    async def record_analysis(self, result: Any) -> None:
        """
        Record an analysis result for learning.

        Args:
            result: AnalysisResult object
        """
        # Store file → test predictions for future validation
        for file_change in getattr(result, 'files_changed', []):
            path = getattr(file_change, 'path', '')
            if path:
                key = self._hash_path(path)
                if key not in self.test_mappings:
                    self.test_mappings[key] = TestMapping(
                        source_file=path,
                        test_files=getattr(result, 'predicted_tests', [])[:5],
                        confidence=0.5,
                        source="prediction"
                    )

        if self.auto_persist:
            await self._persist()

    async def record_test_result(
        self,
        test_run: Dict,
        predicted_tests: List[str],
        actual_failures: List[str]
    ) -> None:
        """
        Record test results to improve predictions.

        Args:
            test_run: Test run metadata
            predicted_tests: Tests we predicted to run
            actual_failures: Tests that actually failed
        """
        self.test_history.append({
            "timestamp": datetime.now().isoformat(),
            "run": test_run,
            "predicted": predicted_tests,
            "failures": actual_failures,
            "prediction_accuracy": self._calculate_accuracy(
                predicted_tests, actual_failures
            )
        })

        # Keep only last 1000 results
        if len(self.test_history) > 1000:
            self.test_history = self.test_history[-1000:]

        if self.auto_persist:
            await self._persist()

    async def record_failure_pattern(
        self,
        failure_type: str,
        error_log: str,
        root_cause: str,
        fix_applied: Optional[str] = None,
        affected_files: Optional[List[str]] = None
    ) -> str:
        """
        Record a failure pattern for future matching.

        Args:
            failure_type: Type of failure
            error_log: Error log content
            root_cause: Identified root cause
            fix_applied: Fix that was applied (if any)
            affected_files: Files related to failure

        Returns:
            Pattern ID
        """
        signature = self._create_error_signature(error_log)
        pattern_id = self._hash_signature(failure_type, signature)

        if pattern_id in self.failure_patterns:
            # Update existing pattern
            pattern = self.failure_patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_seen = datetime.now()
            if fix_applied and not pattern.typical_fix:
                pattern.typical_fix = fix_applied
        else:
            # Create new pattern
            self.failure_patterns[pattern_id] = FailurePattern(
                id=pattern_id,
                failure_type=failure_type,
                error_signature=signature,
                root_cause=root_cause,
                typical_fix=fix_applied or "",
                affected_files=affected_files or [],
            )

        if self.auto_persist:
            await self._persist()

        return pattern_id

    async def find_similar_failures(
        self,
        failure_log: str,
        failure_type: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find similar past failures for context.

        Args:
            failure_log: Current failure log
            failure_type: Type of failure
            limit: Maximum results to return

        Returns:
            List of similar failure patterns
        """
        signature = self._create_error_signature(failure_log)

        # Simple similarity: match by failure type and signature overlap
        similar = []
        for pattern in self.failure_patterns.values():
            if pattern.failure_type != failure_type:
                continue

            similarity = self._calculate_signature_similarity(
                signature, pattern.error_signature
            )

            if similarity > 0.3:  # Threshold for relevance
                similar.append({
                    "pattern_id": pattern.id,
                    "root_cause": pattern.root_cause,
                    "typical_fix": pattern.typical_fix,
                    "occurrence_count": pattern.occurrence_count,
                    "similarity": similarity,
                    "affected_files": pattern.affected_files
                })

        # Sort by similarity and return top results
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]

    def get_patterns_for_files(self, file_paths: List[str]) -> List[Dict]:
        """
        Get historical failure patterns for specific files.

        Args:
            file_paths: List of file paths

        Returns:
            List of patterns affecting these files
        """
        matching_patterns = []

        for pattern in self.failure_patterns.values():
            for path in file_paths:
                if path in pattern.affected_files:
                    matching_patterns.append({
                        "pattern_id": pattern.id,
                        "failure_type": pattern.failure_type,
                        "root_cause": pattern.root_cause,
                        "occurrence_count": pattern.occurrence_count,
                        "affected_files": pattern.affected_files
                    })
                    break

        return matching_patterns

    def get_test_mapping(self, source_file: str) -> Optional[List[str]]:
        """
        Get test files mapped to a source file.

        Args:
            source_file: Source file path

        Returns:
            List of test file paths or None
        """
        key = self._hash_path(source_file)
        mapping = self.test_mappings.get(key)

        if mapping:
            return mapping.test_files
        return None

    async def get_test_history(
        self,
        repo_id: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get test history for a repository.

        Args:
            repo_id: Repository identifier
            days: Number of days to look back

        Returns:
            List of test run results
        """
        cutoff = datetime.now() - timedelta(days=days)

        return [
            h for h in self.test_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]

    def _create_error_signature(self, error_log: str) -> str:
        """
        Create a normalized error signature from log.
        Removes variable parts like line numbers, timestamps, paths.
        """
        import re

        # Remove timestamps
        signature = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '[TIME]', error_log)

        # Remove line numbers
        signature = re.sub(r'line \d+', 'line [N]', signature)
        signature = re.sub(r':\d+:', ':[N]:', signature)

        # Remove file paths (keep filename)
        signature = re.sub(r'/[^\s:]+/([^/\s:]+)', r'\1', signature)

        # Remove hex addresses
        signature = re.sub(r'0x[0-9a-fA-F]+', '[ADDR]', signature)

        # Remove UUIDs
        signature = re.sub(
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            '[UUID]',
            signature
        )

        # Truncate and normalize whitespace
        signature = ' '.join(signature.split())[:500]

        return signature

    def _hash_signature(self, failure_type: str, signature: str) -> str:
        """Create a hash ID for a failure pattern."""
        content = f"{failure_type}:{signature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _hash_path(self, path: str) -> str:
        """Create a hash ID for a file path."""
        return hashlib.sha256(path.encode()).hexdigest()[:16]

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two error signatures."""
        # Simple word overlap similarity
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_accuracy(
        self,
        predicted: List[str],
        actual: List[str]
    ) -> float:
        """Calculate prediction accuracy."""
        if not actual:
            return 1.0 if not predicted else 0.5

        predicted_set = set(predicted)
        actual_set = set(actual)

        if not predicted_set:
            return 0.0

        # How many failures did we predict?
        true_positives = len(predicted_set & actual_set)

        # Precision: of what we predicted, how many were right?
        precision = true_positives / len(predicted_set) if predicted_set else 0

        # Recall: of actual failures, how many did we catch?
        recall = true_positives / len(actual_set) if actual_set else 0

        # F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    async def _persist(self) -> None:
        """Persist patterns to disk."""
        if not self.storage_path:
            return

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Save failure patterns
            patterns_file = self.storage_path / "failure_patterns.json"
            patterns_data = {
                k: {
                    "id": v.id,
                    "failure_type": v.failure_type,
                    "error_signature": v.error_signature,
                    "root_cause": v.root_cause,
                    "typical_fix": v.typical_fix,
                    "affected_files": v.affected_files,
                    "occurrence_count": v.occurrence_count,
                    "last_seen": v.last_seen.isoformat(),
                    "fix_success_rate": v.fix_success_rate,
                }
                for k, v in self.failure_patterns.items()
            }
            patterns_file.write_text(json.dumps(patterns_data, indent=2))

            # Save test mappings
            mappings_file = self.storage_path / "test_mappings.json"
            mappings_data = {
                k: {
                    "source_file": v.source_file,
                    "test_files": v.test_files,
                    "confidence": v.confidence,
                    "source": v.source,
                    "validated": v.validated,
                }
                for k, v in self.test_mappings.items()
            }
            mappings_file.write_text(json.dumps(mappings_data, indent=2))

            logger.debug(f"Persisted {len(self.failure_patterns)} patterns")

        except Exception as e:
            logger.error(f"Failed to persist patterns: {e}")

    def _load_from_disk(self) -> None:
        """Load patterns from disk."""
        if not self.storage_path:
            return

        try:
            # Load failure patterns
            patterns_file = self.storage_path / "failure_patterns.json"
            if patterns_file.exists():
                data = json.loads(patterns_file.read_text())
                for k, v in data.items():
                    self.failure_patterns[k] = FailurePattern(
                        id=v["id"],
                        failure_type=v["failure_type"],
                        error_signature=v["error_signature"],
                        root_cause=v["root_cause"],
                        typical_fix=v["typical_fix"],
                        affected_files=v.get("affected_files", []),
                        occurrence_count=v.get("occurrence_count", 1),
                        last_seen=datetime.fromisoformat(v["last_seen"]),
                        fix_success_rate=v.get("fix_success_rate", 0.0),
                    )

            # Load test mappings
            mappings_file = self.storage_path / "test_mappings.json"
            if mappings_file.exists():
                data = json.loads(mappings_file.read_text())
                for k, v in data.items():
                    self.test_mappings[k] = TestMapping(
                        source_file=v["source_file"],
                        test_files=v["test_files"],
                        confidence=v.get("confidence", 0.5),
                        source=v.get("source", "unknown"),
                        validated=v.get("validated", False),
                    )

            logger.info(
                f"Loaded {len(self.failure_patterns)} patterns, "
                f"{len(self.test_mappings)} mappings"
            )

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
