"""
LLM Client for Agentic CI

Handles all LLM interactions for code understanding, failure explanation,
and fix suggestion. Supports multiple providers with fallback capability.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import json
import re
import httpx
from loguru import logger


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LM_STUDIO = "lm_studio"


class LLMClient:
    """
    Client for LLM interactions in CI analysis.
    Optimized for code understanding and structured output.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "codellama:7b",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 60.0
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider (ollama, openai, anthropic, lm_studio)
            model: Model name/identifier
            base_url: API base URL
            api_key: API key for cloud providers
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.provider = LLMProvider(provider.lower())
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(f"LLM client initialized: {self.provider.value} with {self.model}")

    async def analyze_code_changes(
        self,
        diff: str,
        file_changes: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze code changes and provide semantic understanding.

        Args:
            diff: Git diff of changes
            file_changes: Parsed file change information

        Returns:
            Dict with summary, risk_areas, affected_components, confidence
        """
        prompt = self._build_change_analysis_prompt(diff, file_changes)
        response = await self._generate(prompt)
        return self._parse_json_response(response, {
            "summary": "Unable to analyze changes",
            "change_type": "unknown",
            "risk_areas": [],
            "affected_components": [],
            "test_coverage_concerns": [],
            "confidence": 0.5
        })

    async def explain_failure(
        self,
        failure_log: str,
        failure_type: str,
        code_context: Optional[str] = None,
        diff: Optional[str] = None,
        similar_failures: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Explain a CI failure with root cause analysis.

        Args:
            failure_log: The failure output/logs
            failure_type: Classified type of failure
            code_context: Relevant code context
            diff: Recent changes that may have caused failure
            similar_failures: Similar past failures for context

        Returns:
            Dict with root_cause, explanation, fix_suggestions, confidence
        """
        prompt = self._build_failure_explanation_prompt(
            failure_log=failure_log,
            failure_type=failure_type,
            code_context=code_context,
            diff=diff,
            similar_failures=similar_failures
        )
        response = await self._generate(prompt)
        return self._parse_json_response(response, {
            "root_cause": "Unable to determine root cause",
            "explanation": "Analysis failed",
            "related_code": [],
            "fix_suggestions": [],
            "is_flaky": False,
            "confidence": 0.5
        })

    async def suggest_fix(
        self,
        failure_type: str,
        root_cause: str,
        code_context: str
    ) -> Dict[str, Any]:
        """
        Suggest code fixes for a failure.

        Args:
            failure_type: Type of failure
            root_cause: Identified root cause
            code_context: Relevant code

        Returns:
            Dict with fix_code, explanation, changes, confidence
        """
        prompt = f"""You are an expert software engineer. Based on the following failure, suggest a fix.

Failure Type: {failure_type}
Root Cause: {root_cause}

Code Context:
```
{code_context[:3000]}
```

Provide your fix suggestion in JSON format:
{{
    "fix_code": "the corrected code snippet",
    "explanation": "why this fixes the issue",
    "changes": [
        {{"file": "path/to/file", "line": 42, "original": "old code", "replacement": "new code"}}
    ],
    "confidence": 0.85
}}

JSON Response:"""

        response = await self._generate(prompt)
        return self._parse_json_response(response, {
            "fix_code": "",
            "explanation": "Unable to suggest fix",
            "changes": [],
            "confidence": 0.0
        })

    async def predict_change_impact(
        self,
        file_paths: List[str],
        description: str
    ) -> Dict[str, Any]:
        """
        Predict the impact of planned changes.

        Args:
            file_paths: Files that will be modified
            description: Natural language description of changes

        Returns:
            Dict with affected_tests, risk_areas, recommendations
        """
        prompt = f"""You are a CI/CD expert. Given the following planned changes, predict their impact.

Files to be modified:
{chr(10).join(f"- {f}" for f in file_paths)}

Description of changes:
{description}

Predict the impact in JSON format:
{{
    "affected_tests": ["list of test files/patterns that should run"],
    "risk_areas": ["areas of concern or potential breakage"],
    "estimated_duration": 120,
    "recommendations": ["suggestions for safe implementation"],
    "confidence": 0.8
}}

JSON Response:"""

        response = await self._generate(prompt)
        return self._parse_json_response(response, {
            "affected_tests": [],
            "risk_areas": [],
            "estimated_duration": 0,
            "recommendations": [],
            "confidence": 0.5
        })

    async def test_connection(self) -> bool:
        """
        Test connection to LLM provider.

        Returns:
            True if connection successful
        """
        try:
            if self.provider == LLMProvider.OLLAMA:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/api/tags",
                        timeout=5.0
                    )
                    return response.status_code == 200
            elif self.provider == LLMProvider.OPENAI:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=5.0
                    )
                    return response.status_code == 200
            elif self.provider == LLMProvider.ANTHROPIC:
                # Anthropic doesn't have a simple health endpoint
                return self.api_key is not None
            else:
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _build_change_analysis_prompt(
        self,
        diff: str,
        file_changes: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for code change analysis."""
        files_summary = ""
        if file_changes:
            files_summary = "Files Changed:\n" + "\n".join([
                f"- {f.get('path', 'unknown')}: +{f.get('additions', 0)}/-{f.get('deletions', 0)} ({f.get('change_type', 'unknown')})"
                for f in file_changes
            ]) + "\n\n"

        # Truncate diff if too long
        max_diff_length = 5000
        truncated_diff = diff[:max_diff_length]
        if len(diff) > max_diff_length:
            truncated_diff += "\n... (truncated)"

        return f"""You are a senior software engineer reviewing code changes for a CI system. Analyze the following diff and provide insights.

{files_summary}Diff:
```diff
{truncated_diff}
```

Provide your analysis in JSON format:
{{
    "summary": "Brief summary of what these changes do",
    "change_type": "feature|bugfix|refactor|test|config|documentation|dependency",
    "risk_areas": ["list of potentially risky changes"],
    "affected_components": ["list of affected system components"],
    "test_coverage_concerns": ["areas that need testing attention"],
    "confidence": 0.85
}}

JSON Response:"""

    def _build_failure_explanation_prompt(
        self,
        failure_log: str,
        failure_type: str,
        code_context: Optional[str] = None,
        diff: Optional[str] = None,
        similar_failures: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for failure explanation."""
        context_section = ""
        if code_context:
            context_section = f"\nRelevant Code:\n```\n{code_context[:2000]}\n```\n"

        diff_section = ""
        if diff:
            diff_section = f"\nRecent Changes:\n```diff\n{diff[:1500]}\n```\n"

        similar_section = ""
        if similar_failures:
            similar_section = "\nSimilar Past Failures:\n"
            for i, sf in enumerate(similar_failures[:3], 1):
                similar_section += f"{i}. {sf.get('root_cause', 'Unknown')}\n"

        # Truncate failure log if too long
        max_log_length = 3000
        truncated_log = failure_log[:max_log_length]
        if len(failure_log) > max_log_length:
            truncated_log += "\n... (truncated)"

        return f"""You are a CI/CD expert analyzing a build failure. Provide root cause analysis and actionable fixes.

Failure Type: {failure_type}

Failure Log:
```
{truncated_log}
```
{context_section}{diff_section}{similar_section}
Provide your analysis in JSON format:
{{
    "root_cause": "The specific root cause of this failure",
    "explanation": "Detailed explanation of why this happened",
    "related_code": ["file:line references to problematic code"],
    "fix_suggestions": ["List of actionable fix suggestions"],
    "is_flaky": false,
    "confidence": 0.85
}}

JSON Response:"""

    async def _generate(self, prompt: str) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: The prompt to send

        Returns:
            Generated text response
        """
        try:
            if self.provider == LLMProvider.OLLAMA:
                return await self._generate_ollama(prompt)
            elif self.provider == LLMProvider.OPENAI:
                return await self._generate_openai(prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic(prompt)
            elif self.provider == LLMProvider.LM_STUDIO:
                return await self._generate_lm_studio(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]

    async def _generate_lm_studio(self, prompt: str) -> str:
        """Generate using LM Studio (OpenAI-compatible API)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _parse_json_response(
        self,
        response: str,
        default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with fallback.

        Args:
            response: Raw LLM response
            default: Default values if parsing fails

        Returns:
            Parsed JSON dict or default values
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return default
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return default
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return default
