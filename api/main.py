"""
Agentic CI API

REST API for the intelligent CI system.
Provides endpoints for analysis, prediction, and optimization.
"""

from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from agentic_ci.core import (
    ChangeAnalyzer,
    LLMClient,
    TestPredictor,
    FailureExplainer,
    CIOptimizer,
    RiskScorer,
    PatternStore,
)


# Request/Response Models
class AnalyzeRequest(BaseModel):
    diff: str = Field(..., description="Git diff to analyze")
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    author: Optional[str] = None


class AnalyzeResponse(BaseModel):
    summary: str
    change_type: str
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    predicted_tests: List[str]
    test_confidence: float
    recommendations: List[str]
    files_changed: int
    additions: int
    deletions: int


class PredictTestsRequest(BaseModel):
    changed_files: List[Dict[str, Any]]
    max_duration_ms: Optional[int] = None
    include_flaky: bool = True


class PredictTestsResponse(BaseModel):
    tests: List[Dict[str, Any]]
    total_estimated_duration_ms: int
    coverage_confidence: float
    skipped_tests: List[str]


class ExplainFailureRequest(BaseModel):
    failure_log: str
    code_context: Optional[str] = None
    diff: Optional[str] = None


class ExplainFailureResponse(BaseModel):
    failure_type: str
    root_cause: str
    explanation: str
    file_path: Optional[str]
    line_number: Optional[int]
    fix_suggestions: List[str]
    confidence: float
    is_flaky: bool


class RecordTestRequest(BaseModel):
    test_path: str
    passed: bool
    duration_ms: int
    error_message: Optional[str] = None
    run_id: Optional[str] = None


class OptimizationReportResponse(BaseModel):
    pipeline_health_score: float
    avg_duration_ms: int
    failure_rate: float
    flaky_tests_count: int
    suggestions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_connected: bool


# Global instances
llm_client: Optional[LLMClient] = None
pattern_store: Optional[PatternStore] = None
analyzer: Optional[ChangeAnalyzer] = None
predictor: Optional[TestPredictor] = None
explainer: Optional[FailureExplainer] = None
optimizer: Optional[CIOptimizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global llm_client, pattern_store, analyzer, predictor, explainer, optimizer

    logger.info("Initializing Agentic CI...")

    # Initialize LLM client
    llm_client = LLMClient(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        model=os.getenv("LLM_MODEL", "codellama:7b"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("AGENTIC_CI_API_KEY"),
    )

    # Initialize pattern store
    pattern_store = PatternStore(
        storage_path=os.getenv("PATTERN_STORE_PATH", "./data/patterns"),
        auto_persist=True
    )

    # Initialize components
    risk_scorer = RiskScorer()
    analyzer = ChangeAnalyzer(
        llm_client=llm_client,
        risk_scorer=risk_scorer,
        pattern_store=pattern_store
    )
    predictor = TestPredictor(
        llm_client=llm_client,
        pattern_store=pattern_store
    )
    explainer = FailureExplainer(
        llm_client=llm_client,
        pattern_store=pattern_store
    )
    optimizer = CIOptimizer(
        pattern_store=pattern_store,
        llm_client=llm_client
    )

    logger.info("Agentic CI initialized successfully")

    yield

    logger.info("Shutting down Agentic CI...")


# Create FastAPI app
app = FastAPI(
    title="Agentic CI",
    description="CI that understands context, not just commands",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health & Info
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and LLM connectivity."""
    llm_connected = False
    if llm_client:
        llm_connected = await llm_client.test_connection()

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        llm_connected=llm_connected
    )


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Agentic CI",
        "description": "CI that understands context, not just commands",
        "version": "0.1.0",
        "docs": "/docs",
    }


# Analysis Endpoints
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_changes(request: AnalyzeRequest):
    """Analyze code changes with semantic understanding."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    result = await analyzer.analyze(
        diff=request.diff,
        commit_info={
            "sha": request.commit_sha or "unknown",
            "branch": request.branch or "unknown",
            "author": request.author or "unknown",
        }
    )

    return AnalyzeResponse(
        summary=result.summary,
        change_type=result.change_type.value,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        risk_factors=result.risk_factors,
        predicted_tests=result.predicted_tests,
        test_confidence=result.test_confidence,
        recommendations=result.recommendations,
        files_changed=len(result.files_changed),
        additions=result.total_additions,
        deletions=result.total_deletions,
    )


# Test Prediction Endpoints
@app.post("/predict/tests", response_model=PredictTestsResponse)
async def predict_tests(request: PredictTestsRequest):
    """Predict which tests to run for changed files."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = await predictor.predict(
        changed_files=request.changed_files,
        max_duration_ms=request.max_duration_ms,
        include_flaky=request.include_flaky
    )

    return PredictTestsResponse(
        tests=[
            {
                "path": t.test_path,
                "priority": t.priority.value,
                "confidence": t.confidence,
                "reason": t.reason,
                "estimated_duration_ms": t.estimated_duration_ms,
                "is_flaky": t.is_flaky,
            }
            for t in result.tests
        ],
        total_estimated_duration_ms=result.total_estimated_duration_ms,
        coverage_confidence=result.coverage_confidence,
        skipped_tests=result.skipped_tests,
    )


# Failure Explanation Endpoints
@app.post("/explain/failure", response_model=ExplainFailureResponse)
async def explain_failure(request: ExplainFailureRequest):
    """Explain a CI failure with root cause analysis."""
    if not explainer:
        raise HTTPException(status_code=503, detail="Explainer not initialized")

    result = await explainer.explain(
        failure_log=request.failure_log,
        code_context=request.code_context,
        diff=request.diff
    )

    return ExplainFailureResponse(
        failure_type=result.failure_type.value,
        root_cause=result.root_cause,
        explanation=result.explanation,
        file_path=result.file_path,
        line_number=result.line_number,
        fix_suggestions=result.fix_suggestions,
        confidence=result.confidence,
        is_flaky=result.is_flaky,
    )


# Optimizer Endpoints
@app.post("/optimizer/record")
async def record_test_result(
    request: RecordTestRequest,
    background_tasks: BackgroundTasks
):
    """Record a test run result for optimization."""
    if not optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    background_tasks.add_task(
        optimizer.record_test_run,
        test_path=request.test_path,
        passed=request.passed,
        duration_ms=request.duration_ms,
        error_message=request.error_message,
        run_id=request.run_id
    )

    return {"status": "recorded"}


@app.get("/optimizer/report", response_model=OptimizationReportResponse)
async def get_optimization_report(days: int = 7):
    """Get CI optimization report."""
    if not optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    report = await optimizer.generate_report(days=days)

    return OptimizationReportResponse(
        pipeline_health_score=report.pipeline_health_score,
        avg_duration_ms=report.avg_duration_ms,
        failure_rate=report.failure_rate,
        flaky_tests_count=len(report.flaky_tests),
        suggestions=[
            {
                "action": s.action.value,
                "target": s.target,
                "reason": s.reason,
                "expected_impact": s.expected_impact,
                "confidence": s.confidence,
            }
            for s in report.suggestions
        ]
    )


@app.get("/optimizer/flaky")
async def get_flaky_tests(min_failure_rate: float = 0.0):
    """Get list of flaky tests."""
    if not optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    flaky_tests = await optimizer.get_flaky_tests(min_failure_rate)

    return {
        "flaky_tests": [
            {
                "test_path": ft.test_path,
                "failure_rate": ft.failure_rate,
                "total_runs": ft.total_runs,
                "failures": ft.failures,
                "quarantined": ft.quarantined,
            }
            for ft in flaky_tests
        ]
    }


@app.post("/optimizer/quarantine/{test_path:path}")
async def quarantine_test(test_path: str, reason: str = ""):
    """Quarantine a flaky test."""
    if not optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    success = await optimizer.quarantine_test(test_path, reason)
    return {"success": success, "test_path": test_path}


@app.delete("/optimizer/quarantine/{test_path:path}")
async def unquarantine_test(test_path: str):
    """Remove a test from quarantine."""
    if not optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    success = await optimizer.unquarantine_test(test_path)
    return {"success": success, "test_path": test_path}


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        reload=os.getenv("API_DEBUG", "false").lower() == "true"
    )
