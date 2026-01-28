# Agentic CI

> CI that understands context, not just commands.

**Agentic CI** is an intelligent continuous integration system that uses LLMs to understand your code changes, predict which tests to run, explain failures, and optimize your CI pipeline.

Inspired by [Peter Steinberger's vision](https://steipete.com/) for the future of intelligent CI systems.

## Features

### Semantic Change Analysis
- Understands *what* changed, not just which files
- Identifies risk areas and affected components
- Calculates risk scores based on file criticality, complexity, and history

### Intelligent Test Selection
- Predicts which tests to run based on changes
- Uses multiple signals: conventions, imports, historical patterns
- Reduces CI time by running only relevant tests

### Failure Explanation
- Root cause analysis with LLM-powered understanding
- Extracts file/line references from logs
- Suggests fixes based on similar past failures
- Detects flaky tests automatically

### CI Optimization
- Tracks flaky tests and auto-quarantines problematic ones
- Suggests parallelization strategies
- Provides pipeline health metrics and trends

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) (or OpenAI/Anthropic API key)

### Installation

```bash
# Clone the repository
git clone https://github.com/yksanjo/agentic-ci.git
cd agentic-ci

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull a code-focused LLM (if using Ollama)
ollama pull codellama:7b
```

### Run the API Server

```bash
# Start the server
uvicorn api.main:app --reload --port 8080

# Or use the module directly
python -m api.main
```

Visit `http://localhost:8080/docs` for interactive API documentation.

## Usage

### Analyze Code Changes

```bash
# Get a diff
git diff HEAD~1 > changes.diff

# Analyze it
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "diff": "'"$(cat changes.diff)"'",
    "branch": "feature/new-feature",
    "author": "developer@example.com"
  }'
```

### Predict Tests to Run

```bash
curl -X POST http://localhost:8080/predict/tests \
  -H "Content-Type: application/json" \
  -d '{
    "changed_files": [
      {"path": "src/api/users.py", "additions": 50, "deletions": 10}
    ]
  }'
```

### Explain a Failure

```bash
curl -X POST http://localhost:8080/explain/failure \
  -H "Content-Type: application/json" \
  -d '{
    "failure_log": "AssertionError: Expected 5 but got 4\n  at test_calculator.py:42"
  }'
```

### Get Optimization Report

```bash
curl http://localhost:8080/optimizer/report?days=7
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# LLM settings
llm:
  provider: "ollama"  # or openai, anthropic
  model: "codellama:7b"
  base_url: "http://localhost:11434"

# Risk assessment weights
risk:
  weights:
    file_criticality: 0.30
    change_complexity: 0.25
    historical_risk: 0.20
    semantic_risk: 0.15
    dependency_risk: 0.10

# Flaky test thresholds
optimizer:
  flaky_threshold: 0.10
  quarantine_threshold: 0.30
```

### Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER=ollama
export LLM_MODEL=codellama:7b
export LLM_BASE_URL=http://localhost:11434
export AGENTIC_CI_API_KEY=your-api-key  # For OpenAI/Anthropic

# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8080

# Storage
export PATTERN_STORE_PATH=./data/patterns
```

## Architecture

```
agentic-ci/
├── agentic_ci/
│   ├── __init__.py
│   └── core/
│       ├── analyzer.py      # Change analysis orchestrator
│       ├── predictor.py     # Test prediction engine
│       ├── explainer.py     # Failure root cause analysis
│       ├── optimizer.py     # CI optimization & flaky test management
│       ├── llm_client.py    # Multi-provider LLM client
│       ├── risk_scorer.py   # Risk assessment
│       └── pattern_store.py # Historical pattern storage
├── api/
│   └── main.py              # FastAPI REST API
├── config/
│   └── config.yaml          # Configuration
├── data/                    # Persisted patterns
└── tests/                   # Test suite
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & LLM connectivity |
| `/analyze` | POST | Analyze code changes |
| `/predict/tests` | POST | Predict tests to run |
| `/explain/failure` | POST | Explain CI failure |
| `/optimizer/record` | POST | Record test result |
| `/optimizer/report` | GET | Get optimization report |
| `/optimizer/flaky` | GET | List flaky tests |
| `/optimizer/quarantine/{path}` | POST/DELETE | Manage test quarantine |

## Integration

### GitHub Actions

```yaml
name: Agentic CI

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get diff
        run: git diff ${{ github.event.before }}..${{ github.sha }} > changes.diff

      - name: Analyze changes
        run: |
          curl -X POST $AGENTIC_CI_URL/analyze \
            -H "Content-Type: application/json" \
            -d '{"diff": "'"$(cat changes.diff)"'"}'
```

### GitLab CI

```yaml
agentic-analyze:
  stage: test
  script:
    - git diff HEAD~1 > changes.diff
    - |
      curl -X POST $AGENTIC_CI_URL/analyze \
        -H "Content-Type: application/json" \
        -d '{"diff": "'"$(cat changes.diff)"'"}'
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Format code
black agentic_ci/
isort agentic_ci/

# Type checking
mypy agentic_ci/

# Lint
ruff check agentic_ci/
```

## Roadmap

- [ ] MCP (Model Context Protocol) server integration
- [ ] VS Code extension
- [ ] GitHub App for automated PR analysis
- [ ] Historical trend visualization dashboard
- [ ] Support for more LLM providers
- [ ] Kubernetes-native deployment

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- Concept inspired by [Peter Steinberger](https://steipete.com/)
- Built with [FastAPI](https://fastapi.tiangolo.com/), [Ollama](https://ollama.ai/), and [Loguru](https://github.com/Delgan/loguru)

---

**Made with love for the developer experience.**
