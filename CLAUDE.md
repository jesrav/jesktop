# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
- `just install` - Install dependencies and pre-commit hooks using uv
- `uv sync --all-groups` - Sync all dependencies including dev group

### Running the Application
- `just run` - Start the FastAPI development server (uses `fastapi dev app.py`)

### Testing and Linting
- `just test` - Run pytest test suite
- `just lint` - Run pre-commit hooks on all files (includes ruff, mypy, etc.)
- `uv run pytest` - Run tests directly
- `uv run pre-commit run --all-files` - Run linting directly

### Data Ingestion
- `just ingest` - Process notes from `data/notes` folder and create vector embeddings
- `uv run scripts/ingest.py --in-folder data/notes --outfile-vector-db data/vector.json` - Run ingestion manually

### Docker Operations
- `just build` - Build Docker image locally
- `just build_linux` - Build Linux AMD64 Docker image
- `just run_in_docker` - Run application in Docker container

## Development Guidelines

### Testing Requirements
- **Always create tests** for new implementations and add them to the `tests/` folder
- Use pytest as the testing framework - structure tests as functions prefixed with `test_`
- Include both unit tests and integration tests where appropriate
- Test edge cases, error conditions, and normal operation
- When implementing new features:
  1. Write tests for core functionality
  2. Test with sample data when available
  3. Verify backward compatibility
  4. Run `just test` or `uv run pytest` to ensure all tests pass
- Use `pytest.skip()` for tests that require optional data or resources

### Code Quality Requirements
- **Always run `just lint`** after implementing something and fix any errors
- The linting process includes ruff, mypy, and other code quality tools
- Address all linting errors before considering implementation complete
- Pre-commit hooks will also run these checks, but run manually first to catch issues early