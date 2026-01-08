.PHONY: help format lint typecheck security test test-integration coverage all check check-unit check-full full clean install-dev sync update-lock

help:
	@echo "Single Document Generator - Available targets:"
	@echo ""
	@echo "Quality checks:"
	@echo "  make format           - Run ruff format"
	@echo "  make lint             - Run ruff check with auto-fix"
	@echo "  make typecheck        - Run mypy type checking"
	@echo "  make security         - Run bandit security scanning"
	@echo "  make test             - Run pytest (unit tests only)"
	@echo "  make test-integration - Run integration tests (requires LLM)"
	@echo "  make coverage         - Run pytest with coverage report"
	@echo "  make check            - Code quality only (format, lint, security, typecheck)"
	@echo "  make check-unit       - Code quality + unit tests"
	@echo "  make check-full       - Code quality + unit tests + integration tests"
	@echo ""
	@echo "Dependency management (using uv):"
	@echo "  make install-dev  - Install all dependencies including dev group"
	@echo "  make sync         - Sync dependencies from uv.lock"
	@echo "  make update-lock  - Update uv.lock with latest compatible versions"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean build artifacts"

format:
	@echo "Running ruff format..."
	uv run ruff format src tests

lint:
	@echo "Running ruff check with auto-fix..."
	uv run ruff check --fix src tests

typecheck:
	@echo "Running mypy type checking..."
	uv run python -m mypy -p single_doc_generator

security:
	@echo "Running security checks with bandit..."
	uv run bandit -r src/ -f screen -ll

test:
	@echo "Running unit tests..."
	uv run pytest

test-integration:
	@echo "Running integration tests (requires LLM)..."
	uv run pytest -m integration -v --no-cov

coverage:
	@echo "Running pytest with coverage..."
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

all:
	@echo "ERROR: 'make all' is ambiguous. Use one of:"
	@echo "  make check      - Code quality only (format, lint, security, typecheck)"
	@echo "  make check-unit - Code quality + unit tests"
	@echo "  make check-full - Code quality + unit tests + integration tests"
	@exit 1

check: format lint security typecheck
	@echo "Code quality checks passed!"

check-unit: format lint security typecheck test
	@echo "Code quality + unit tests passed!"

check-full: format lint security typecheck test test-integration
	@echo "Code quality + all tests passed!"

full: check-full

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf coverage_html .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

install-dev:
	@echo "Installing development dependencies..."
	uv sync
	@echo "Installation complete!"

sync:
	@echo "Syncing dependencies from uv.lock..."
	uv sync

update-lock:
	@echo "Updating dependency lock file..."
	uv lock --upgrade
