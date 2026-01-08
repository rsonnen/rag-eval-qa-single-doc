# Contributing

Contributions are welcome. Please follow these guidelines.

## Development Setup

```bash
git clone https://github.com/rsonnen/rag-eval-qa-single-doc.git
cd rag-eval-qa-single-doc
make install-dev
```

## Code Quality

All contributions must pass the quality gate:

```bash
make all  # format, lint, security, typecheck, test
```

Individual checks:
- `make format` - Auto-format with ruff
- `make lint` - Lint with ruff
- `make typecheck` - Type check with mypy
- `make security` - Security scan with bandit
- `make test` - Run unit tests

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make all` to verify quality
5. Submit a pull request

## Code Style

- Follow existing patterns in the codebase
- Type hints required on all functions
- Google-style docstrings
- 88 character line length (ruff default)

## Testing

- Unit tests go in `tests/unit/`
- Integration tests (real LLM calls) go in `tests/integration/` and are marked with `@pytest.mark.integration`
- Maintain 80% code coverage minimum
