.PHONY: help install install-dev install-wsd lint format test test-wsd check all clean verify-gold eval-wsd-gold

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install base dependencies"
	@echo "  make install-dev  - Install dev dependencies (pytest, black, ruff)"
	@echo "  make install-wsd  - Install WSD dependencies (sentence-transformers)"
	@echo "  make install-all  - Install all dependencies"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run ruff linter"
	@echo "  make test         - Run all tests"
	@echo "  make test-wsd     - Run only WSD tests"
	@echo "  make check        - Run format + lint + test"
	@echo "  make verify-gold  - Verify gold test set checksum"
	@echo "  make eval-wsd-gold - Evaluate WSD on gold dataset"
	@echo "  make clean        - Remove cache files"

# Installation
install:
	uv sync

install-dev:
	uv sync --extra dev

install-wsd:
	uv sync --extra wsd

install-all:
	uv sync --extra dev --extra wsd

# Formatting
format:
	uv run isort src tests scripts
	uv run black src tests scripts

# Linting
lint:
	uv run ruff check src tests scripts

lint-fix:
	uv run ruff check --fix src tests scripts

# Testing
test:
	uv run pytest tests/ -v

test-wsd:
	uv run pytest tests/test_wsd*.py -v

test-cov:
	uv run pytest tests/ -v --cov=src/eng_words --cov-report=term-missing

# Combined checks
check: format lint test
	@echo "✓ All checks passed!"

# Quick check (no tests)
quick-check: format lint
	@echo "✓ Format and lint passed!"

# Verify gold dataset checksum
verify-gold:
	uv run python scripts/verify_gold_checksum.py

# Evaluate WSD on gold dataset
eval-wsd-gold:
	uv run python scripts/eval_wsd_on_gold.py

eval-wsd-gold-quick:
	uv run python scripts/eval_wsd_on_gold.py --limit 100

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned cache files"

