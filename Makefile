# vLLM MUSA Platform Plugin - Makefile
# =====================================

.PHONY: help install dev-install pre-commit test test-cov build publish publish-test clean all

# Default target
help:
	@echo "vLLM MUSA Platform Plugin - Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    make dev-install  - Install package in development mode"
	@echo "    make install      - Install package"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make pre-commit   - Run pre-commit hooks on all files"
	@echo ""
	@echo "  Testing:"
	@echo "    make test         - Run all tests"
	@echo "    make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "  Build & Publish:"
	@echo "    make build        - Build wheel and sdist"
	@echo "    make publish      - Build and publish to PyPI"
	@echo "    make publish-test - Build and publish to TestPyPI"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean        - Remove build artifacts"
	@echo ""
	@echo "  Combined:"
	@echo "    make all          - pre-commit, test, build"

# =============================================================================
# Development
# =============================================================================

dev-install:
	pip install -e ".[dev]" --no-build-isolation -v

install:
	pip install . --no-build-isolation -v

# =============================================================================
# Code Quality (via pre-commit)
# =============================================================================

pre-commit:
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=vllm_musa --cov-report=term-missing --cov-report=html

# =============================================================================
# Build & Publish
# =============================================================================

# Build wheel and source distribution
build: clean
	python -m build

# Publish to PyPI
publish: build
	python -m twine upload --repository pypi dist/*

# Publish to TestPyPI (for testing)
publish-test: build
	python -m twine upload --repository testpypi dist/*

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf vllm_musa.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Combined Targets
# =============================================================================

all: pre-commit test build
