# Makefile for Stylistic Poetry LLM Framework

.PHONY: help setup install test lint format clean validate info

# Default target
help:
	@echo "Available targets:"
	@echo "  setup     - Set up virtual environment and install dependencies"
	@echo "  install   - Install package in development mode"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting"
	@echo "  format    - Format code"
	@echo "  clean     - Clean up generated files"
	@echo "  validate  - Validate system setup"
	@echo "  info      - Show system information"

# Set up the development environment
setup:
	python setup_env.py

# Install package in development mode
install:
	pip install -e .

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=100

# Format code
format:
	black src/ tests/ --line-length=100

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Validate system setup
validate:
	python -m src.main validate --check-deps

# Show system information
info:
	python -m src.main info

# Run all checks (format, lint, test)
check: format lint test

# Development workflow
dev: setup install validate test