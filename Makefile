# Makefile for Random Forest Student Dropout Prediction Model

.PHONY: help install test clean run-model setup-venv

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-venv    - Create and activate virtual environment"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  run-model     - Run the Random Forest model"
	@echo "  clean         - Clean up cache files"
	@echo "  install-dev   - Install development dependencies"

# Setup virtual environment
setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv test_env
	@echo "Virtual environment created. Activate with: source test_env/bin/activate"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest coverage black flake8

# Run tests
test:
	@echo "Running tests..."
	python tests/run_tests_from_root.py

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	python tests/run_tests_from_root.py -v 2

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	python tests/run_tests_from_root.py -c

# Run the model
run-model:
	@echo "Running Random Forest model..."
	python src/models/rf.py

# Clean up cache files
clean:
	@echo "Cleaning up cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleanup complete!"

# Full setup
setup: setup-venv install
	@echo "Setup complete! Activate environment with: source test_env/bin/activate"

# Development setup
dev-setup: setup install-dev
	@echo "Development setup complete!"

# Quick test
quick-test:
	@echo "Running quick test..."
	python tests/run_tests_from_root.py -v 1

# List all tests
list-tests:
	@echo "Listing available tests..."
	python tests/run_tests_from_root.py --list 