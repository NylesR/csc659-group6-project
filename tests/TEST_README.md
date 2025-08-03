# Random Forest Model Test Suite

This directory contains comprehensive tests for the Random Forest model implementation. The tests cover unit testing, integration testing, and performance validation.

## Test Files

- `test_rf_model.py` - Main test suite with unit tests for model functionality
- `test_integration.py` - Integration tests using real data
- `run_tests.py` - Test runner script with various options
- `test_config.py` - Test-specific configuration and settings
- `pytest.ini` - Pytest configuration file

## Test Coverage

### Unit Tests (`test_rf_model.py`)

The unit tests cover:

1. **Data Loading & Preprocessing**
   - CSV file loading
   - Feature extraction
   - Target variable extraction
   - Data validation

2. **Model Functionality**
   - Model initialization
   - Model training
   - Prediction generation
   - Probability prediction
   - Feature importance calculation

3. **Evaluation Metrics**
   - Accuracy, F1-score, Precision, Recall
   - Cross-validation
   - Confusion matrix
   - Out-of-bag (OOB) score

4. **Hyperparameter Tuning**
   - Grid search functionality
   - Best model selection
   - Parameter validation

5. **Data Validation**
   - Missing value detection
   - Data type validation
   - Train-test split validation

6. **Model Persistence**
   - Model saving and loading
   - Prediction consistency

7. **Error Handling**
   - Invalid data handling
   - Invalid parameter handling

### Integration Tests (`test_integration.py`)

The integration tests cover:

1. **Real Data Processing**
   - Actual dataset loading
   - Real data preprocessing
   - Data consistency checks

2. **Model Performance with Real Data**
   - Training with actual dataset
   - Performance threshold validation
   - Feature importance with real features

3. **Model Reproducibility**
   - Consistent results with same random state
   - Deterministic behavior

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Test Execution

**From the Test folder:**
```bash
cd Test
python test_rf_model.py
python test_integration.py
python run_tests.py
```

**From the project root:**
```bash
python run_tests_from_root.py
```

### Using the Test Runner

The `run_tests.py` script provides additional options:

```bash
# Run all tests with verbose output
python run_tests.py -v 2

# Run tests with coverage analysis
python run_tests.py -c

# Run only tests matching a pattern
python run_tests.py -p "test_model_training"

# Stop on first failure
python run_tests.py -f

# List all available tests
python run_tests.py --list

# Run with quiet output
python run_tests.py -v 0
```

### Using Pytest

Run tests with pytest for additional features:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run only unit tests
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Run tests in parallel
pytest -n auto
```

## Test Configuration

### Settings

The tests use the same configuration as the main model (`rf_settings` from `original/rf.py`):

- Data path: `./data/studentdras.csv`
- Target column: `Target`
- Test size: 0.2 (20%)
- Random state: 42
- Cross-validation folds: 3
- F1-score averaging: 'macro'

### Performance Thresholds

The integration tests include performance thresholds:

- Accuracy > 0.5
- F1-score > 0.3
- OOB score > 0.3

These thresholds ensure the model performs reasonably well on the real dataset.

## Test Data

### Synthetic Data

Unit tests use synthetic data to ensure:
- Fast execution
- Controlled test conditions
- No dependency on external data files

### Real Data

Integration tests use the actual dataset (`data/studentdras.csv`) to ensure:
- End-to-end functionality
- Real-world performance validation
- Data compatibility

## Test Categories

### Unit Tests
- **Fast execution** (< 1 second per test)
- **Isolated** - no external dependencies
- **Deterministic** - same results every time
- **Comprehensive** - cover all model functionality

### Integration Tests
- **Real data** - use actual dataset
- **Performance validation** - ensure model works well
- **End-to-end** - test complete workflow
- **Reproducibility** - ensure consistent results

### Performance Tests
- **Baseline validation** - ensure model beats random chance
- **Consistency checks** - ensure deterministic behavior
- **Threshold validation** - ensure acceptable performance

## Expected Test Results

### Unit Tests
All unit tests should pass with:
- ✅ 100% pass rate
- ✅ No errors or failures
- ✅ Fast execution (< 30 seconds total)

### Integration Tests
Integration tests should pass with:
- ✅ All tests pass (if data file exists)
- ✅ Performance above thresholds
- ✅ Consistent results across runs

### Coverage
Expected coverage:
- ✅ > 90% code coverage
- ✅ All critical paths tested
- ✅ Edge cases handled

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root directory
   cd /path/to/csc659-group6-project
   ```

2. **Missing Data File**
   ```bash
   # Check if data file exists
   ls data/studentdras.csv
   ```

3. **Missing Dependencies**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   ```

4. **Test Failures**
   - Check that data file exists and is readable
   - Verify all dependencies are installed
   - Ensure you're running from the correct directory

### Debug Mode

Run tests with maximum verbosity for debugging:

```bash
python run_tests.py -v 2 -f
```

This will show detailed output and stop on the first failure.

## Adding New Tests

### Unit Test Template

```python
def test_new_functionality(self):
    """Test description."""
    # Arrange
    # Set up test data and conditions
    
    # Act
    # Execute the functionality being tested
    
    # Assert
    # Verify the expected outcomes
    self.assertEqual(actual, expected)
```

### Integration Test Template

```python
def test_new_integration(self):
    """Test integration with real data."""
    # Skip if data not available
    if not os.path.exists(self.data_path):
        self.skipTest("Data file not found")
    
    # Test with real data
    # Verify performance thresholds
    self.assertGreater(performance_metric, threshold)
```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- **Fast execution** - completes in < 2 minutes
- **Deterministic** - same results every time
- **Comprehensive** - covers all critical functionality
- **Informative** - clear pass/fail status

## Contributing

When adding new functionality to the model:

1. **Add unit tests** for the new functionality
2. **Add integration tests** if it affects real data processing
3. **Update this README** if test structure changes
4. **Ensure all tests pass** before submitting

## Test Maintenance

Regular maintenance tasks:

1. **Update thresholds** if model performance changes significantly
2. **Add tests** for new features or bug fixes
3. **Review test coverage** periodically
4. **Update dependencies** as needed 