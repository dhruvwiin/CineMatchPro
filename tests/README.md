# CineMatchPro Test Suite

This directory contains comprehensive tests for the CineMatchPro application.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_utils_faiss.py      # Tests for utils_faiss.py (recommendation engine)
├── test_model.py            # Tests for model.py (model building)
├── test_app.py              # Tests for app.py (application logic)
├── fixtures/                # Test data and mock responses
└── integration/             # Integration tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

This will install all testing dependencies including:
- pytest
- pytest-cov (coverage reporting)
- pytest-mock (mocking utilities)
- responses (HTTP mocking)

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

This will generate:
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory

### Run Specific Test Files

```bash
# Test only utils_faiss.py
pytest tests/test_utils_faiss.py

# Test only model.py
pytest tests/test_model.py

# Test only app.py
pytest tests/test_app.py
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only model-related tests
pytest -m model

# Run all except slow tests
pytest -m "not slow"
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_utils_faiss.py::TestLoadModelData

# Run a specific test function
pytest tests/test_utils_faiss.py::TestLoadModelData::test_load_model_data_success
```

### Verbose Output

```bash
# Show more details
pytest -v

# Show even more details (print statements)
pytest -vv -s
```

## Test Coverage Goals

- **Current Coverage**: Run `pytest --cov` to see current coverage
- **Target Coverage**:
  - `utils_faiss.py`: 85%+
  - `model.py`: 80%+
  - `app.py`: 70%+ (UI logic is harder to test)

## Writing New Tests

### Using Fixtures

Fixtures are defined in `conftest.py` and are automatically available in all test files:

```python
def test_example(mock_model_data, sample_movies_df):
    # mock_model_data and sample_movies_df are automatically provided
    result = some_function(mock_model_data)
    assert result is not None
```

### Mocking External APIs

Use the `responses` library for HTTP mocking:

```python
import responses

@responses.activate
def test_api_call():
    responses.add(
        responses.GET,
        'http://api.example.com/',
        json={'key': 'value'},
        status=200
    )
    # Your test code here
```

### Testing Exceptions

```python
def test_exception_handling():
    with pytest.raises(ValueError):
        function_that_should_raise()
```

## Continuous Integration

Tests should be run automatically on every commit using GitHub Actions or similar CI/CD tools.

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Categories

### Unit Tests
- Test individual functions in isolation
- Fast execution
- Mock all external dependencies

### Integration Tests
- Test multiple components working together
- May use real FAISS indexes and model data
- Slower execution

### API Tests
- Test external API integrations
- Always mocked (never hit real APIs in tests)
- Test error handling and edge cases

## Common Issues

### Missing Test Data

If tests fail due to missing fixtures:
```bash
# Ensure you're in the project root
cd /path/to/CineMatchPro
pytest
```

### Import Errors

If you get import errors:
```bash
# Install the package in development mode
pip install -e .
```

Or add the project root to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/CineMatchPro"
```

### Slow Tests

To skip slow tests during development:
```bash
pytest -m "not slow"
```

## Coverage Reports

After running tests with coverage, open the HTML report:

```bash
# Generate coverage report
pytest --cov --cov-report=html

# Open in browser (Linux)
xdg-open htmlcov/index.html

# Open in browser (macOS)
open htmlcov/index.html

# Open in browser (Windows)
start htmlcov/index.html
```

## Best Practices

1. **Test naming**: Use descriptive names that explain what is being tested
2. **One assertion per test**: Keep tests focused on a single behavior
3. **Arrange-Act-Assert**: Structure tests clearly (setup, execute, verify)
4. **Use fixtures**: Reuse common test data via fixtures
5. **Mock external calls**: Never hit real APIs or services in tests
6. **Test edge cases**: Include tests for error conditions and boundary cases
7. **Keep tests fast**: Unit tests should run in milliseconds

## Contributing

When adding new features:
1. Write tests first (TDD) or alongside the feature
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov`
4. Aim for >80% coverage on new code
5. Update this README if adding new test categories or fixtures
