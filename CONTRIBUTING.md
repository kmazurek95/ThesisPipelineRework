# Contributing to Interest Group Prominence Analysis

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ThesisPipelineRework.git
   cd ThesisPipelineRework
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/kmazurek95/ThesisPipelineRework.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy

# Verify installation
python scripts/validate_data.py
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-analysis` - New features
- `fix/correct-regression-bug` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/clean-etl-pipeline` - Code refactoring

### Workflow

1. **Create a new branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:
   ```bash
   pytest tests/
   python scripts/validate_data.py
   ```

4. **Commit your changes** with clear messages:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

## Submitting Changes

### Pull Request Process

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Describe your changes** in the PR:
   - What problem does this solve?
   - How did you test it?
   - Any breaking changes?

4. **Address review feedback** if requested

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Tests pass locally
- [ ] New code includes appropriate tests
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these preferences:

- **Line length**: 100 characters max
- **Imports**: Grouped (standard library, third-party, local)
- **Docstrings**: Google style
- **Type hints**: Encouraged for public functions

```python
def load_data(file_path: Path, n_rows: int = None) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        n_rows: Number of rows to read (optional).

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    pass
```

### Code Formatting

```bash
# Format with black
black interest_group_analysis/ scripts/

# Check with flake8
flake8 interest_group_analysis/ scripts/

# Type check with mypy
mypy interest_group_analysis/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=interest_group_analysis tests/

# Run specific test file
pytest tests/test_classification.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for shared setup

```python
import pytest
from interest_group_analysis.utils.file_io import load_csv

def test_load_csv_valid_file(tmp_path):
    """Test loading a valid CSV file."""
    # Create test file
    test_file = tmp_path / "test.csv"
    test_file.write_text("col1,col2\n1,2\n3,4")

    # Test
    df = load_csv(test_file)
    assert len(df) == 2
    assert list(df.columns) == ["col1", "col2"]
```

## Documentation

### Updating Documentation

- Keep the README.md up to date with any new features
- Add docstrings to all public functions and classes
- Update data/README.md if data structure changes
- Include inline comments for complex logic

### Documentation Style

- Use Markdown for all documentation files
- Keep language clear and concise
- Include code examples where helpful
- Link to relevant files and sections

## Questions?

If you have questions about contributing:

1. Check existing [Issues](https://github.com/kmazurek95/ThesisPipelineRework/issues)
2. Open a new Issue for discussion
3. Reach out to the maintainer

Thank you for contributing!
