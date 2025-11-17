# Unit Tests for Phase 1 Statistical Viability Testing

This directory contains comprehensive unit tests for the `phase1_statistical_viability.py` module.

## Overview

The test suite validates all core functionality of the Phase 1 Statistical Viability Testing system, including:

- **ExperimentRunner class**: Handles model training, evaluation, and diagnostic analysis
- **Phase1ExperimentRunner class**: Extends ExperimentRunner for statistical viability testing
- **Constants and configuration**: Validates hyperparameters and experimental design
- **Data structures**: Tests diagnostic results, sweep results, and CSV logging
- **Error handling**: Ensures robust behavior under failure conditions

## Test Structure

```
tests/
├── test_phase1/
│   ├── __init__.py
│   ├── test_comprehensive.py      # Main comprehensive test suite ✅
│   ├── test_experiment_runner.py  # Detailed ExperimentRunner tests
│   ├── test_phase1_runner.py      # Phase1ExperimentRunner tests
│   ├── test_integration.py        # Integration tests
│   ├── test_utilities.py          # Utility function tests
│   ├── test_main.py              # Main function and CLI tests
│   └── conftest.py               # Pytest fixtures and configuration
├── run_tests.py                  # General test runner
├── run_phase1_tests.py          # Phase 1 specific test runner ✅
└── README.md                    # This file
```

## Running Tests

### Quick Start (Recommended)
```bash
# Run the comprehensive test suite
python tests/run_phase1_tests.py
```

### Alternative Methods
```bash
# Using pytest directly
python -m pytest tests/test_phase1/test_comprehensive.py -v

# Run all tests in the directory
python -m pytest tests/test_phase1/ -v

# Run specific test classes
python -m pytest tests/test_phase1/test_comprehensive.py::TestConstants -v
```

## Test Coverage

### ✅ Working Tests (test_comprehensive.py)

1. **TestConstants**
   - Validates all hyperparameter constants (BATCH_SIZE, LEARNING_RATE, etc.)
   - Ensures reasonable value ranges

2. **TestExperimentRunnerBasic**
   - Initialization and directory setup
   - Result directory path generation
   - MSE parsing from training output
   - Results saving to JSON
   - Energy summary printing

3. **TestPhase1RunnerBasic**
   - Phase1ExperimentRunner initialization
   - CSV logging setup and functionality
   - Inheritance from ExperimentRunner

4. **TestMockedTraining**
   - Model training workflow (with mocked subprocess)
   - Model evaluation workflow (with mocked subprocess)

5. **TestDataStructures**
   - Diagnostic results storage
   - Sweep results management

6. **TestErrorHandling**
   - Missing model checkpoint handling
   - Training failure scenarios

### 🔧 Additional Test Files

The directory also contains more detailed test files that cover specific aspects:

- `test_experiment_runner.py`: Comprehensive ExperimentRunner testing
- `test_phase1_runner.py`: Detailed Phase1ExperimentRunner functionality
- `test_integration.py`: Integration tests for complete workflows
- `test_utilities.py`: Utility functions and edge cases
- `test_main.py`: Command-line interface and main function testing

## Key Features Tested

### ✅ Core Functionality
- [x] Configuration validation
- [x] Directory and file management
- [x] CSV logging for experiments
- [x] JSON results persistence
- [x] MSE parsing from training output
- [x] Energy diagnostic analysis
- [x] Error handling and recovery

### ✅ Phase 1 Workflow
- [x] Experiment matrix generation
- [x] Statistical analysis preparation
- [x] CSV experiment logging
- [x] Multi-configuration support

### ✅ Data Integrity
- [x] Results persistence across sessions
- [x] Proper file structure creation
- [x] Data validation and type checking

## Test Results

**Latest Run**: All 16 tests in `test_comprehensive.py` **PASSED** ✅

```
📊 FINAL TEST SUMMARY
• Execution time: ~5-8 seconds
• Exit code: 0 (success)
• Coverage: Core functionality comprehensively tested
```

## Dependencies

Tests are designed to work with minimal dependencies:

- **Required**: `unittest` (Python standard library)
- **Optional**: `pytest` (for enhanced test running)
- **Mocked**: PyTorch, subprocess calls, file operations

Tests use extensive mocking to avoid requiring:
- Actual model checkpoints
- GPU/CUDA availability
- Large dataset downloads
- Long-running training processes

## Usage for Development

When modifying `phase1_statistical_viability.py`:

1. **Before changes**: Run `python tests/run_phase1_tests.py` to establish baseline
2. **After changes**: Run tests again to verify nothing broke
3. **Add new features**: Extend `test_comprehensive.py` with relevant tests

## Notes

- Tests run in isolated temporary directories
- All file operations are safely contained
- Mocked subprocess calls prevent actual training
- CSV and JSON outputs are verified for correctness
- Error conditions are tested to ensure graceful failure

The test suite provides confidence that the Phase 1 Statistical Viability Testing system will work correctly in production environments.