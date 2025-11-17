#!/usr/bin/env python3
"""
Configuration file for pytest
Provides shared fixtures and setup for Phase 1 tests
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for tests"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_phase1_config():
    """Fixture providing a mock Phase 1 configuration"""
    config = Mock()
    config.name = 'test_config'
    config.description = 'Test configuration for unit tests'
    config.use_anm = True
    config.anm_adversarial_steps = 10
    config.use_random_noise = False
    return config


@pytest.fixture
def mock_baseline_config():
    """Fixture providing a mock baseline configuration"""
    config = Mock()
    config.name = 'ired_baseline'
    config.description = 'Baseline IRED configuration'
    config.use_anm = False
    config.use_random_noise = False
    return config


@pytest.fixture
def sample_experiment_results():
    """Fixture providing sample experiment results"""
    return {
        'ired_baseline': [0.1234, 0.1456, 0.1123, 0.1567, 0.1345],
        'anm_aggressive': [0.0987, 0.1023, 0.0876, 0.1134, 0.0945]
    }


@pytest.fixture
def mock_torch_imports():
    """Fixture to handle PyTorch imports in tests"""
    try:
        import torch
        import numpy as np
        return {'torch': torch, 'np': np, 'available': True}
    except ImportError:
        return {'torch': None, 'np': None, 'available': False}