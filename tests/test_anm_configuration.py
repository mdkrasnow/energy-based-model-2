#!/usr/bin/env python3
"""
Unit tests for ANM (Adversarial Negative Mining) configuration validation.

Tests ensure that GaussianDiffusion1D properly validates the coupling between
use_adversarial_corruption and supervise_energy_landscape parameters.
"""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D


def create_mock_model():
    """Create a mock model with the required interface for GaussianDiffusion1D"""
    model = Mock()
    model.inp_dim = 32  # Required attribute
    model.out_dim = 32  # Required attribute
    return model


def test_anm_without_energy_supervision_raises_error():
    """Verify that ANM requires energy supervision - should raise ValueError"""
    model = create_mock_model()
    
    with pytest.raises(ValueError, match='ANM.*requires supervise_energy_landscape'):
        GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=False
        )


def test_anm_with_energy_supervision_succeeds():
    """Verify that ANM works when energy supervision is enabled"""
    model = create_mock_model()
    
    # Should not raise any exception
    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=32,
        timesteps=10,
        use_adversarial_corruption=True,
        supervise_energy_landscape=True
    )
    
    # Verify the configuration was set correctly
    assert diffusion.use_adversarial_corruption == True
    assert diffusion.supervise_energy_landscape == True


def test_no_anm_with_any_energy_setting_succeeds():
    """Verify that when ANM is disabled, energy supervision setting doesn't matter"""
    model = create_mock_model()
    
    # Both combinations should work when use_adversarial_corruption=False
    diffusion1 = GaussianDiffusion1D(
        model=model,
        seq_length=32,
        timesteps=10,
        use_adversarial_corruption=False,
        supervise_energy_landscape=True
    )
    
    diffusion2 = GaussianDiffusion1D(
        model=model,
        seq_length=32,
        timesteps=10,
        use_adversarial_corruption=False,
        supervise_energy_landscape=False
    )
    
    # Verify configurations
    assert diffusion1.use_adversarial_corruption == False
    assert diffusion1.supervise_energy_landscape == True
    
    assert diffusion2.use_adversarial_corruption == False
    assert diffusion2.supervise_energy_landscape == False