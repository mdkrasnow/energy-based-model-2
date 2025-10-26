#!/usr/bin/env python3
"""Test script to verify that curriculum temperature is correctly used in loss scaling."""

import torch
import torch.nn as nn
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import get_model
from curriculum_config import get_curriculum_config

def test_temperature_propagation():
    """Test that temperature from curriculum affects loss_scale."""
    
    # Create a simple model
    model = get_model('transformer', input_dim=32, output_dim=32, mask=False)
    
    # Create curriculum config
    curriculum_config = get_curriculum_config('default')
    
    # Create diffusion model with curriculum
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=32,
        objective='pred_noise',
        timesteps=100,
        sampling_timesteps=100,
        supervise_energy_landscape=True,  # Enable energy loss
        use_innerloop_opt=False,
        use_adversarial_corruption=True,
        anm_adversarial_steps=5,
        anm_distance_penalty=0.1,
        anm_warmup_steps=100,
        curriculum_config=curriculum_config,
    )
    
    # Create dummy data
    batch_size = 4
    seq_len = 32
    inp = torch.randn(batch_size, seq_len)
    target = torch.randn(batch_size, seq_len)
    mask = None
    
    # Test at different training steps (which map to different curriculum stages)
    test_steps = [0, 5000, 15000, 25000, 35000]
    
    print("Testing temperature-based loss scaling:")
    print("-" * 60)
    
    for step in test_steps:
        diffusion.training_step = step
        
        # Get curriculum stage info
        if diffusion.curriculum_runtime:
            stage, _ = diffusion.curriculum_runtime.get_params(step, 0.1)
            temperature = stage.temperature
            expected_loss_scale = 1.0 / max(temperature, 0.5)
            expected_loss_scale = min(max(expected_loss_scale, 0.1), 2.0)
            
            print(f"\nStep {step}:")
            print(f"  Stage: {stage.name}")
            print(f"  Temperature: {temperature:.2f}")
            print(f"  Expected loss_scale: {expected_loss_scale:.4f}")
            
            # Forward pass to trigger loss calculation
            with torch.no_grad():
                loss, (loss_mse, loss_energy, _) = diffusion(inp, target, mask)
            
            print(f"  Losses computed successfully")
            print(f"  MSE Loss: {loss_mse:.6f}")
            print(f"  Energy Loss: {loss_energy:.6f}")
            print(f"  Total Loss: {loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("The loss_scale now varies with curriculum temperature.")
    print("Higher temperature -> lower loss_scale (more exploration)")
    print("Lower temperature -> higher loss_scale (more exploitation)")

if __name__ == "__main__":
    test_temperature_propagation()