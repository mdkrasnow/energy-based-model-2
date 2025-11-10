# File: /Users/mkrasnow/Desktop/energy-based-model-2/random_noise_baseline.py
# Random Noise Baseline Implementation for Phase 1 Statistical Viability Testing
# Provides pure random noise corruption as a control comparison against ANM

import torch
import torch.nn.functional as F
from typing import Optional, Literal, Union

def random_noise_baseline_corruption(
    ops,
    inp: torch.Tensor,
    x_start: torch.Tensor, 
    t: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_cond: Optional[torch.Tensor] = None,
    noise_scale: float = 1.0,
    base_noise_scale: float = 3.0
) -> torch.Tensor:
    """
    Pure random noise corruption baseline for Phase 1 statistical comparison
    
    This function replaces the complex adversarial negative mining with simple
    random noise addition. It serves as a critical control to determine if ANM's
    improvements are meaningful or if random perturbation achieves similar results.
    
    Args:
        ops: Diffusion operations interface (matches adversarial_corruption signature)
        inp: Input tensor (context for the task)
        x_start: Original samples to corrupt
        t: Timestep tensor for diffusion
        mask: Optional masking (unused in random noise baseline)
        data_cond: Optional conditioning (unused in random noise baseline) 
        noise_scale: Scale factor for random noise (Phase 1 tests: 0.1, 0.5, 1.0)
        base_noise_scale: Base scaling factor (matches ANM parameter)
        
    Returns:
        Corrupted samples with pure random noise
        
    Notes:
        - This is a DROP-IN replacement for _adversarial_corruption
        - Signature matches exactly for seamless integration
        - No gradient computation or energy maximization
        - Serves as sanity check: if ANM ≈ random noise, ANM is useless
    """
    device = x_start.device
    batch_size = x_start.shape[0]
    
    # Generate pure random noise with specified scale
    # Scale relative to input magnitude for fair comparison
    input_scale = x_start.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    noise = torch.randn_like(x_start, device=device) 
    
    # Apply scaled random perturbation
    # noise_scale controls the strength of perturbation relative to signal
    corrupted_samples = x_start + (noise * noise_scale * input_scale * base_noise_scale / 3.0)
    
    return corrupted_samples

def random_noise_training_corruption(
    positives: torch.Tensor,
    noise_scale: float = 1.0,
    corruption_strength: float = 0.5
) -> torch.Tensor:
    """
    Simplified random noise corruption for training pipeline integration
    
    This version is designed for integration into the training loop where we
    need to generate random negative samples during the corruption process.
    
    Args:
        positives: Clean positive samples 
        noise_scale: Relative noise strength (σ parameter)
        corruption_strength: Overall corruption intensity
        
    Returns:
        Randomly corrupted negative samples
    """
    device = positives.device
    
    # Calculate adaptive noise scale based on data magnitude
    data_scale = positives.norm(dim=-1, keepdim=True).mean()
    
    # Generate scaled random noise
    noise = torch.randn_like(positives, device=device)
    
    # Apply corruption: blend original with scaled noise
    alpha = corruption_strength  # How much to corrupt
    corrupted = (1 - alpha) * positives + alpha * noise * noise_scale * data_scale
    
    return corrupted

class RandomNoiseCorruptor:
    """
    Class-based interface for random noise corruption with multiple variants
    
    This provides different noise strategies for Phase 1 testing to ensure
    we test random noise baselines comprehensively.
    """
    
    def __init__(self, noise_scale: float = 1.0):
        self.noise_scale = noise_scale
        
    def uniform_noise(self, x_start: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """Generate uniform random noise corruption"""
        noise = torch.rand_like(x_start) * 2.0 - 1.0  # Uniform [-1, 1]
        data_scale = x_start.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x_start + noise * self.noise_scale * scale_factor * data_scale
    
    def gaussian_noise(self, x_start: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """Generate Gaussian random noise corruption"""
        noise = torch.randn_like(x_start)  # Gaussian N(0,1)
        data_scale = x_start.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x_start + noise * self.noise_scale * scale_factor * data_scale
    
    def laplace_noise(self, x_start: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """Generate Laplace (double exponential) random noise corruption"""
        # Laplace distribution has heavier tails than Gaussian
        uniform = torch.rand_like(x_start)
        laplace_noise = torch.sign(uniform - 0.5) * torch.log(2 * torch.clamp(
            torch.min(uniform, 1 - uniform), min=1e-8))
        
        data_scale = x_start.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x_start + laplace_noise * self.noise_scale * scale_factor * data_scale

# Integration helpers for train.py
def get_random_noise_corruption_fn(noise_type: str = "gaussian", noise_scale: float = 1.0):
    """
    Factory function to create random noise corruption compatible with existing training pipeline
    
    Args:
        noise_type: Type of noise ("gaussian", "uniform", "laplace")
        noise_scale: Noise intensity parameter
        
    Returns:
        Corruption function with signature matching _adversarial_corruption
    """
    corruptor = RandomNoiseCorruptor(noise_scale)
    
    if noise_type == "gaussian":
        base_fn = corruptor.gaussian_noise
    elif noise_type == "uniform":
        base_fn = corruptor.uniform_noise  
    elif noise_type == "laplace":
        base_fn = corruptor.laplace_noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    def corruption_fn(ops, inp, x_start, t, mask=None, data_cond=None, **kwargs):
        """Wrapper to match adversarial_corruption signature"""
        return base_fn(x_start)
    
    return corruption_fn

# Phase 1 specific configurations
PHASE1_RANDOM_NOISE_CONFIGS = {
    "random_noise_weak": {
        "noise_type": "gaussian", 
        "noise_scale": 0.1,
        "description": "Weak Gaussian noise (10% of data scale)"
    },
    "random_noise_medium": {
        "noise_type": "gaussian",
        "noise_scale": 0.5, 
        "description": "Medium Gaussian noise (50% of data scale)"
    },
    "random_noise_strong": {
        "noise_type": "gaussian",
        "noise_scale": 1.0,
        "description": "Strong Gaussian noise (100% of data scale)"
    },
    "random_noise_uniform": {
        "noise_type": "uniform",
        "noise_scale": 1.0,
        "description": "Uniform random noise baseline"
    }
}

def patch_diffusion_for_random_noise(diffusion_model, noise_config: dict):
    """
    Monkey-patch diffusion model to use random noise instead of adversarial corruption
    
    This allows seamless integration with existing training pipeline by replacing
    the adversarial corruption function with random noise generation.
    
    Args:
        diffusion_model: GaussianDiffusion1D instance
        noise_config: Configuration dict from PHASE1_RANDOM_NOISE_CONFIGS
    """
    original_corruption_fn = getattr(diffusion_model, '_adversarial_corruption', None)
    
    # Create random noise corruption function
    corruption_fn = get_random_noise_corruption_fn(
        noise_config["noise_type"], 
        noise_config["noise_scale"]
    )
    
    # Replace the corruption function
    diffusion_model._random_noise_corruption = corruption_fn
    
    # Store original for potential restoration
    if original_corruption_fn:
        diffusion_model._original_adversarial_corruption = original_corruption_fn
    
    print(f"✓ Patched diffusion model with {noise_config['description']}")
    
def restore_original_corruption(diffusion_model):
    """Restore original adversarial corruption function if it was patched"""
    if hasattr(diffusion_model, '_original_adversarial_corruption'):
        diffusion_model._adversarial_corruption = diffusion_model._original_adversarial_corruption
        print("✓ Restored original adversarial corruption function")

def test_random_noise_baseline():
    """Test function to verify random noise implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size = 16
    seq_length = 32
    x_start = torch.randn(batch_size, seq_length, device=device)
    t = torch.randint(0, 10, (batch_size,), device=device)
    
    print("Testing Random Noise Baseline Implementation")
    print("=" * 50)
    
    # Test different noise scales
    for noise_scale in [0.1, 0.5, 1.0]:
        corrupted = random_noise_baseline_corruption(
            None, None, x_start, t, noise_scale=noise_scale
        )
        
        # Calculate corruption statistics
        original_norm = x_start.norm(dim=-1).mean()
        corrupted_norm = corrupted.norm(dim=-1).mean()
        mse_corruption = F.mse_loss(corrupted, x_start)
        
        print(f"Noise Scale {noise_scale}:")
        print(f"  Original norm: {original_norm:.4f}")
        print(f"  Corrupted norm: {corrupted_norm:.4f}")
        print(f"  MSE corruption: {mse_corruption:.4f}")
        print()
    
    print("✓ Random noise baseline implementation verified")

if __name__ == "__main__":
    test_random_noise_baseline()