"""
Energy-based hard negative mining for IRED.

This module implements the core algorithm for generating hard negatives by:
1. Sampling candidates from different energy landscapes
2. Refining each via pure energy descent
3. Selecting the most deceptive (low energy + high error)
"""

from __future__ import annotations
from typing import Optional, Protocol
import torch
import torch.nn.functional as F


class DiffusionOps(Protocol):
    """Protocol for diffusion operations required by HNM."""
    def q_sample(self, x_start, t, noise=None): ...
    def model(self, inp, x, t, return_energy=False, return_both=False): ...
    sqrt_alphas_cumprod: torch.Tensor
    opt_step_size: torch.Tensor
    continuous: bool
    shortest_path: bool


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Extract values from tensor a at indices t, matching the pattern from adversarial_corruption.py"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def energy_based_hard_negative_mining(
    ops: DiffusionOps,
    inp: torch.Tensor,           # Input (e.g., matrix A for inverse task)
    x_start: torch.Tensor,       # Ground truth solution
    t: torch.Tensor,             # Timestep
    mask: Optional[torch.Tensor] = None,     # Optional mask for constrained tasks
    data_cond: Optional[torch.Tensor] = None,# Optional conditioning data
    num_candidates: int = 10,    # Number of landscapes to sample
    refinement_steps: int = 5,   # Steps of energy descent per candidate
    lambda_weight: float = 1.0   # Balance between energy and error (default equal)
) -> torch.Tensor:
    """
    Generate hard negatives by sampling from different energy landscapes and selecting
    the most deceptive ones (low energy + high error).
    
    Args:
        ops: Diffusion operations object
        inp: Input conditioning (e.g., matrix A for inverse problems)
        x_start: Ground truth solution
        t: Timestep tensor
        mask: Optional mask for constrained tasks
        data_cond: Optional conditioning data
        num_candidates: Number of candidate negatives to generate
        refinement_steps: Number of energy descent steps per candidate
        lambda_weight: Weight for error term in deception score
        
    Returns:
        Selected hard negative samples (same shape as x_start)
        
    JUSTIFICATION FOR CHOICES:
    - num_candidates=10: Match IRED's 10 landscapes for natural coverage
    - refinement_steps=5: Match standard IRED opt_step default
    - Pure energy descent: No constraints - we WANT low energy negatives
    - Selection by -energy + error: Directly targets "deceptive" negatives
    """
    
    batch_size = x_start.shape[0]
    device = x_start.device
    
    # Ensure we have enough timesteps for the requested candidates
    # Plan specifies 10 landscapes to match IRED architecture
    max_timesteps = ops.sqrt_alphas_cumprod.shape[0]
    if num_candidates > max_timesteps:
        # Use modulo to cycle through available timesteps if needed
        pass  # Still proceed with full num_candidates as specified in plan
    
    # Storage for all candidates across batch
    all_candidates = []
    
    for landscape_idx in range(num_candidates):
        # Sample from different landscape indices, cycling if necessary
        # Each landscape has different noise scale σ_k for diversity
        actual_idx = landscape_idx % max_timesteps
        t_landscape = torch.full((t.shape[0],), actual_idx, device=t.device, dtype=t.dtype)
        
        # Generate noisy starting point at this landscape
        noise = torch.randn_like(x_start)
        x_init = ops.q_sample(x_start=x_start, t=t_landscape, noise=noise)
        
        # Apply mask if needed (for Sudoku-like tasks)
        if mask is not None:
            x_init = x_init * (1 - mask) + mask * data_cond
        
        # Pure energy descent (no distance penalty)
        # We WANT candidates to reach low energy regions
        x_refined = x_init.clone().detach()
        
        for step in range(refinement_steps):
            x_refined = x_refined.detach()
            x_refined.requires_grad_(True)
            
            try:
                # Get energy gradient
                grad = ops.model(inp, x_refined, t)
                
                # Pure gradient descent on energy
                opt_step_size = _extract(ops.opt_step_size, t, x_refined.shape)
                x_refined = x_refined - opt_step_size * grad
                
                # Reapply mask
                if mask is not None:
                    x_refined = x_refined * (1 - mask) + mask * data_cond
                
                # Clamp to valid range
                if ops.continuous:
                    sf = 2.0
                elif ops.shortest_path:
                    sf = 0.1
                else:
                    sf = 1.0
                max_val = _extract(ops.sqrt_alphas_cumprod, t, x_refined.shape) * sf
                x_refined = torch.clamp(x_refined, -max_val, max_val)
                
            except Exception:
                # If energy computation fails, use the previous iteration
                break
        
        x_refined = x_refined.detach()
        
        # Evaluate this candidate
        try:
            with torch.no_grad():
                energy = ops.model(inp, x_refined, t, return_energy=True)
                error = F.mse_loss(x_refined, x_start, reduction='none').mean(dim=list(range(1, x_refined.ndim)))
                
                all_candidates.append({
                    'x': x_refined,
                    'energy': energy,  # shape: [batch_size]
                    'error': error     # shape: [batch_size]
                })
        except Exception:
            # If energy evaluation fails, skip this candidate
            continue
    
    # Fallback if no candidates were successfully generated
    if not all_candidates:
        # Return a simple noisy version as fallback
        noise = torch.randn_like(x_start)
        return ops.q_sample(x_start=x_start, t=t, noise=3.0 * noise)
    
    # Select per-batch-item independently
    # Each training example needs its own hard negative
    selected_negatives = torch.zeros_like(x_start)
    
    for b in range(batch_size):
        try:
            # Gather energies and errors for this batch item
            energies_b = torch.stack([c['energy'][b] for c in all_candidates])
            errors_b = torch.stack([c['error'][b] for c in all_candidates])
            
            # Check for edge cases (all same values)
            energy_range = energies_b.max() - energies_b.min()
            error_range = errors_b.max() - errors_b.min()
            
            if energy_range < 1e-8 and error_range < 1e-8:
                # All candidates are essentially identical, pick randomly
                best_idx = torch.randint(0, len(all_candidates), (1,)).item()
            else:
                # Normalize to [0,1] before combining
                # Handle zero range case
                if energy_range < 1e-8:
                    energy_norm = torch.zeros_like(energies_b)
                else:
                    energy_norm = (energies_b - energies_b.min()) / energy_range
                
                if error_range < 1e-8:
                    error_norm = torch.zeros_like(errors_b)
                else:
                    error_norm = (errors_b - errors_b.min()) / error_range
                
                # Deception score = -energy + λ*error
                # - Negative energy: lower energy = higher score (model is confident)
                # - Positive error: higher error = higher score (actually wrong)
                deception_scores = -energy_norm + lambda_weight * error_norm
                
                # Select most deceptive
                best_idx = deception_scores.argmax().item()
            
            selected_negatives[b] = all_candidates[best_idx]['x'][b]
            
        except Exception:
            # If selection fails for this batch item, use a random candidate
            if all_candidates:
                random_idx = torch.randint(0, len(all_candidates), (1,)).item()
                selected_negatives[b] = all_candidates[random_idx]['x'][b]
            else:
                # Final fallback: use noisy version of ground truth
                noise = torch.randn_like(x_start[b:b+1])
                selected_negatives[b] = ops.q_sample(x_start=x_start[b:b+1], t=t[b:b+1], noise=3.0 * noise)[0]
    
    return selected_negatives