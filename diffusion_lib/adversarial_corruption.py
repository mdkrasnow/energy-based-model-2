from __future__ import annotations

from typing import Literal, Protocol, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .energy_hard_negatives import energy_based_hard_negative_mining

CorruptionType = Literal["clean", "adversarial", "gaussian", "standard", "hard_negative"]


class DiffusionOps(Protocol):
    # required ops/buffers from GaussianDiffusion1D (read/write as used)
    def q_sample(self, x_start, t, noise=None): ...
    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True): ...
    sqrt_alphas_cumprod: torch.Tensor
    opt_step_size: torch.Tensor
    continuous: bool
    shortest_path: bool
    sudoku: bool
    anm_adversarial_steps: int
    model: object  # callable like model(inp, x, t, return_both/return_energy)
    # Hard negative mining parameters
    hnm_num_candidates: int
    hnm_refinement_steps: int
    hnm_lambda_weight: float


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _sample_corruption_type(stage) -> CorruptionType:
    r = torch.rand(1).item()
    if r < stage.clean_ratio:
        return "clean"
    if r < stage.clean_ratio + stage.adversarial_ratio:
        return "adversarial"
    if r < stage.clean_ratio + stage.adversarial_ratio + stage.hard_negative_ratio:
        return "hard_negative"
    return "gaussian"


def _clean_corruption(ops: DiffusionOps, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    return ops.q_sample(x_start=x_start, t=t, noise=noise)


def _gaussian_noise_corruption(
    ops: DiffusionOps, x_start: torch.Tensor, t: torch.Tensor, scale: float = 3.0
) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    return ops.q_sample(x_start=x_start, t=t, noise=scale * noise)


def _standard_ired_corruption(
    ops: DiffusionOps, inp: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor],
    data_cond: Optional[torch.Tensor], base_noise_scale: float
) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    xmin_noise = ops.q_sample(x_start=x_start, t=t, noise=base_noise_scale * noise)

    if mask is not None:
        xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

    step = 20 if ops.sudoku else 5
    xmin_noise = ops.opt_step(inp, xmin_noise, t, mask, data_cond, step=step, sf=1.0)
    return xmin_noise


def _adversarial_corruption(
    ops: DiffusionOps, inp: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor],
    data_cond: Optional[torch.Tensor], base_noise_scale: float, epsilon: float
) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    xmin_noise = ops.q_sample(x_start=x_start, t=t, noise=base_noise_scale * noise)

    if mask is not None:
        xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

    xmin_noise.requires_grad_(True)
    opt_step_size = _extract(ops.opt_step_size, t, xmin_noise.shape)
    xmin_noise_orig = xmin_noise.clone().detach()

    for i in range(ops.anm_adversarial_steps):
        energy, grad = ops.model(inp, xmin_noise, t, return_both=True)

        distance_penalty = F.mse_loss(xmin_noise, xmin_noise_orig)
        adaptive_penalty_weight = epsilon * torch.clamp(1.0 / (distance_penalty + 1e-6), 0.1, 2.0)

        penalty_grad = torch.autograd.grad(distance_penalty, xmin_noise, create_graph=False, retain_graph=True)[0]
        modified_grad = grad - adaptive_penalty_weight * penalty_grad

        step_scale = 1.0 * (0.7 ** i)
        xmin_noise = xmin_noise - opt_step_size * modified_grad * step_scale

        if mask is not None:
            xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

        if ops.continuous:
            sf = 2.0
        elif ops.shortest_path:
            sf = 0.1
        else:
            sf = 1.0

        max_val = _extract(ops.sqrt_alphas_cumprod, t, xmin_noise.shape) * sf
        xmin_noise = torch.clamp(xmin_noise, -max_val, max_val)
        xmin_noise.requires_grad_(True)

    return xmin_noise.detach()


# def _adversarial_corruption_lowrank(
#     ops: DiffusionOps,
#     inp: torch.Tensor,
#     x_start: torch.Tensor,
#     t: torch.Tensor,
#     mask: Optional[torch.Tensor],
#     data_cond: Optional[torch.Tensor],
#     base_noise_scale: float,
#     epsilon: float,
#     rank: int
# ) -> torch.Tensor:
#     """
#     Generate adversarial negatives for low-rank matrix completion.
#     Uses factorized parameterization X = U·V^T to preserve rank constraint.
    
#     Args:
#         ops: Diffusion operations
#         inp: Input context (masked matrix)
#         x_start: Starting point (positive sample)
#         t: Diffusion timestep
#         mask: Optional mask for conditional generation
#         data_cond: Optional conditioning data
#         base_noise_scale: Noise scale for initial corruption
#         epsilon: Distance penalty weight
#         rank: Rank constraint (r in rank-r matrices)
    
#     Returns:
#         Adversarial corrupted sample (guaranteed to be rank-r or less)
#     """
#     # Initial noise corruption
#     noise = torch.randn_like(x_start)
#     xmin_noise = ops.q_sample(x_start=x_start, t=t, noise=base_noise_scale * noise)
    
#     if mask is not None:
#         xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
    
#     # Reshape to matrix form (assuming flattened input)
#     batch_size = xmin_noise.shape[0]
#     total_dim = xmin_noise.shape[1]
#     # Infer matrix dimensions (assume square matrix)
#     matrix_dim = int(total_dim ** 0.5)
#     assert matrix_dim * matrix_dim == total_dim, f"Input must be flattened square matrix, got {total_dim}"
    
#     x_matrix = xmin_noise.reshape(batch_size, matrix_dim, matrix_dim)
    
#     # Factorize: X = U·V^T using SVD for initialization
#     try:
#         U_full, S_full, Vt_full = torch.linalg.svd(x_matrix, full_matrices=False)
#         # Take only top-r components
#         U = U_full[:, :, :rank].clone()  # [B, m, r]
#         S_r = S_full[:, :rank].clone()   # [B, r]
#         V = Vt_full[:, :rank, :].transpose(1, 2).clone()  # [B, n, r]
        
#         # Incorporate singular values into factors
#         S_sqrt = torch.sqrt(S_r + 1e-8).unsqueeze(1)  # [B, 1, r]
#         U = U * S_sqrt  # [B, m, r]
#         V = V * S_sqrt  # [B, n, r]
#     except:
#         # Fallback if SVD fails
#         U = torch.randn(batch_size, matrix_dim, rank, device=x_matrix.device) * 0.1
#         V = torch.randn(batch_size, matrix_dim, rank, device=x_matrix.device) * 0.1
    
#     # Make factors require gradients
#     U.requires_grad_(True)
#     V.requires_grad_(True)
    
#     # Store original for distance penalty
#     x_matrix_orig = x_matrix.clone().detach()
    
#     # Get step size
#     opt_step_size = _extract(ops.opt_step_size, t, xmin_noise.shape)
    
#     # Gradient ascent in factor space
#     for i in range(ops.anm_adversarial_steps):
#         # Reconstruct matrix from factors
#         x_reconstructed = torch.bmm(U, V.transpose(1, 2))  # [B, m, n]
        
#         # Flatten for energy computation
#         x_flat = x_reconstructed.reshape(batch_size, -1)
        
#         # Apply mask if present
#         if mask is not None:
#             x_flat = x_flat * (1 - mask) + mask * data_cond
        
#         # Compute energy
#         energy = ops.model(inp, x_flat, t, return_energy=True)
        
#         # Distance penalty to keep corruption bounded
#         distance = F.mse_loss(x_reconstructed, x_matrix_orig, reduction='none').sum(dim=[1, 2])
#         adaptive_penalty_weight = epsilon * torch.clamp(1.0 / (distance + 1e-6), 0.1, 2.0)
        
#         # Total objective: maximize energy while limiting distance
#         objective = energy.sum() - (adaptive_penalty_weight * distance).sum()
        
#         # Compute gradients w.r.t. factors
#         grads = torch.autograd.grad(
#             -objective,  # Negative because we want to maximize
#             [U, V],
#             create_graph=False,
#             retain_graph=False
#         )
#         grad_U, grad_V = grads
        
#         # Gradient ascent step with decay
#         step_scale = 1.0 * (0.7 ** i)
        
#         with torch.no_grad():
#             # Use step size
#             step_size_scalar = opt_step_size.mean()
            
#             U = U - step_size_scalar * grad_U * step_scale
#             V = V - step_size_scalar * grad_V * step_scale
            
#             # Apply clamping to maintain reasonable scale
#             if ops.continuous:
#                 sf = 2.0
#             elif ops.shortest_path:
#                 sf = 0.1
#             else:
#                 sf = 1.0
            
#             max_val = _extract(ops.sqrt_alphas_cumprod, t, xmin_noise.shape).mean()
#             max_val_factor = (max_val * sf) / (rank ** 0.5)  # Scale for factors
            
#             U = torch.clamp(U, -max_val_factor, max_val_factor)
#             V = torch.clamp(V, -max_val_factor, max_val_factor)
            
#             # Re-enable gradients for next iteration
#             if i < ops.anm_adversarial_steps - 1:
#                 U.requires_grad_(True)
#                 V.requires_grad_(True)
    
#     # Final reconstruction
#     with torch.no_grad():
#         x_final = torch.bmm(U, V.transpose(1, 2))
#         x_final_flat = x_final.reshape(batch_size, -1)
        
#         # Apply mask one final time
#         if mask is not None:
#             x_final_flat = x_final_flat * (1 - mask) + mask * data_cond
        
#         # Verify no NaN values
#         if torch.any(torch.isnan(x_final_flat)):
#             # Fallback to standard corruption if something went wrong
#             return _adversarial_corruption(ops, inp, x_start, t, mask, data_cond, base_noise_scale, epsilon)
    
#     return x_final_flat.detach()


def _hard_negative_corruption(
    ops: DiffusionOps,
    inp: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor],
    data_cond: Optional[torch.Tensor],
    base_noise_scale: float = 3.0
) -> torch.Tensor:
    """
    Generate hard negatives using energy-based hard negative mining.
    
    This is a wrapper around the core HNM algorithm that follows the same
    interface as other corruption functions. HNM parameters are taken from
    the ops object.
    """
    return energy_based_hard_negative_mining(
        ops=ops,
        inp=inp,
        x_start=x_start,
        t=t,
        mask=mask,
        data_cond=data_cond,
        num_candidates=ops.hnm_num_candidates,
        refinement_steps=ops.hnm_refinement_steps,
        lambda_weight=ops.hnm_lambda_weight
    )


def enhanced_corruption_step_v2(
    ops: DiffusionOps,
    stage,
    epsilon: float,
    inp: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor],
    data_cond: Optional[torch.Tensor],
    base_noise_scale: float = 3.0,
    constraint_config: Optional[object] = None,  # NEW parameter
) -> Tuple[CorruptionType, torch.Tensor]:
    """
    Curriculum-aware dispatcher with optional constraint-aware corruption.
    
    Args:
        ops: Diffusion operations
        stage: Current curriculum stage
        epsilon: Distance penalty parameter
        inp: Input context
        x_start: Starting point
        t: Diffusion timestep
        mask: Optional mask for conditional generation
        data_cond: Optional conditioning data
        base_noise_scale: Base noise scale
        constraint_config: Optional ConstraintConfig for constraint-aware corruption
        
    Returns:
        Tuple of (corruption_type, corrupted_sample)
    """
    corruption_type = _sample_corruption_type(stage)

    if corruption_type == "clean":
        x_corrupted = _clean_corruption(ops, x_start, t)
    elif corruption_type == "gaussian":
        noise_scale = base_noise_scale * (2.0 / max(stage.temperature, 1.0))
        x_corrupted = _gaussian_noise_corruption(ops, x_start, t, noise_scale)
    elif corruption_type == "hard_negative":
        # Energy-based hard negative mining
        x_corrupted = _hard_negative_corruption(
            ops, inp, x_start, t, mask, data_cond, base_noise_scale
        )
    else:  # adversarial
        # Standard unconstrained adversarial corruption
        x_corrupted = _adversarial_corruption(
            ops, inp, x_start, t, mask, data_cond, 
            base_noise_scale, epsilon
        )

    return corruption_type, x_corrupted


def enhanced_corruption_step_legacy(
    ops: DiffusionOps,
    use_adversarial_corruption: bool,
    training_step: int,
    anm_warmup_steps: int,
    anm_distance_penalty: float,
    recent_energy_diffs: list,
    inp: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor],
    data_cond: Optional[torch.Tensor],
    base_noise_scale: float = 3.0,
) -> Tuple[CorruptionType, torch.Tensor]:
    """
    Legacy behavior when curriculum is disabled (mirrors the previous inline method).
    """
    if training_step < anm_warmup_steps:
        curriculum_weight = 0.0
    else:
        progress = (training_step - anm_warmup_steps) / max(anm_warmup_steps, 1)
        curriculum_weight = min(1.0, progress)
        if len(recent_energy_diffs) > 10:
            quality_factor = np.mean(recent_energy_diffs[-10:])
            curriculum_weight *= np.clip(quality_factor, 0.1, 1.0)

    if torch.rand(1).item() < curriculum_weight and use_adversarial_corruption:
        return "adversarial", _adversarial_corruption(
            ops, inp, x_start, t, mask, data_cond, base_noise_scale, anm_distance_penalty
        )
    else:
        return "standard", _standard_ired_corruption(
            ops, inp, x_start, t, mask, data_cond, base_noise_scale
        )
