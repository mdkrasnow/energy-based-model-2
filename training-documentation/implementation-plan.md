# Implementation Plan: Enhanced Adversarial Corruption for IRED-Style EBMs

## Problem Analysis

The current `_adversarial_corruption` implementation in `diffusion_lib/adversarial_corruption.py:100-234` suffers from several critical issues that prevent it from being a principled adversarial negative mining scheme for IRED/EBMs:

### Root Cause Analysis

1. **False Negative Problem**: No explicit "incorrectness" constraint means the adversary can generate valid solutions as negatives, injecting label noise into contrastive learning
2. **Poor Scheduling**: Fixed adversarial hyperparameters across all noise levels violates annealed EBM principles 
3. **Insufficient Constraint Enforcement**: "Projection" only does amplitude clamping, not semantic constraint validation
4. **Geometric Issues**: L2 radius calculation includes masked dimensions, artificially limiting adversary strength on modifiable variables
5. **Structural Blindness**: Starts from noisy positives rather than task-aware corrupted negatives

### Impact Assessment

- **Immediate**: Code runs but leaves "performance on the table" and may inject training noise
- **Long-term**: Will fail on structured tasks (Sudoku, shortest path) where constraint violations are discrete
- **Scalability**: Over-sharpening early landscapes will hurt optimization in complex reasoning tasks

## Implementation Strategy

### Phase 1: Core Infrastructure (1-2 weeks)

#### 1.1 Extend DiffusionOps Interface

**File**: `diffusion_lib/adversarial_corruption.py:14-29`

Add these protocol methods to the `DiffusionOps` class:

```python
class DiffusionOps(Protocol):
    # ... existing methods ...
    
    def corrupt_label(self, x_start: Tensor, data_cond: Optional[Tensor], 
                     mask: Optional[Tensor], rng: Optional[Generator]) -> Tensor:
        """Generate task-aware structurally wrong version of x_start at x_0 level."""
    
    def constraint_violation(self, x_candidate: Tensor, data_cond: Optional[Tensor], 
                           mask: Optional[Tensor]) -> Tensor:
        """Return per-sample scalar measuring constraint violation (0 = valid solution)."""
    
    def is_correct(self, x_candidate: Tensor, data_cond: Optional[Tensor], 
                  mask: Optional[Tensor], tol: float = 1e-3) -> Tensor:
        """Return boolean mask: True where candidate is effectively correct."""
    
    def adv_schedule(self, t: Tensor) -> 'AdversarialConfig':
        """Map diffusion timestep to landscape-aware adversarial configuration."""
```

#### 1.2 Adversarial Configuration Structure

**File**: `diffusion_lib/adversarial_corruption.py:11-13` (after CorruptionType)

```python
@dataclass
class AdversarialConfig:
    num_steps: int
    step_size: float  
    max_radius_scale: float
    violation_tol: float  # tolerance below which candidate is "too correct"
```

#### 1.3 Task-Specific Constraint Implementations

**File**: `diffusion_lib/task_constraints.py` (new)

Implement domain-specific corruption and validation:
- `corrupt_addition()`: perturb subset of digits, flip signs, swap operands
- `corrupt_sudoku()`: break row/column/subgrid constraints systematically  
- `corrupt_shortest_path()`: insert extra hops, create cycles
- `constraint_violation_*()`: quantify how wrong each candidate is

### Phase 2: Enhanced Adversarial Algorithm (1-2 weeks)

#### 2.1 Replace Core Adversarial Function

**File**: `diffusion_lib/adversarial_corruption.py:100-234`

Replace `_adversarial_corruption` with:

```python
def _enhanced_adversarial_corruption(
    ops: "DiffusionOps",
    inp: Tensor,
    x_start: Tensor, 
    t: Tensor,
    mask: Optional[Tensor],
    data_cond: Optional[Tensor],
    base_noise_scale: float,
    rng: Optional[Generator] = None,
) -> Tensor:
    """
    Enhanced adversarial corruption with:
    - Task-aware initial corruption
    - Landscape-aware scheduling
    - Explicit incorrectness enforcement
    - Free-dimension-only radius computation
    """
```

**Key algorithmic changes:**

1. **Task-aware initialization**: 
   ```python
   x_neg_init = ops.corrupt_label(x_start, data_cond, mask, rng)
   x_adv = ops.q_sample(x_start=x_neg_init, t=t, noise=base_noise_scale * noise)
   ```

2. **Free-dimension radius**:
   ```python
   free_mask = (~mask).float() if mask is not None else 1.0
   delta0 = (x_adv - x_start) * free_mask
   radius = l2_norm_per_sample(delta0)
   ```

3. **Correctness gating**:
   ```python
   violation = ops.constraint_violation(x_step, data_cond, mask)
   too_correct = violation < adv_cfg.violation_tol
   x_adv = torch.where(too_correct, x_adv, x_step)  # reject if too correct
   ```

4. **Landscape-aware scheduling**:
   ```python
   adv_cfg = ops.adv_schedule(t)
   step_size = adv_cfg.step_size  # not tied to base_noise_scale
   ```

#### 2.2 Helper Utilities

**File**: `diffusion_lib/adversarial_utils.py` (new)

```python
def l2_norm_per_sample(delta: Tensor) -> Tensor:
    """Compute L2 norm per batch sample, reshape for broadcasting."""

def normalize_per_sample(grad: Tensor) -> Tensor:
    """Normalize gradient per sample for stable adversarial steps."""

def project_to_l2_ball(delta: Tensor, max_radius: Tensor) -> Tensor:
    """Project delta vectors into per-sample L2 balls."""
```

### Phase 3: Scheduling and Integration (1 week)

#### 3.1 Implement Landscape-Aware Scheduling

**File**: `diffusion_lib/scheduling.py` (new)

```python
def default_adv_schedule(t: Tensor, num_timesteps: int) -> AdversarialConfig:
    """
    Default schedule: gentle at high noise, aggressive at low noise.
    
    High noise (early): few steps, small radius, loose violation tolerance  
    Low noise (late): more steps, larger radius, strict violation tolerance
    """
    t_norm = t.float() / (num_timesteps - 1)  # 0=high noise, 1=low noise
    
    num_steps = int(torch.lerp(1.0, 4.0, t_norm).round().item())
    step_size = torch.lerp(0.1, 0.5, t_norm).item() 
    max_radius_scale = torch.lerp(1.0, 3.0, t_norm).item()
    violation_tol = torch.lerp(0.1, 0.01, t_norm).item()  # stricter at low noise
    
    return AdversarialConfig(num_steps, step_size, max_radius_scale, violation_tol)
```

#### 3.2 Update Training Integration

**File**: `train.py` (wherever `enhanced_corruption_step_v2` is called)

Update the corruption dispatcher to use enhanced adversarial corruption:

```python
# In enhanced_corruption_step_v2, line ~458-463
else:  # adversarial
    x_corrupted = _enhanced_adversarial_corruption(
        ops, inp, x_start, t, mask, data_cond, 
        base_noise_scale, rng=torch.Generator().manual_seed(global_step)
    )
```

### Phase 4: Monitoring and Validation (1 week)

#### 4.1 Enhanced Logging

**File**: `train.py` (in training loop)

Add validation metrics:

```python
# Log constraint violation stats  
if corruption_type == "adversarial":
    with torch.no_grad():
        violation_scores = ops.constraint_violation(x_corrupted, data_cond, mask)
        wandb.log({
            "adv_violation_mean": violation_scores.mean(),
            "adv_violation_min": violation_scores.min(),
            "adv_false_negative_rate": (violation_scores < 1e-3).float().mean()
        })

# Log energy margins
E_pos = ops.model(inp, x_pos, t, return_energy=True)
E_neg = ops.model(inp, x_corrupted, t, return_energy=True) 
energy_margin = E_neg - E_pos
wandb.log({
    "energy_margin_mean": energy_margin.mean(),
    "energy_margin_std": energy_margin.std()
})
```

#### 4.2 Validation Experiments

**File**: `experiments/validate_corruption.py` (new)

Create validation script to test:
- Constraint violation rates across corruption types
- Energy landscape sharpness by timestep  
- Optimization convergence on harder test problems
- Ablation: enhanced vs original adversarial corruption

## Success Criteria

### Immediate Validation
- [ ] `constraint_violation()` returns > 0 for all adversarial negatives
- [ ] False negative rate < 1% (adversarial negatives that are actually correct)
- [ ] Energy margins `E_neg - E_pos` remain positive and stable  
- [ ] No gradient explosion or NaN values during adversarial steps

### Performance Validation  
- [ ] Improved test accuracy on same-difficulty tasks vs current implementation
- [ ] Better generalization to harder difficulty levels (2-3x increase in constraint complexity)
- [ ] Faster optimization convergence at inference time (fewer gradient steps to solution)
- [ ] Maintained training stability (no loss spikes, energy divergence)

### Algorithmic Validation
- [ ] Adversarial strength scales appropriately with noise level (more aggressive at low noise)
- [ ] Task constraints are respected (no invalid Sudoku solutions as negatives)
- [ ] L2 balls are computed correctly on free dimensions only
- [ ] Scheduling reduces early landscape over-sharpening

## Risk Mitigation

### High-Risk Areas
1. **Constraint implementation complexity**: Start with simple tasks (addition) before complex ones (Sudoku)
2. **Training instability**: Keep fallback to original corruption if adversarial steps fail
3. **Performance regression**: A/B test enhanced vs original on existing benchmarks

### Fallback Strategies  
- Phase implementation allows reverting to any stable checkpoint
- Keep original `_adversarial_corruption` as `_adversarial_corruption_legacy`
- Feature flags to enable/disable enhanced components individually

## Implementation Timeline

**Week 1-2**: Phase 1 - Infrastructure and Protocol Extensions
**Week 3-4**: Phase 2 - Core Algorithm Implementation  
**Week 5**: Phase 3 - Scheduling and Integration
**Week 6**: Phase 4 - Validation and Monitoring
**Week 7**: Performance analysis and refinement

**Total Estimated Effort**: 6-7 weeks for complete implementation and validation

This plan addresses all identified issues systematically while maintaining backward compatibility and providing clear validation criteria for each phase.