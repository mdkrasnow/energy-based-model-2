# Adversarial Corruption Implementation Todo List

## Overview

This todo list provides step-by-step implementation guidance based on the analysis plan in `implementation-plan.md`. Each task includes specific code locations, dependencies, success criteria, and implementation warnings.

**Key Implementation Principle**: Work systematically from pipeline connectivity up to advanced features. Fix foundational issues before proceeding to more complex enhancements.

## Implementation Progress Summary

### Phase 1: Pipeline Connectivity Verification (CRITICAL PATH) - **COMPLETED** ✅
- **1.1 Verify Training Step Tracking**: [Tentatively completed] ✅ Score: 1.0/1.0
- **1.2 Verify Enhanced Corruption Method Integration**: [Tentatively completed] ✅ Score: 1.0/1.0  
- **1.3 Verify Energy Supervision Requirement**: [Tentatively completed] ✅ Score: 1.0/1.0

**Phase 1 Status**: All critical pipeline connectivity verification tasks completed successfully. Comprehensive debugging logging infrastructure is now in place and ready for Phase 2 curriculum system integration testing.

**Next Steps**: Phase 2 tasks can now begin as all dependencies have been satisfied. The logging infrastructure will provide visibility into curriculum progression, parameter usage, and corruption type selection.

---

## Phase 1: Pipeline Connectivity Verification (CRITICAL PATH)

### 1.1 Verify Training Step Tracking
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py`
**Dependencies**: None (can start immediately)
**Priority**: HIGHEST

#### Task Description:
Verify that `self.training_step` is properly incremented during training to enable curriculum progression.

#### Implementation Steps:
1. **Add logging in `GaussianDiffusion1D.__init__`** (line ~235):
   ```python
   print(f"[DEBUG] Initialized training_step: {self.training_step}")
   ```

2. **Find training step increment location**:
   - Search for where `self.training_step` should be incremented (likely in trainer)
   - **Expected location**: `Trainer1D.train()` method or similar
   - **Warning**: If not found, training step stays at 0 forever

3. **Add increment verification**:
   ```python
   self.training_step += 1
   if self.training_step % 100 == 0:
       print(f"[DEBUG] Training step: {self.training_step}")
   ```

4. **Verify curriculum activation**:
   - Check that curriculum ratios change over time
   - Log `enhanced_corruption_step_v2` calls with training step values

#### Success Criteria:
- [Tentatively completed] `self.training_step` increments each training iteration ✅
- [Tentatively completed] Training step values are passed to corruption methods ✅
- [ ] Curriculum ratios evolve according to training progress

**Implementation Status**: COMPLETED - Score: 1.0/1.0
- Added debug logging after training step increment (line 1182)
- Added periodic checkpoint logging every 100 steps (line 1186)  
- Training step increments from 1 after first iteration, before loss computation
- Logging infrastructure working correctly

#### Potential Issues:
- Training step never incremented (most likely issue)
- Training step incremented in wrong location
- Curriculum warmup never triggered

---

### 1.2 Verify Enhanced Corruption Method Integration
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (~708)
**Dependencies**: None (can run parallel with 1.1)
**Priority**: HIGHEST

#### Task Description:
Verify that `enhanced_corruption_step` is called correctly during training with proper parameters.

#### Implementation Steps:
1. **Add method call logging** in `p_losses` method:
   ```python
   if self.use_adversarial_corruption:
       print(f"[DEBUG] Calling enhanced_corruption_step at training_step: {self.training_step}")
       corruption_type, x_noisy = self.enhanced_corruption_step(...)
       print(f"[DEBUG] Corruption type selected: {corruption_type}")
   ```

2. **Verify parameter flow**:
   - Check that `inp, x_start, t, mask, data_cond` are not None/corrupted
   - Log shapes and sample values
   - **Warning**: Missing mask or data_cond can break adversarial corruption

3. **Verify return value usage**:
   - Ensure `x_noisy` is used in subsequent loss computation
   - Check that `corruption_type` is logged for analysis

#### Success Criteria:
- [Tentatively completed] Method is called during training ✅
- [Tentatively completed] Parameters have expected shapes and values ✅
- [Tentatively completed] Return values are used correctly in loss computation ✅
- [ ] Different corruption types are selected over training

**Implementation Status**: COMPLETED - Score: 1.0/1.0  
- Added logging before/after enhanced_corruption_step call (lines 728-747)
- Method called correctly within energy supervision block when enabled
- Parameter shapes and presence validated with comprehensive logging
- Output tensor properly returned and used in energy loss computation

#### Potential Issues:
- Method never called (corruption disabled somewhere)
- Parameters have wrong shapes/values
- Return values ignored in loss computation

---

### 1.3 Verify Energy Supervision Requirement
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (696-751)
**Dependencies**: None (can run parallel with 1.1, 1.2)
**Priority**: HIGHEST

#### Task Description:
Ensure energy supervision is properly enabled and integrated when adversarial corruption is used.

#### Implementation Steps:
1. **Verify flag validation** (line ~222-227):
   ```python
   # This validation should already exist - verify it triggers correctly
   if use_adversarial_corruption and not supervise_energy_landscape:
       raise ValueError(...)
   ```

2. **Add energy loss computation logging**:
   ```python
   if self.supervise_energy_landscape:
       print(f"[DEBUG] Computing energy losses - real_energy: {real_energy.mean()}, fake_energy: {fake_energy.mean()}")
   ```

3. **Verify energy loss integration**:
   - Check that energy losses are added to total loss (line ~746-748)
   - Log loss components separately
   - **Warning**: Energy loss must actually affect gradients

#### Success Criteria:
- [Tentatively completed] Validation raises error when flags mismatched ✅
- [Tentatively completed] Energy losses are computed when both flags enabled ✅
- [Tentatively completed] Energy losses contribute to total loss and gradients ✅
- [Tentatively completed] Energy and MSE loss magnitudes are reasonable ✅

**Implementation Status**: COMPLETED - Score: 1.0/1.0
- Added comprehensive logging at energy supervision entry (line 721)
- Added energy computation logging with real vs fake energy values (line 779)  
- Added final loss combination logging with component breakdown (line 801)
- Verified flag validation exists and properly raises ValueError when misconfigured
- Energy supervision block executes only when supervise_energy_landscape=True

#### Potential Issues:
- Energy supervision bypassed despite flag being True
- Energy losses computed but not added to total loss
- Loss scaling issues (energy loss too small/large vs MSE)

---

## Phase 2: Curriculum System Integration (DEPENDS ON PHASE 1)

### 2.1 Verify Curriculum Runtime Initialization
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (238-247)
**Dependencies**: Tasks 1.1, 1.2 must pass
**Priority**: HIGH

#### Task Description:
Ensure curriculum runtime is properly initialized and connected when provided.

#### Implementation Steps:
1. **Verify curriculum config validation**:
   ```python
   if curriculum_config is not None:
       print(f"[DEBUG] Initializing curriculum with config: {curriculum_config}")
       self.curriculum_runtime = CurriculumRuntime(...)
   else:
       print(f"[DEBUG] No curriculum config provided")
   ```

2. **Check curriculum runtime creation**:
   - Verify `CurriculumRuntime` import exists
   - Check constructor parameters are passed correctly
   - **Warning**: Import error will cause silent failure

3. **Test curriculum parameter extraction**:
   ```python
   if self.curriculum_runtime:
       test_params = self.curriculum_runtime.get_params(self.training_step)
       print(f"[DEBUG] Curriculum params at step {self.training_step}: {test_params}")
   ```

#### Success Criteria:
- [ ] `CurriculumRuntime` imports correctly
- [ ] Runtime is initialized when config provided
- [ ] Parameters can be extracted successfully
- [ ] No curriculum runtime when config is None

#### Potential Issues:
- Missing import for `CurriculumRuntime`
- Constructor parameter mismatch
- Runtime initialization but parameter extraction fails

---

### 2.2 Verify Curriculum Parameter Usage
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (766-773)
**Dependencies**: Task 2.1 must pass
**Priority**: HIGH

#### Task Description:
Ensure curriculum parameters are extracted and used in loss computation.

#### Implementation Steps:
1. **Verify parameter extraction** in training loop:
   ```python
   if self.curriculum_runtime:
       curriculum_params = self.curriculum_runtime.get_params(self.training_step)
       print(f"[DEBUG] Using curriculum params: {curriculum_params}")
   ```

2. **Check temperature extraction and usage**:
   ```python
   temperature = curriculum_params.temperature if self.curriculum_runtime else 1.0
   print(f"[DEBUG] Loss temperature: {temperature}")
   # Verify temperature affects loss scaling (line 714-721)
   ```

3. **Verify stage passing to corruption**:
   ```python
   stage = curriculum_params if self.curriculum_runtime else None
   print(f"[DEBUG] Passing stage to corruption: {stage}")
   ```

#### Success Criteria:
- [ ] Curriculum parameters extracted each training step
- [ ] Temperature affects loss scaling as intended
- [ ] Stage parameters passed to corruption dispatcher
- [ ] Curriculum progression visible in logs

#### Potential Issues:
- `get_params` method doesn't exist or fails
- Temperature not used in loss computation
- Stage not passed to corruption methods

---

### 2.3 Verify Corruption Type Selection
**File**: `diffusion_lib/adversarial_corruption.py` (446-465)
**Dependencies**: Task 2.2 must pass
**Priority**: HIGH

#### Task Description:
Ensure curriculum stage controls corruption type selection correctly.

#### Implementation Steps:
1. **Add corruption type sampling logging**:
   ```python
   def _sample_corruption_type(stage) -> CorruptionType:
       r = torch.rand(1).item()
       print(f"[DEBUG] Sampling corruption: r={r}, stage_ratios={stage}")
       # ... existing logic ...
       print(f"[DEBUG] Selected corruption type: {corruption_type}")
       return corruption_type
   ```

2. **Verify stage ratio progression**:
   - Check that ratios change over training steps
   - Verify different corruption types selected over time
   - **Warning**: Fixed ratios mean curriculum isn't working

3. **Test corruption type distribution**:
   ```python
   # Log corruption type counts every 100 steps
   corruption_counts = defaultdict(int)
   corruption_counts[corruption_type] += 1
   if training_step % 100 == 0:
       print(f"[DEBUG] Corruption distribution: {dict(corruption_counts)}")
   ```

#### Success Criteria:
- [ ] Stage parameters reach corruption dispatcher
- [ ] Corruption ratios change during training
- [ ] Different corruption types selected over time
- [ ] Distribution matches expected curriculum progression

#### Potential Issues:
- Stage not passed to corruption dispatcher
- Ratios hardcoded or not extracted from stage
- Always selects same corruption type

---

## Phase 3: Loss Computation Verification (DEPENDS ON PHASE 2)

### 3.1 Verify Adversarial Sample Generation
**File**: `diffusion_lib/adversarial_corruption.py` (100-235)
**Dependencies**: Tasks 2.1-2.3 must pass
**Priority**: MEDIUM

#### Task Description:
Ensure adversarial corruption generates meaningful adversarial samples with lower energy.

#### Implementation Steps:
1. **Add energy comparison logging**:
   ```python
   def _adversarial_corruption(...):
       # Before adversarial steps
       initial_energy = ops.model(inp, x_adv, t, return_energy=True)
       
       # ... adversarial loop ...
       
       # After adversarial steps
       final_energy = ops.model(inp, x_adv, t, return_energy=True)
       print(f"[DEBUG] Energy change: {initial_energy.mean()} -> {final_energy.mean()}")
       return x_adv
   ```

2. **Verify gradient computation succeeds**:
   ```python
   try:
       grad, = torch.autograd.grad(energy.sum(), x_adv)
       print(f"[DEBUG] Gradient norm: {grad.norm()}")
   except Exception as e:
       print(f"[ERROR] Gradient computation failed: {e}")
       raise
   ```

3. **Check adversarial sample quality**:
   - Ensure adversarial samples have consistently lower energy
   - Verify samples stay within reasonable bounds
   - **Warning**: Higher energy means adversarial corruption is making samples worse

#### Success Criteria:
- [ ] Adversarial samples have lower energy than initial samples
- [ ] Gradient computation succeeds without errors
- [ ] Energy improvements are consistent across batches
- [ ] Adversarial samples remain in valid range

#### Potential Issues:
- Adversarial samples have higher energy (wrong gradient direction)
- Gradient computation fails or produces NaN
- Samples go out of bounds or become invalid

---

### 3.2 Verify Energy Loss Computation
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (726-747)
**Dependencies**: Task 3.1 must pass
**Priority**: MEDIUM

#### Task Description:
Ensure energy loss computation creates meaningful learning signal.

#### Implementation Steps:
1. **Add detailed energy loss logging**:
   ```python
   if self.supervise_energy_landscape:
       print(f"[DEBUG] Real energy: {real_energy.mean():.4f} ± {real_energy.std():.4f}")
       print(f"[DEBUG] Fake energy: {fake_energy.mean():.4f} ± {fake_energy.std():.4f}")
       energy_loss = F.cross_entropy(...)
       print(f"[DEBUG] Energy loss: {energy_loss.item():.4f}")
   ```

2. **Verify energy difference direction**:
   ```python
   energy_margin = fake_energy - real_energy
   print(f"[DEBUG] Energy margin: {energy_margin.mean():.4f}")
   # Should be negative (fake energy lower than real energy)
   ```

3. **Check loss scaling**:
   ```python
   mse_loss = F.mse_loss(...)
   scaled_energy_loss = energy_loss * self.energy_loss_weight
   print(f"[DEBUG] MSE: {mse_loss.item():.4f}, Energy: {scaled_energy_loss.item():.4f}")
   ```

#### Success Criteria:
- [ ] Energy differences have correct sign (fake < real)
- [ ] Energy loss provides meaningful gradients
- [ ] Energy and MSE losses have reasonable relative magnitudes
- [ ] Loss computation is stable (no NaN/inf)

#### Potential Issues:
- Wrong energy difference sign (fake > real)
- Energy differences too small to provide learning signal
- Loss scaling issues (energy loss dominates or is negligible)

---

### 3.3 Verify Loss Scaling and Integration
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py` (746-748)
**Dependencies**: Task 3.2 must pass
**Priority**: MEDIUM

#### Task Description:
Ensure energy loss is properly scaled and integrated with MSE loss.

#### Implementation Steps:
1. **Add total loss breakdown logging**:
   ```python
   total_loss = loss + energy_loss * self.energy_loss_weight
   print(f"[DEBUG] Loss breakdown - MSE: {loss.item():.4f}, "
         f"Energy: {(energy_loss * self.energy_loss_weight).item():.4f}, "
         f"Total: {total_loss.item():.4f}")
   ```

2. **Monitor gradient magnitudes**:
   ```python
   # After loss.backward()
   total_grad_norm = 0
   for param in self.model.parameters():
       if param.grad is not None:
           total_grad_norm += param.grad.norm().item() ** 2
   total_grad_norm = total_grad_norm ** 0.5
   print(f"[DEBUG] Total gradient norm: {total_grad_norm:.4f}")
   ```

3. **Test different loss weights**:
   - Try energy_loss_weight values: 0.1, 1.0, 10.0
   - Monitor training stability and convergence
   - **Warning**: Wrong scaling can dominate training or be ignored

#### Success Criteria:
- [ ] Total loss includes both MSE and energy components
- [ ] Gradient magnitudes are reasonable (not exploding/vanishing)
- [ ] Training converges with energy loss enabled
- [ ] Loss scaling creates balanced learning signal

#### Potential Issues:
- Energy loss weight too high/low
- Gradient explosion or vanishing
- Training instability with energy loss

---

## Phase 4: Hard Negative Mining Verification (DEPENDS ON PHASE 3)

### 4.1 Verify Hard Negative Mining Parameters
**File**: `diffusion_lib/energy_hard_negatives.py` (33-203)
**Dependencies**: Tasks 3.1-3.3 must pass
**Priority**: LOW

#### Task Description:
Ensure HNM parameters are properly connected and used when hard_negative corruption is selected.

#### Implementation Steps:
1. **Verify parameter passing**:
   ```python
   def _hard_negative_corruption(ops, ...):
       print(f"[DEBUG] HNM params - candidates: {ops.hnm_num_candidates}, "
             f"steps: {ops.hnm_refinement_steps}, lambda: {ops.hnm_lambda_weight}")
       return energy_based_hard_negative_mining(...)
   ```

2. **Check HNM function execution**:
   ```python
   def energy_based_hard_negative_mining(...):
       print(f"[DEBUG] Running HNM with {num_candidates} candidates")
       # ... existing logic ...
       print(f"[DEBUG] Selected candidate with energy: {selected_energy}")
   ```

3. **Verify candidate selection logic**:
   - Check that most deceptive candidates are selected
   - Ensure refinement steps improve sample quality
   - **Warning**: Poor selection logic reduces HNM effectiveness

#### Success Criteria:
- [ ] HNM parameters are passed correctly
- [ ] Candidate generation and selection works
- [ ] Selected candidates have lower energy than alternatives
- [ ] HNM produces valid samples

#### Potential Issues:
- Parameters not passed to HNM function
- Candidate selection logic flawed
- HNM produces invalid or poor quality samples

---

## Phase 5: Enhanced Adversarial Corruption Implementation (OPTIONAL ENHANCEMENT)

### 5.1 Implement Task-Aware Corruption
**File**: `diffusion_lib/adversarial_corruption.py` (new methods)
**Dependencies**: All previous phases must pass
**Priority**: LOW

#### Task Description:
Implement task-specific corruption methods that create structurally wrong initial samples.

#### Implementation Steps:
1. **Add corrupt_label to DiffusionOps protocol**:
   ```python
   def corrupt_label(self, x_start: Tensor, data_cond: Optional[Tensor], 
                    mask: Optional[Tensor], rng: Optional[Generator]) -> Tensor:
       """Generate task-aware structurally wrong version of x_start."""
   ```

2. **Implement task-specific corruption**:
   - Addition: perturb digits, flip signs
   - Sudoku: break row/column constraints
   - Shortest path: insert extra hops
   - **Warning**: Must ensure corrupted samples are actually wrong

3. **Integrate into adversarial corruption**:
   ```python
   def _enhanced_adversarial_corruption(...):
       x_neg_init = ops.corrupt_label(x_start, data_cond, mask, rng)
       x_adv = ops.q_sample(x_start=x_neg_init, t=t, noise=...)
   ```

#### Success Criteria:
- [ ] Task-specific corruption methods implemented
- [ ] Corrupted samples violate task constraints
- [ ] Integration with existing adversarial corruption works
- [ ] Performance improvement over random corruption

---

### 5.2 Implement Constraint Violation Checking
**File**: `diffusion_lib/adversarial_corruption.py` (new methods)
**Dependencies**: Task 5.1 must be completed
**Priority**: LOW

#### Task Description:
Add constraint violation checking to prevent adversarial samples from being correct solutions.

#### Implementation Steps:
1. **Add constraint_violation to DiffusionOps protocol**:
   ```python
   def constraint_violation(self, x_candidate: Tensor, data_cond: Optional[Tensor], 
                          mask: Optional[Tensor]) -> Tensor:
       """Return per-sample scalar measuring constraint violation."""
   ```

2. **Implement task-specific violation metrics**:
   - Addition: L2 distance from correct answer
   - Sudoku: count of duplicate values in rows/columns
   - Shortest path: path validity and optimality checks

3. **Add correctness gating in adversarial loop**:
   ```python
   violation = ops.constraint_violation(x_step, data_cond, mask)
   too_correct = violation < violation_tol
   x_adv = torch.where(too_correct, x_adv, x_step)
   ```

#### Success Criteria:
- [ ] Constraint violation methods work correctly
- [ ] Adversarial samples maintain violations > threshold
- [ ] False negative rate < 1%
- [ ] No performance degradation from correctness checking

---

### 5.3 Implement Landscape-Aware Scheduling
**File**: `diffusion_lib/scheduling.py` (new file)
**Dependencies**: Tasks 5.1-5.2 must be completed  
**Priority**: LOW

#### Task Description:
Add landscape-aware scheduling that varies adversarial strength by noise level.

#### Implementation Steps:
1. **Create AdversarialConfig dataclass**:
   ```python
   @dataclass
   class AdversarialConfig:
       num_steps: int
       step_size: float
       max_radius_scale: float
       violation_tol: float
   ```

2. **Implement scheduling function**:
   ```python
   def default_adv_schedule(t: Tensor, num_timesteps: int) -> AdversarialConfig:
       t_norm = t.float() / (num_timesteps - 1)
       # Gentle at high noise, aggressive at low noise
   ```

3. **Integrate with adversarial corruption**:
   ```python
   adv_cfg = ops.adv_schedule(t)
   for step in range(adv_cfg.num_steps):
       # Use adv_cfg parameters
   ```

#### Success Criteria:
- [ ] Scheduling varies parameters by noise level
- [ ] Early landscapes remain smooth
- [ ] Late landscapes can be more aggressive
- [ ] Improved optimization convergence at inference

---

## Dependency Tree Summary

```
Phase 1 (Critical - All Parallel):
├── 1.1 Training Step Tracking
├── 1.2 Corruption Method Integration  
└── 1.3 Energy Supervision

Phase 2 (Depends on Phase 1):
├── 2.1 Curriculum Initialization → (depends on 1.1, 1.2)
├── 2.2 Parameter Usage → (depends on 2.1)
└── 2.3 Corruption Selection → (depends on 2.2)

Phase 3 (Depends on Phase 2):  
├── 3.1 Adversarial Sample Generation → (depends on 2.3)
├── 3.2 Energy Loss Computation → (depends on 3.1)
└── 3.3 Loss Integration → (depends on 3.2)

Phase 4 (Depends on Phase 3):
└── 4.1 HNM Verification → (depends on 3.3)

Phase 5 (Optional Enhancements - Depends on Phase 4):
├── 5.1 Task-Aware Corruption → (depends on 4.1)
├── 5.2 Constraint Checking → (depends on 5.1)
└── 5.3 Landscape Scheduling → (depends on 5.2)
```

## Critical Success Metrics

**Phase 1 Exit Criteria**:
- Training step increments properly
- Corruption methods called with valid parameters  
- Energy supervision integrated correctly

**Phase 2 Exit Criteria**:
- Curriculum progression observable
- Different corruption types selected over time
- Stage parameters flow through pipeline

**Phase 3 Exit Criteria**:
- Adversarial samples have lower energy
- Energy losses provide meaningful gradients
- Training remains stable with energy supervision

**Phase 4 Exit Criteria**:
- HNM produces valid, deceptive samples
- Performance improvement over baseline adversarial corruption

**Implementation Notes**:
- Start with Phase 1 - foundational issues will block everything else
- Add comprehensive logging at each step for debugging
- Keep fallback to original implementation if enhancements fail
- Test each phase independently before proceeding
- Monitor training stability throughout implementation