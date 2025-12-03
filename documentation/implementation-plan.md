# Adversarial Corruption Pipeline Implementation Analysis Plan

## Executive Summary

This document provides a systematic implementation plan to diagnose why adversarial corruption (ANM - Adversarial Negative Mining) is not improving model performance. Rather than examining the adversarial corruption algorithm itself, this plan focuses on identifying misspecifications in the pipeline that connects adversarial corruption to the energy training system.

## Problem Statement

The adversarial corruption implementation exists but is not delivering expected performance improvements. This suggests potential issues in:
1. Pipeline integration and data flow
2. Loss computation and backpropagation
3. Parameter connectivity and timing
4. Curriculum and training state management

## Analysis Architecture

### Key Components Identified

1. **Main Training Script**: `train.py:489-498` - Entry point where ANM parameters are configured
2. **Diffusion Model**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:216-221` - Core validation that ANM requires energy supervision
3. **Adversarial Corruption**: `diffusion_lib/adversarial_corruption.py:416-502` - Corruption dispatcher and implementation
4. **Energy Hard Negatives**: `diffusion_lib/energy_hard_negatives.py:33-203` - Hard negative mining implementation
5. **Loss Computation**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:651-752` - Where adversarial samples enter training
6. **Curriculum System**: `curriculum_config.py` and `diffusion_lib/curriculum_runtime.py` - Dynamic training parameters

## Iterative Investigation Plan

### Phase 1: Pipeline Connectivity Verification (IMMEDIATE)

#### Step 1.1: Verify Training Step Tracking
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:229`
**Issue**: The `self.training_step` is initialized but may not be incremented properly during training.

**Investigation**:
1. Check if `self.training_step` is incremented in the training loop
2. Verify that `enhanced_corruption_step` receives correct training step values
3. Add logging to track training step progression
4. **Expected Finding**: Training step may be stuck at 0, causing curriculum to never advance

**Fix Action**: Ensure `self.training_step` is incremented in the trainer's training loop

#### Step 1.2: Verify Enhanced Corruption Step Call
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:708`
**Issue**: The `enhanced_corruption_step` method may not be called with correct parameters.

**Investigation**:
1. Add logging to verify `enhanced_corruption_step` is called during training
2. Check parameter values passed (inp, x_start, t, mask, data_cond)
3. Verify return values are used correctly in loss computation
4. **Expected Finding**: Method may not be called when expected or parameters may be incorrect

**Fix Action**: Debug parameter flow and ensure proper method invocation

#### Step 1.3: Verify Energy Supervision Requirement
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:696-751`
**Issue**: Even with `supervise_energy_landscape=True`, energy losses may not be computed correctly.

**Investigation**:
1. Verify that both `use_adversarial_corruption=True` AND `supervise_energy_landscape=True`
2. Check that energy computation path is executed when both flags are set
3. Verify energy losses are added to the total loss correctly
4. **Expected Finding**: Energy supervision may be bypassed or computed incorrectly

**Fix Action**: Ensure energy loss computation executes and contributes to gradients

### Phase 2: Curriculum System Integration (IF PHASE 1 PASSES)

#### Step 2.1: Verify Curriculum Runtime Initialization
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:232-241`
**Issue**: `curriculum_runtime` may not be properly initialized or connected.

**Investigation**:
1. Verify `curriculum_config` is not None when ANM is used
2. Check that `CurriculumRuntime` is instantiated correctly
3. Verify curriculum parameters are passed correctly to runtime
4. **Expected Finding**: Curriculum may not be initialized even when specified

**Fix Action**: Ensure curriculum is initialized when provided

#### Step 2.2: Verify Curriculum Parameter Usage
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:766-773`
**Issue**: Temperature from curriculum may not be used in loss computation.

**Investigation**:
1. Check that `get_params` is called on curriculum runtime
2. Verify temperature is extracted and passed to `p_losses`
3. Check that temperature affects loss scaling as intended (line 714-721)
4. **Expected Finding**: Temperature may not be extracted or used properly

**Fix Action**: Ensure curriculum parameters are extracted and used

#### Step 2.3: Verify Corruption Type Selection
**File**: `diffusion_lib/adversarial_corruption.py:446-465`
**Issue**: Curriculum stage may not be passed correctly to corruption dispatcher.

**Investigation**:
1. Verify that curriculum stage is passed to `enhanced_corruption_step_v2`
2. Check that corruption type sampling uses stage ratios correctly
3. Verify that different corruption types are being selected over time
4. **Expected Finding**: May always select clean/gaussian corruption, never adversarial

**Fix Action**: Ensure stage is passed and corruption types vary according to curriculum

### Phase 3: Loss Computation Verification (IF PHASE 2 PASSES)

#### Step 3.1: Verify Adversarial Sample Generation
**File**: `diffusion_lib/adversarial_corruption.py:100-235`
**Issue**: Adversarial corruption may not generate meaningful adversarial samples.

**Investigation**:
1. Log energy values of generated adversarial samples vs clean samples
2. Verify that adversarial samples have lower energy than clean samples
3. Check that gradient computation succeeds without errors
4. **Expected Finding**: Adversarial samples may have same/higher energy than clean samples

**Fix Action**: Debug adversarial sample generation to ensure lower energy

#### Step 3.2: Verify Energy Loss Computation
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:726-747`
**Issue**: Energy loss computation may not create meaningful learning signal.

**Investigation**:
1. Log energy values for both real and fake samples
2. Verify cross-entropy loss computation is correct
3. Check that energy differences create meaningful gradients
4. **Expected Finding**: Energy differences may be too small or wrong sign

**Fix Action**: Ensure energy loss provides meaningful learning signal

#### Step 3.3: Verify Loss Scaling and Integration
**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:746-748`
**Issue**: Energy loss may not be scaled properly relative to MSE loss.

**Investigation**:
1. Log MSE loss and energy loss magnitudes separately
2. Verify that loss scaling creates balanced gradients
3. Check that combined loss leads to meaningful parameter updates
4. **Expected Finding**: Loss scaling may be incorrect, energy loss too small/large

**Fix Action**: Tune loss scaling for balanced training

### Phase 4: Hard Negative Mining Verification (IF PHASE 3 PASSES)

#### Step 4.1: Verify Hard Negative Mining Parameters
**File**: `diffusion_lib/energy_hard_negatives.py:33-203`
**Issue**: HNM parameters may not be properly connected or used.

**Investigation**:
1. Verify `hnm_num_candidates`, `hnm_refinement_steps`, `hnm_lambda_weight` are passed correctly
2. Check that hard negative generation produces valid samples
3. Verify candidate selection logic chooses most deceptive samples
4. **Expected Finding**: HNM parameters may not be used or selection may be flawed

**Fix Action**: Ensure HNM parameters are used and selection works correctly

### Phase 5: Advanced Debugging (IF ALL PREVIOUS PHASES PASS)

#### Step 5.1: Gradient Flow Analysis
**Investigation**:
1. Check that gradients flow from energy loss back to model parameters
2. Verify that adversarial corruption gradients don't interfere with main gradients
3. Check for gradient explosion/vanishing in adversarial paths

#### Step 5.2: Training Dynamics Analysis
**Investigation**:
1. Log training dynamics with/without adversarial corruption
2. Compare loss curves, energy landscapes, and model outputs
3. Analyze if adversarial training is helping or hurting convergence

#### Step 5.3: Model Architecture Compatibility
**Investigation**:
1. Verify that the EBM architecture is compatible with adversarial training
2. Check that energy computation produces meaningful gradients
3. Verify that diffusion wrapper correctly interfaces with adversarial corruption

## Implementation Strategy

### Development Approach
1. **Add Comprehensive Logging**: Insert detailed logging at each pipeline stage
2. **Incremental Validation**: Test each phase before proceeding to next
3. **Baseline Comparison**: Always compare against clean training baseline
4. **Rollback Strategy**: Keep working version while debugging

### Success Criteria
- Adversarial corruption should consistently produce samples with lower energy than clean samples
- Energy loss should provide meaningful gradients that improve model performance
- Curriculum progression should be observable through training metrics
- Training with ANM should outperform baseline training on validation metrics

### Failure Modes to Watch For
1. **Silent Failures**: ANM appears to run but has no effect
2. **Parameter Disconnection**: Curriculum parameters not reaching corruption logic
3. **Training Step Issues**: Training progression not tracked correctly
4. **Loss Scaling Problems**: Energy loss dominated by MSE loss or vice versa

## Expected Timeline

- **Phase 1**: 1-2 hours (pipeline connectivity verification)
- **Phase 2**: 2-3 hours (curriculum system integration)
- **Phase 3**: 3-4 hours (loss computation verification)
- **Phase 4**: 2-3 hours (hard negative mining verification)
- **Phase 5**: 4-6 hours (advanced debugging if needed)

**Total Estimated Time**: 12-18 hours of focused debugging

## Adaptation Strategy

This plan should be executed iteratively. If issues are found in earlier phases, fix them immediately and re-run training to verify improvement before proceeding. The plan may need adjustment based on findings - for example, if Phase 1 reveals major connectivity issues, the later phases may become irrelevant until the fundamental problems are fixed.

The key is to work systematically from the ground up, ensuring each component works correctly before moving to more complex integration issues.