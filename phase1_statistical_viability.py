# File: /Users/mkrasnow/Desktop/energy-based-model-2/smoke_test_with_diagnostics.py
# Changes: Added hyperparameter sweep functionality for ANM model
#   - Added run_hyperparameter_sweep() method to test 9 ANM configurations
#   - Added evaluate_model_with_config() for evaluating specific hyperparameter configs  
#   - Added _print_sweep_summary() and _save_sweep_results() for analysis
#   - Modified get_result_dir() to support adv_steps specific directories
#   - Modified train_model() to accept custom hyperparameters
#   - Added --sweep, --dataset CLI arguments to enable sweep mode
#   - Sweep tests specific configurations targeting different energy bands

import os
import subprocess

subprocess.run(['rm', '-rf', 'energy-based-model-2'], check=False)
subprocess.run(['git', 'clone', 'https://github.com/mdkrasnow/energy-based-model-2.git'], check=True)
os.chdir('energy-based-model-2')

# Add the cloned repository to Python path
import sys
sys.path.insert(0, os.getcwd())

import os
import subprocess
import argparse
import json
from pathlib import Path
import time
import re
import torch
import glob
from checkpoint_manager import get_checkpoint_manager, ModelConfig
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Import model components for diagnostics
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Addition, Inverse, LowRankDataset
from diffusion_lib.adversarial_corruption import _adversarial_corruption

# Phase 1 specific imports
from statistical_utils import StatisticalAnalyzer, StatisticalResult
from phase1_configs import (
    PHASE1_CONFIGURATIONS, 
    PHASE1_EXPERIMENTAL_DESIGN,
    get_all_phase1_configs,
    generate_experiment_matrix,
    validate_phase1_configs
)


# Hyperparameters from the paper (Appendix A)
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 1000 
DIFFUSION_STEPS = 10
RANK = 20  # For 20x20 matrices

# Tasks to run
TASKS = ['inverse']

class ExperimentRunner:
    def __init__(self, base_dir='experiments'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.results = {}
        self.diagnostic_results = defaultdict(dict)  # Store diagnostic results
        self.sweep_results = []  # Store hyperparameter sweep results
    
    def _sample_batch(self, dataset, batch_size):
        """Helper method to sample a batch from a PyTorch dataset
        
        Args:
            dataset: PyTorch dataset with __getitem__ and __len__ methods
            batch_size: Number of samples to get
            
        Returns:
            Dictionary with 'x' (inputs) and 'y' (outputs) as torch tensors
        """
        indices = torch.randperm(len(dataset))[:batch_size]
        
        x_list = []
        y_list = []
        
        for idx in indices:
            x, y = dataset[idx.item()]
            x_list.append(torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
            y_list.append(torch.from_numpy(y) if isinstance(y, np.ndarray) else y)
        
        return {
            'x': torch.stack(x_list).float(),
            'y': torch.stack(y_list).float()
        }
        
    def get_result_dir(self, dataset, model_type='baseline', adv_steps=None, seed=None):
        """
        Get the results directory for a given dataset and model type.
        
        Args:
            dataset: Dataset name ('addition', 'inverse', 'lowrank')
            model_type: Model type - MUST be exactly 'anm' or 'baseline' 
            adv_steps: Number of adversarial steps (required for ANM, ignored for baseline)
            seed: Random seed for unique directory naming
            
        Raises:
            ValueError: If model_type is invalid or required parameters missing
        """
        # STRICT INPUT VALIDATION to prevent config_name/model_type confusion
        if model_type not in ['anm', 'baseline']:
            raise ValueError(
                f"Invalid model_type='{model_type}'. Must be exactly 'anm' or 'baseline'. "
                f"If you're seeing config names like 'ired_baseline', 'anm_complete', etc., "
                f"you need to derive model_type from config.use_anm instead."
            )
        
        if model_type == 'anm' and adv_steps is None:
            raise ValueError(
                f"model_type='anm' requires adv_steps parameter to be specified. "
                f"Cannot generate unique path for ANM model without adversarial steps."
            )
        
        if dataset not in ['addition', 'inverse', 'lowrank']:
            raise ValueError(f"Unknown dataset: '{dataset}'. Must be one of: addition, inverse, lowrank")
        
        base = f'results/ds_{dataset}/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        if model_type == 'anm':
            if adv_steps is not None:
                base += f'_anm_steps{adv_steps}'
            else:
                base += '_anm_curriculum'  # Default ANM
        
        # Add seed suffix for Phase 1 experiments to ensure unique training directories
        if seed is not None:
            base += f'_seed{seed}'
        
        return base

    def load_model_for_diagnostics(self, model_dir, device='cuda', train_steps=None):
        """Load model checkpoint for diagnostic purposes with robust prefix handling"""
        # Fix critical bug: Use dynamic milestone calculation instead of hardcoded model-20.pt
        if train_steps is None:
            train_steps = TRAIN_ITERATIONS
        save_interval = 1000
        final_milestone = train_steps // save_interval
        checkpoint_path = Path(model_dir) / f'model-{final_milestone}.pt'
        if not checkpoint_path.exists():
            return None
            
        device = device if torch.cuda.is_available() else 'cpu'
        
        # CHECKPOINT VERIFICATION: Verify which checkpoint is actually loaded
        checkpoint_manager = get_checkpoint_manager()
        # Note: We don't have full config here, so just verify file integrity
        verification = checkpoint_manager.verify_checkpoint_integrity(checkpoint_path, None)
        
        print(f"📁 [CHECKPOINT_LOAD] Loading: {checkpoint_path}")
        print(f"   File size: {verification.get('size_mb', 'unknown')} MB")
        print(f"   File hash: {verification.get('file_hash', 'unknown')}")
        print(f"   Valid: {verification.get('is_valid', False)}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get dataset dimensions
        if 'addition' in str(model_dir):
            dataset = Addition("train", RANK, False)
        elif 'inverse' in str(model_dir):
            dataset = Inverse("train", RANK, False)
        else:
            dataset = LowRankDataset("train", RANK, False)
        
        # Initialize model (using the same model from train.py)
        model = EBM(
            inp_dim=dataset.inp_dim,
            out_dim=dataset.out_dim,
        )
        model = DiffusionWrapper(model)
        
        # Load state dict - handle different checkpoint formats
        state_dict = None
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint and 'ema_model' in checkpoint['ema']:
            # Try EMA model if available
            state_dict = checkpoint['ema']['ema_model']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
        
        # Robust prefix handling
        if state_dict:
            model_state = model.state_dict()
            fixed_state_dict = {}
            
            # Get sample keys to detect prefix patterns
            sample_checkpoint_key = next(iter(state_dict.keys()))
            sample_model_key = next(iter(model_state.keys()))
            
            # Check for common prefixes that need removal
            prefixes_to_remove = ['model.', 'module.']
            prefix_removed = False
            
            for prefix in prefixes_to_remove:
                if sample_checkpoint_key.startswith(prefix) and not sample_model_key.startswith(prefix):
                    # Remove this prefix from all keys
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            new_key = k[len(prefix):]
                            # Only include keys that exist in the target model
                            if new_key in model_state:
                                fixed_state_dict[new_key] = v
                    prefix_removed = True
                    print(f"Info: Removed '{prefix}' prefix from checkpoint keys")
                    break
            
            if not prefix_removed:
                # No prefix issue, but still filter out non-model parameters
                # (like diffusion parameters that might be in the checkpoint)
                for k, v in state_dict.items():
                    # Skip diffusion-related parameters
                    if any(k.startswith(skip) for skip in ['betas', 'alphas', 'sqrt', 'log', 'posterior', 'loss', 'opt_step']):
                        continue
                    # Only include keys that exist in the model
                    if k in model_state:
                        fixed_state_dict[k] = v
            
            # Update state_dict with the fixed version
            state_dict = fixed_state_dict
        
        # Load the cleaned state dict
        missing_keys = []
        unexpected_keys = []
        
        if state_dict:
            # Try strict loading first
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                error_msg = str(e)
                # Parse missing and unexpected keys from error message
                if "Missing key(s)" in error_msg:
                    missing_match = re.search(r'Missing key\(s\) in state_dict: (.+?)(?:\. |$)', error_msg)
                    if missing_match:
                        missing_keys = [k.strip().strip('"') for k in missing_match.group(1).split(',')]
                
                if "Unexpected key(s)" in error_msg:
                    unexpected_match = re.search(r'Unexpected key\(s\) in state_dict: (.+?)(?:\. |$)', error_msg)
                    if unexpected_match:
                        unexpected_keys = [k.strip().strip('"') for k in unexpected_match.group(1).split(',')]
                
                # Log the issue and fall back to non-strict loading
                print(f"Warning: Checkpoint loading issues detected:")
                if missing_keys:
                    print(f"  Missing {len(missing_keys)} keys in model")
                if unexpected_keys:
                    print(f"  Found {len(unexpected_keys)} unexpected keys in checkpoint")
                print("  Loading with strict=False to continue...")
                
                # Load with strict=False as fallback
                model.load_state_dict(state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        
        # Try to extract ANM parameters from model directory name (if specific config was used)
        anm_adversarial_steps = 5  # Default
        
        if "_anm_steps" in str(model_dir):
            # Extract parameters from directory name like "_anm_steps20"
            import re
            steps_match = re.search(r'_anm_steps(\d+)', str(model_dir))
            if steps_match:
                anm_adversarial_steps = int(steps_match.group(1))
        
        # Setup diffusion with ANM parameters for proper adversarial corruption
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=DIFFUSION_STEPS,
            sampling_timesteps=DIFFUSION_STEPS,
            continuous=True,
            show_inference_tqdm=False,
            # Add ANM parameters to match training configuration
            use_adversarial_corruption=True,
            anm_adversarial_steps=anm_adversarial_steps,
            anm_warmup_steps=0,  # Not relevant for diagnostics
            sudoku=False,  # Required by DiffusionOps protocol
            shortest_path=False,  # Required by DiffusionOps protocol
        )
        
        # Move diffusion object to the same device as the model
        diffusion = diffusion.to(device)
        
        return model, diffusion, device, dataset
    
    def run_energy_diagnostics(self, model_dir, dataset='addition', num_batches=5):
        """Run energy distribution diagnostics on a trained model"""
        result = self.load_model_for_diagnostics(model_dir)
        if result is None:
            return None
            
        model, diffusion, device, data = result
        
        energies = {
            'clean': [],
            'ired_standard': [],
            'anm_adversarial': [],
            'gaussian_noise': []
        }
        
        print(f"  Running energy distribution analysis...")
        
        for _ in range(num_batches):
            # Get test batch - properly split into input and output
            batch = self._sample_batch(data, 256)
            x_clean = batch['x'].to(device)
            y_clean = batch['y'].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (len(x_clean),), device=device)
            
            # 1. Clean samples energy
            with torch.no_grad():
                energy_clean = diffusion.energy_score(x_clean, y_clean, t)
                energies['clean'].append(energy_clean.mean().item())
            
            # 2. Standard IRED corruption (corrupt the output y, not input x)
            noise = torch.randn_like(y_clean)
            alpha = 1.0 - (t.float() / diffusion.num_timesteps).view(-1, 1)
            y_ired = alpha * y_clean + (1 - alpha) * noise
            with torch.no_grad():
                energy_ired = diffusion.energy_score(x_clean, y_ired, t)
                energies['ired_standard'].append(energy_ired.mean().item())
            
            # 3. ANM adversarial corruption (simulated on output)
            y_anm = self._simulate_anm_output(x_clean, y_clean.clone(), t, diffusion, num_steps=5)
            with torch.no_grad():
                energy_anm = diffusion.energy_score(x_clean, y_anm, t)
                energies['anm_adversarial'].append(energy_anm.mean().item())
            
            # 4. Gaussian noise corruption (on output)
            y_gaussian = y_clean + 0.1 * torch.randn_like(y_clean)
            with torch.no_grad():
                energy_gaussian = diffusion.energy_score(x_clean, y_gaussian, t)
                energies['gaussian_noise'].append(energy_gaussian.mean().item())
        
        return energies

    def _simulate_anm_output(self, x_clean, y_clean, t, diffusion, num_steps=5):
        """Use actual ANM adversarial corruption instead of simulation"""
        
        # Use the actual adversarial corruption with real parameters
        # Extract real training parameters from diffusion object
        anm_adversarial_steps = getattr(diffusion, 'anm_adversarial_steps', num_steps)
        
        # Data scale check - only print once per batch
        if not hasattr(self, '_data_scale_printed'):
            print("=" * 60)
            print("USING ACTUAL ADVERSARIAL CORRUPTION:")
            print(f"Clean output norm: {y_clean.norm(dim=-1).mean().item():.6f}")
            print(f"Clean output range: {y_clean.min().item():.6f} to {y_clean.max().item():.6f}")
            print(f"ANM adversarial steps: {anm_adversarial_steps}")
            print("=" * 60)
            self._data_scale_printed = True
        
        # Call the actual adversarial corruption function
        # diffusion object should already implement the DiffusionOps protocol
        return _adversarial_corruption(
            ops=diffusion,
            inp=x_clean,
            x_start=y_clean,
            t=t,
            mask=None,  # No masking in diagnostic context
            data_cond=None,  # No conditioning in diagnostic context
            base_noise_scale=3.0  # Standard base noise scale
        )

    def run_comparative_diagnostics(self, baseline_dir, anm_dir, dataset='addition', config_name='default'):
        """Critical test: Direct comparison on same batch"""
        baseline_result = self.load_model_for_diagnostics(baseline_dir)
        anm_result = self.load_model_for_diagnostics(anm_dir)

        if baseline_result is None or anm_result is None:
            return None
            
        _, baseline_diffusion, device, data = baseline_result
        _, anm_diffusion, _, _ = anm_result
        
        print(f"  Running comparative analysis...")
        
        # Get test batch - properly split into input and output
        batch = self._sample_batch(data, 100)
        x_clean = batch['x'].to(device)
        y_clean = batch['y'].to(device)
        
        t = torch.randint(0, baseline_diffusion.num_timesteps, (len(x_clean),), device=device)
        
        # Generate negatives using both methods (on output y)
        noise = torch.randn_like(y_clean)
        alpha = 1.0 - (t.float() / baseline_diffusion.num_timesteps).view(-1, 1)
        y_ired = alpha * y_clean + (1 - alpha) * noise
        y_anm = self._simulate_anm_output(x_clean, y_clean.clone(), t, anm_diffusion, num_steps=10)
        
        # Compute energies
        with torch.no_grad():
            energy_ired_baseline = baseline_diffusion.energy_score(x_clean, y_ired, t).mean().item()
            energy_anm_model = anm_diffusion.energy_score(x_clean, y_anm, t).mean().item()
        
        # Compute distances
        dist_ired = F.mse_loss(y_ired, y_clean).item()
        dist_anm = F.mse_loss(y_anm, y_clean).item()
        
        return {
            'config_name': config_name,
            'energy_ired': energy_ired_baseline,
            'energy_anm': energy_anm_model,
            'distance_ired': dist_ired,
            'distance_anm': dist_anm,
            'energy_ratio': energy_anm_model / (energy_ired_baseline + 1e-8),
            'energy_gap_percent': ((energy_anm_model - energy_ired_baseline) / abs(energy_ired_baseline)) * 100
        }
    
    def train_model(self, dataset, model_type='baseline', force_retrain=False, 
                   adv_steps=None, train_steps=None, seed=None):
        """Train a model for a specific dataset and model type
        
        Args:
            dataset: Dataset name
            model_type: One of 'baseline', 'anm' (which always uses curriculum)
            force_retrain: Force retraining even if model exists
            adv_steps: ANM adversarial steps override (for hyperparameter search)
            train_steps: Training steps override (default: TRAIN_ITERATIONS)
            seed: Random seed for Phase 1 experiments (creates unique directories)
        """
        result_dir = self.get_result_dir(dataset, model_type, adv_steps, seed)
        actual_train_steps = train_steps if train_steps is not None else TRAIN_ITERATIONS
        
        # Check if model already exists
        if not force_retrain and os.path.exists(f'{result_dir}/model-1.pt'):
            print(f"\n{'='*80}")
            print(f"Model for {dataset} ({model_type}) already exists. Skipping training.")
            print(f"Use --force to retrain.")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            
            # Run diagnostics on existing model
            print(f"Running diagnostics on existing {model_type} model...")
            energies = self.run_energy_diagnostics(result_dir, dataset)
            if energies:
                self.diagnostic_results[f'{dataset}_{model_type}']['energies'] = energies
                self._print_energy_summary(energies, model_type)
            
            return True
            
        print(f"\n{'='*80}")
        print(f"Training IRED ({model_type.upper()}) on {dataset.upper()} task")
        print(f"{'='*80}")
        print(f"Model Type: {model_type}")
        if adv_steps is not None:
            print(f"ANM Hyperparameters: adversarial_steps={adv_steps}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Training iterations: {actual_train_steps}")
        print(f"Diffusion steps: {DIFFUSION_STEPS}")
        print(f"Matrix rank: {RANK}")
        print(f"Result directory: {result_dir}")
        
        if model_type == 'anm':
            print(f"\nANM with AGGRESSIVE Curriculum Schedule (% of {TRAIN_ITERATIONS} steps):")
            print(f"  Warmup (0-10%): 100% clean, 0% adversarial")
            print(f"  Rapid Introduction (10-25%): 50% clean, 40% adversarial, 10% gaussian")
            print(f"  Aggressive Ramp (25-50%): 20% clean, 70% adversarial, 10% gaussian")
            print(f"  High Intensity (50-80%): 10% clean, 85% adversarial, 5% gaussian")
            print(f"  Extreme Hardening (80-100%): 5% clean, 90% adversarial, 5% gaussian")
            
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        # Build command
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(actual_train_steps),
        ]
        
        # Add seed for reproducible Phase 1 experiments
        if seed is not None:
            cmd.extend(['--seed', str(seed)])
        
        # Add model-specific parameters
        if model_type == 'anm':
            # Use custom hyperparameters if provided
            anm_adv_steps = adv_steps if adv_steps is not None else 5
            
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', str(anm_adv_steps),
                '--anm-temperature', '1.0',
                '--anm-clean-ratio', '0.1',
                '--anm-adversarial-ratio', '0.8',
                '--anm-gaussian-ratio', '0.1',
                # ANM now always uses curriculum, no need for --use-curriculum flag
            ])
        
        # Run training with real-time output
        try:
            start_time = time.time()
            
            # Use subprocess.Popen for real-time output with flushing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Display output line by line as it comes
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    sys.stdout.flush()
            
            # Wait for process to complete
            result = process.wait()
            elapsed = time.time() - start_time
            
            if result == 0:
                print(f"\n{'='*80}")
                print(f"Training completed for {dataset} ({model_type}) in {elapsed/60:.2f} minutes")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                # Run diagnostics immediately after training
                print(f"Running diagnostics on newly trained {model_type} model...")
                energies = self.run_energy_diagnostics(result_dir, dataset)
                if energies:
                    self.diagnostic_results[f'{dataset}_{model_type}']['energies'] = energies
                    self._print_energy_summary(energies, model_type)
                
                return True
            else:
                print(f"\n{'='*80}")
                print(f"ERROR: Training failed for {dataset} ({model_type}) with exit code {result}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return False
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Training failed for {dataset} ({model_type}): {e}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return False
    
    def _print_energy_summary(self, energies, model_type):
        """Print energy distribution summary"""
        print(f"\n  Energy Statistics for {model_type.upper()} model:")
        print("  " + "-" * 50)
        for corruption_type, values in energies.items():
            mean_energy = np.mean(values)
            std_energy = np.std(values)
            print(f"  {corruption_type:20s}: {mean_energy:.4f} ± {std_energy:.4f}")
        
        # Key insight for ANM
        if model_type == 'anm':
            mean_ired = np.mean(energies['ired_standard'])
            mean_anm = np.mean(energies['anm_adversarial'])
            improvement = ((mean_anm - mean_ired) / abs(mean_ired)) * 100
            
            print("\n  " + "="*50)
            print("  ANM DIAGNOSTIC:")
            if abs(improvement) < 5:
                print("  ❌ ANM energies ≈ IRED energies → ANM is REDUNDANT")
            elif improvement < -10:
                print("  ⚠️  ANM energies < IRED energies → ANM is TOO WEAK")
            elif improvement > 50:
                print("  ⚠️  ANM energies >> IRED energies → ANM may be OFF-MANIFOLD")
            else:
                print(f"  ✓ ANM provides {improvement:.1f}% energy increase over IRED")
            print("  " + "="*50 + "\n")
    
    def run_hyperparameter_sweep(self, dataset='addition', force_retrain=False):
        """Run systematic hyperparameter sweep for ANM adversarial steps"""
        
        # Define adversarial steps configurations to test
        # Since epsilon and distance_penalty are no longer used, focus on adversarial steps
        adv_steps_configs = [
            {'name': 'low_steps', 'adv_steps': 5},
            {'name': 'med_low_steps', 'adv_steps': 15}, 
            {'name': 'medium_steps', 'adv_steps': 25},
            {'name': 'med_high_steps', 'adv_steps': 45},
            {'name': 'high_steps', 'adv_steps': 65},
            {'name': 'very_high_steps', 'adv_steps': 95},
            {'name': 'extreme_steps', 'adv_steps': 135},
        ]
        
        all_configs = adv_steps_configs
        train_steps = TRAIN_ITERATIONS
        
        print(f"\n{'#'*80}")
        print(f"# HYPERPARAMETER SWEEP FOR ANM - ADVERSARIAL STEPS")
        print(f"# Dataset: {dataset}")
        print(f"# Training steps: {train_steps} (~12 min each)")
        print(f"# Total configs: {len(all_configs)}")
        print(f"# Testing adversarial steps: {[c['adv_steps'] for c in all_configs]}")
        print(f"# Fixed parameters:")
        print(f"#   - Learning rate: {LEARNING_RATE}")
        print(f"#   - Temperature: 1.0")
        print(f"#   - Clean ratio: 0.1")
        print(f"#   - Adversarial ratio: 0.8")
        print(f"#   - Gaussian ratio: 0.1")
        print(f"{'#'*80}\n")
        
        # Train baseline once if needed
        baseline_dir = self.get_result_dir(dataset, 'baseline')
        if not os.path.exists(f'{baseline_dir}/model-1.pt') or force_retrain:
            print("Training baseline model first...")
            self.train_model(dataset, 'baseline', force_retrain, train_steps=train_steps)
        
        # --- NEW: Evaluate and record baseline (IRED) for reference ---
        print("\nCollecting baseline (IRED) reference metrics...")
        # Energy diagnostics on baseline
        baseline_energies = self.run_energy_diagnostics(baseline_dir, dataset)
        # In-distribution and OOD MSE for baseline
        baseline_mse_same = self.evaluate_model(dataset, 'baseline', ood=False)
        baseline_mse_harder = self.evaluate_model(dataset, 'baseline', ood=True)

        # Store baseline entry at the top of sweep_results
        sweep_results = [{
            'config': 'baseline',                 # sentinel config id
            'adv_steps': None,
            'energy_gap_percent': None,           # not applicable
            'energy_ratio': None,                 # not applicable
            'mse_same': baseline_mse_same,
            'mse_harder': baseline_mse_harder,
            'energies': baseline_energies,
            'comparative': None                   # not applicable
        }]
        self.sweep_results.append(sweep_results[0])
        
        # Run sweep
        total_configs = len(all_configs)
        config_num = 0
        
        for config in all_configs:
            config_num += 1
            # Extract parameters from config
            config_name = config['name']
            adv_steps = config['adv_steps']
            
            print(f"\n{'='*80}")
            print(f"CONFIG {config_num}/{total_configs}: {config_name}, adversarial_steps={adv_steps}")
            print(f"{'='*80}")
            
            # Train model with these hyperparameters
            success = self.train_model(
                dataset, 
                model_type='anm',
                force_retrain=force_retrain,
                adv_steps=adv_steps,
                train_steps=train_steps
            )
            
            if success:
                # Run diagnostics
                anm_dir = self.get_result_dir(dataset, 'anm', adv_steps)
                
                # Energy diagnostics
                energies = self.run_energy_diagnostics(anm_dir, dataset)
                
                # Comparative diagnostics
                comp_result = self.run_comparative_diagnostics(
                    baseline_dir, anm_dir, dataset, config_name
                )
                
                # Evaluate on test set
                mse_same = self.evaluate_model_with_config(
                    dataset, adv_steps, ood=False, train_steps=train_steps
                )
                mse_harder = self.evaluate_model_with_config(
                    dataset, adv_steps, ood=True, train_steps=train_steps
                )
                
                # Store results
                result = {
                    'config': config_name,
                    'adv_steps': adv_steps,
                    'energy_gap_percent': comp_result['energy_gap_percent'] if comp_result else None,
                    'energy_ratio': comp_result['energy_ratio'] if comp_result else None,
                    'mse_same': mse_same,
                    'mse_harder': mse_harder,
                    'energies': energies,
                    'comparative': comp_result
                }
                sweep_results.append(result)
                self.sweep_results.append(result)
                
                # Print quick summary
                if comp_result:
                    print(f"\n  ✓ Config complete:")
                    print(f"    Energy gap: {comp_result['energy_gap_percent']:+.1f}%")
                    print(f"    MSE (same): {mse_same:.4f}" if mse_same else "    MSE (same): N/A")
                    print(f"    MSE (harder): {mse_harder:.4f}" if mse_harder else "    MSE (harder): N/A")
            else:
                print("  ✗ Training failed for this config; skipping diagnostics/eval.")
        
        # Analyze and print sweep results
        self._print_sweep_summary(sweep_results, dataset)
        self._save_sweep_results(sweep_results, dataset, train_steps)
        
        return sweep_results
    
    def evaluate_model_with_config(self, dataset, adv_steps, ood=False, train_steps=None, seed=None):
        """Evaluate a model trained with specific hyperparameters"""
        result_dir = self.get_result_dir(dataset, 'anm', adv_steps, seed)
        actual_train_steps = train_steps if train_steps is not None else TRAIN_ITERATIONS
        
        # ENHANCED CHECKPOINT VERIFICATION: Log comprehensive checkpoint info
        checkpoint_path = f'{result_dir}/model-1.pt'
        print(f"\n[CHECKPOINT_DEBUG] Config: anm_steps{adv_steps}, Seed: {seed}")
        print(f"[CHECKPOINT_DEBUG] Expected path: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"[CHECKPOINT_DEBUG] ❌ Checkpoint does not exist: {checkpoint_path}")
            return None
        else:
            # Add comprehensive checkpoint verification
            checkpoint_manager = get_checkpoint_manager()
            
            # Create ModelConfig for this evaluation (with error handling)
            try:
                model_config = ModelConfig(
                    dataset=dataset,
                    model_type='anm',
                    anm_adversarial_steps=adv_steps,
                    seed=seed
                )
                verification = checkpoint_manager.verify_checkpoint_integrity(
                    Path(checkpoint_path), model_config
                )
            except Exception as e:
                print(f"[CHECKPOINT_DEBUG] ⚠️  Config validation failed: {e}")
                # Fallback to basic verification without config
                verification = checkpoint_manager.verify_checkpoint_integrity(
                    Path(checkpoint_path), None
                )
            
            print(f"[CHECKPOINT_DEBUG] ✅ Checkpoint verification:")
            print(f"   Path: {checkpoint_path}")
            print(f"   Size: {verification.get('size_mb', 'unknown')} MB")
            print(f"   Hash: {verification.get('file_hash', 'unknown')}")
            print(f"   Modified: {verification.get('modified_timestamp', 'unknown')}")
            print(f"   Valid: {verification.get('is_valid', False)}")
        
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(actual_train_steps),
            '--load-milestone', '1',
            '--evaluate',
            '--use-anm',
            '--anm-adversarial-steps', str(adv_steps),
        ]
        if seed is not None:
            cmd.extend(['--seed', str(seed)])
        
        if ood:
            cmd.append('--ood')
        
        try:
            output_lines = []
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
            
            result = process.wait()
            
            if result == 0:
                output_text = ''.join(output_lines)
                mse = self._parse_mse_from_output(output_text, '')
                return mse
            
        except Exception:
            pass
        
        return None
    
    def _print_sweep_summary(self, sweep_results, dataset):
        """Print summary of hyperparameter sweep"""
        print(f"\n{'#'*80}")
        print(f"# HYPERPARAMETER SWEEP RESULTS - {dataset.upper()}")
        print(f"{'#'*80}\n")
        
        # Sort by energy gap (descending)
        sorted_results = sorted(sweep_results, key=lambda x: x['energy_gap_percent'] or -999, reverse=True)
        
        print(f"{'Config':<20s} {'Steps':>6s} {'Energy Gap':>12s} {'MSE (same)':>12s} {'MSE (harder)':>13s} {'Status':>10s}")
        print("-" * 85)
        
        # Print Baseline row first (if present)
        baseline = next((r for r in sweep_results if r.get('config') == 'baseline'), None)
        if baseline is not None:
            mse_same_str = f"{baseline['mse_same']:.4f}" if baseline['mse_same'] is not None else "N/A"
            mse_harder_str = f"{baseline['mse_harder']:.4f}" if baseline['mse_harder'] is not None else "N/A"
            print(f"{'BASELINE (IRED)':<20s} {'-':>6s} {'-':>12s} {mse_same_str:>12s} {mse_harder_str:>13s} {'REF':>10s}")
        
        # Then print ANM configs sorted by energy gap
        for r in [rr for rr in sorted_results if rr.get('config') != 'baseline']:
            gap = r['energy_gap_percent']
            gap_str = f"{gap:+.1f}%" if gap is not None else "N/A"
            mse_same_str = f"{r['mse_same']:.4f}" if r['mse_same'] is not None else "N/A"
            mse_harder_str = f"{r['mse_harder']:.4f}" if r['mse_harder'] is not None else "N/A"
            
            # Status based on energy gap
            if gap is None:
                status = "FAILED"
            elif gap < 1:
                status = "❌ WEAK"
            elif gap < 5:
                status = "⚠️  LOW"
            elif gap > 20:
                status = "⚠️  HIGH"
            else:
                status = "✓ GOOD"
            
            print(f"{r['config']:<20s} {r['adv_steps']:>6d} {gap_str:>12s} {mse_same_str:>12s} {mse_harder_str:>13s} {status:>10s}")
        
        # Highlight best configs
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Filter configs with good energy gaps (5-20%)
        good_configs = [r for r in sorted_results if r['energy_gap_percent'] and 5 <= r['energy_gap_percent'] <= 20]
        
        if good_configs:
            print("\n✓ GOOD CONFIGURATIONS (energy gap 5-20%):")
            for r in good_configs[:3]:  # Top 3
                print(f"  {r['config']}: steps={r['adv_steps']}")
                print(f"    Energy gap: {r['energy_gap_percent']:+.1f}%")
                if r['mse_harder']:
                    print(f"    Test MSE: {r['mse_harder']:.4f}")
            
            print("\n→ Run these configs at 50k steps to validate:")
            for r in good_configs[:2]:  # Top 2
                print(f"   python train.py --use-anm --anm-adversarial-steps {r['adv_steps']} --train-steps 50000")
        else:
            print("\n❌ NO GOOD CONFIGURATIONS FOUND")
            print("   All configs have either too weak (<5%) or too strong (>20%) energy gaps.")
            print("   Consider:")
            print("   - Testing higher adversarial steps: [150, 200, 300]")
            print("   - Testing lower adversarial steps: [2, 3, 10]")
            print("   - Adjusting curriculum parameters")
        
        print(f"\n{'='*80}\n")
    
    def _save_sweep_results(self, sweep_results, dataset, train_steps):
        """Save sweep results to JSON"""
        results_file = self.base_dir / f'hyperparameter_sweep_{dataset}_{train_steps}steps.json'
        
        data = {
            'metadata': {
                'dataset': dataset,
                'train_steps': train_steps,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'diffusion_steps': DIFFUSION_STEPS,
                'config_type': 'specific_configurations',
                'total_configs': len(sweep_results) - 1,  # Exclude baseline
                'config_categories': {
                    'very_negative': 5,
                    'midpoint': 1, 
                    'positive': 4
                },
                'includes_baseline': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'sweep_results': sweep_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Sweep results saved to: {results_file}")
    
    def evaluate_model(self, dataset, model_type='baseline', ood=False, seed=None):
        """Evaluate a trained model on same or harder difficulty"""
        result_dir = self.get_result_dir(dataset, model_type, seed=seed)
        
        # DEBUG: Log checkpoint path being used
        expected_model_path = f'{result_dir}/model-1.pt'
        print(f"\n[CHECKPOINT_DEBUG] Config: {model_type}, Seed: {seed}, Path: {expected_model_path}")
        
        # Enhanced fail-fast validation with actionable error messages
        if not os.path.exists(expected_model_path):
            print(f"[CHECKPOINT_DEBUG] ❌ Checkpoint does not exist: {expected_model_path}")
            print(f"\n{'='*80}")
            print(f"ERROR: No trained model found for {dataset} ({model_type})")
            print(f"Expected location: {expected_model_path}")
            
            # Scan for similar model paths to help debug
            results_base = os.path.dirname(result_dir)
            if os.path.exists(results_base):
                similar_paths = glob.glob(f'{results_base}/*/model-1.pt')
                if similar_paths:
                    print(f"\nSimilar models found:")
                    for path in similar_paths[:5]:  # Show max 5 to avoid clutter
                        rel_path = os.path.relpath(path, results_base)
                        print(f"  • {rel_path}")
                    
                    print(f"\nLikely issues to check:")
                    print(f"  - Verify --seed parameter matches training (current: seed={seed})")
                    print(f"  - Check if model_type matches training (current: {model_type})")
                    print(f"  - Ensure same --train-steps used for training")
                else:
                    print(f"\nNo model checkpoints found in {results_base}")
                    print(f"You may need to train the model first:")
                    if model_type == 'baseline':
                        print(f"  python train.py --dataset {dataset} --train-steps 1000" + 
                              (f" --seed {seed}" if seed is not None else ""))
                    else:
                        print(f"  python train.py --dataset {dataset} --use-anm --train-steps 1000" +
                              (f" --seed {seed}" if seed is not None else ""))
            else:
                print(f"\nResults directory does not exist: {results_base}")
                print(f"No models have been trained for dataset '{dataset}' yet.")
            
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return None
        
        difficulty = "Harder Difficulty (OOD)" if ood else "Same Difficulty"
        print(f"\n{'='*80}")
        print(f"Evaluating IRED ({model_type.upper()}) on {dataset.upper()} - {difficulty}")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        # Build command
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(TRAIN_ITERATIONS),  # Pass for consistency
            '--load-milestone', '1',
            '--evaluate',
        ]
        
        # Add seed for reproducible Phase 1 experiments (matches train_model pattern)
        if seed is not None:
            cmd.extend(['--seed', str(seed)])
        
        # Add model-specific parameters for evaluation
        if model_type == 'anm':
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', '5',
                # ANM now always uses curriculum, no need for --use-curriculum flag
            ])
        
        if ood:
            cmd.append('--ood')
        
        # Run evaluation with real-time output
        try:
            # Collect output for MSE parsing while also displaying it
            output_lines = []
            
            # Use subprocess.Popen for real-time output with flushing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Display output line by line as it comes
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    sys.stdout.flush()
                    output_lines.append(line)
            
            # Wait for process to complete
            result = process.wait()
            
            if result == 0:
                # Parse output to extract MSE
                output_text = ''.join(output_lines)
                mse = self._parse_mse_from_output(output_text, '')
                
                print(f"\n{'='*80}")
                print(f"Evaluation completed for {dataset} ({model_type}) - {difficulty}")
                if mse is not None:
                    print(f"MSE: {mse:.4f}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                return mse
            else:
                print(f"\n{'='*80}")
                print(f"ERROR: Evaluation failed for {dataset} ({model_type}) - {difficulty} with exit code {result}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return None
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Evaluation failed for {dataset} ({model_type}) - {difficulty}: {e}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return None
    
    def _parse_mse_from_output(self, stdout, stderr):
        """Parse MSE from training/evaluation output"""
        output = stdout + stderr
        lines = output.split('\n')
        
        # Look for validation result tables with MSE values
        mse_value = None
        for i, line in enumerate(lines):
            # Look for the specific pattern of mse in a table
            if line.startswith('mse') and '  ' in line:
                # This looks like a table row with MSE
                parts = line.split()
                if len(parts) >= 2 and parts[0] == 'mse':
                    try:
                        mse_value = float(parts[1])
                        # Continue searching to find the last MSE value (most recent)
                    except (ValueError, IndexError):
                        pass
        
        # If we didn't find MSE in table format, try alternative formats
        if mse_value is None:
            # Look for patterns like "mse_error  0.635722"
            for line in lines:
                if 'mse_error' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'mse' in part.lower() and i + 1 < len(parts):
                            try:
                                mse_value = float(parts[i + 1])
                            except ValueError:
                                pass
        
        return mse_value
    
    def train_all(self, force_retrain=False):
        """Train all models (baseline and ANM with curriculum)"""
        print(f"\n{'#'*80}")
        print(f"# TRAINING ALL CONTINUOUS TASKS WITH DIAGNOSTICS")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"# Model Types: baseline, anm (with curriculum)")
        print(f"# Training Steps: {TRAIN_ITERATIONS}")
        print(f"# ANM uses AGGRESSIVE curriculum (predefined from curriculum_config.py)")
        print(f"# Diagnostics will run automatically after each training")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        success = {}
        model_types = ['baseline', 'anm']
        
        for dataset in TASKS:
            for model_type in model_types:
                key = f"{dataset}_{model_type}"
                success[key] = self.train_model(dataset, model_type, force_retrain)
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING SUMMARY")
        print(f"{'#'*80}")
        for dataset in TASKS:
            print(f"\n{dataset.upper()}:")
            for model_type in model_types:
                key = f"{dataset}_{model_type}"
                status_str = "✓ SUCCESS" if success.get(key, False) else "✗ FAILED"
                model_desc = "ANM+Curriculum" if model_type == 'anm' else model_type
                print(f"  {model_desc:20s}: {status_str}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        return all(success.values())
    
    def evaluate_all(self):
        """Evaluate all models on both same and harder difficulty"""
        print(f"\n{'#'*80}")
        print(f"# EVALUATING ALL CONTINUOUS TASKS")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"# Model Types: baseline, anm (with curriculum)")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        results = {}
        model_types = ['baseline', 'anm']
        
        for dataset in TASKS:
            results[dataset] = {}
            for model_type in model_types:
                results[dataset][model_type] = {
                    'same_difficulty': self.evaluate_model(dataset, model_type, ood=False),
                    'harder_difficulty': self.evaluate_model(dataset, model_type, ood=True)
                }
        
        self.results = results
        
        # Run comparative diagnostics after evaluation
        print(f"\n{'#'*80}")
        print(f"# RUNNING COMPARATIVE DIAGNOSTICS")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            baseline_dir = self.get_result_dir(dataset, 'baseline')
            anm_dir = self.get_result_dir(dataset, 'anm')
            
            comp_result = self.run_comparative_diagnostics(baseline_dir, anm_dir, dataset)
            if comp_result:
                self.diagnostic_results[f'{dataset}_comparative'] = comp_result
        
        self._print_results_table()
        self._save_results()
        
        return results
    
    def _print_results_table(self):
        """Print results in a comparison table format"""
        print(f"\n{'#'*80}")
        print(f"# RESULTS COMPARISON TABLE")
        print(f"# Training Steps: {TRAIN_ITERATIONS}")
        print(f"{'#'*80}\n")
        
        # Print header
        print(f"{'Task':<20s} {'Method':<25s} {'Same Difficulty':>15s} {'Harder Difficulty':>17s}")
        print(f"{'-'*20} {'-'*25} {'-'*15} {'-'*17}")
        
        # Task name mapping for display
        task_display = {
            'addition': 'Addition',
            'lowrank': 'Matrix Completion',
            'inverse': 'Matrix Inverse'
        }
        
        # Print results for each task
        for dataset in TASKS:
            task_name = task_display.get(dataset, dataset)
            
            # Baseline
            baseline_same = self.results.get(dataset, {}).get('baseline', {}).get('same_difficulty')
            baseline_harder = self.results.get(dataset, {}).get('baseline', {}).get('harder_difficulty')
            baseline_same_str = f"{baseline_same:.4f}" if baseline_same is not None else "N/A"
            baseline_harder_str = f"{baseline_harder:.4f}" if baseline_harder is not None else "N/A"
            print(f"{task_name:<20s} {'IRED (baseline)':<25s} {baseline_same_str:>15s} {baseline_harder_str:>17s}")
            
            # ANM with curriculum
            anm_same = self.results.get(dataset, {}).get('anm', {}).get('same_difficulty')
            anm_harder = self.results.get(dataset, {}).get('anm', {}).get('harder_difficulty')
            anm_same_str = f"{anm_same:.4f}" if anm_same is not None else "N/A"
            anm_harder_str = f"{anm_harder:.4f}" if anm_harder is not None else "N/A"
            print(f"{'':<20s} {'IRED + ANM (curriculum)':<25s} {anm_same_str:>15s} {anm_harder_str:>17s}")
            
            print()  # Blank line between tasks
        
        # Print improvement percentages if baseline exists
        print(f"\n{'#'*80}")
        print(f"# RELATIVE IMPROVEMENTS vs BASELINE")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            task_name = task_display.get(dataset, dataset)
            baseline_same = self.results.get(dataset, {}).get('baseline', {}).get('same_difficulty')
            baseline_harder = self.results.get(dataset, {}).get('baseline', {}).get('harder_difficulty')
            
            if baseline_same and baseline_harder:
                print(f"{task_name}:")
                
                # ANM improvements
                anm_same = self.results.get(dataset, {}).get('anm', {}).get('same_difficulty')
                anm_harder = self.results.get(dataset, {}).get('anm', {}).get('harder_difficulty')
                if anm_same and anm_harder:
                    same_imp = ((baseline_same - anm_same) / baseline_same) * 100
                    harder_imp = ((baseline_harder - anm_harder) / baseline_harder) * 100
                    print(f"  ANM+Curriculum: {same_imp:+.1f}% (same), {harder_imp:+.1f}% (harder)")
        
        print(f"\n{'#'*80}")
        print(f"# Paper's reported IRED results for comparison:")
        print(f"{'#'*80}")
        print(f"{'Addition':<20s} {'IRED (paper)':<25s} {'0.0002':>15s} {'0.0020':>17s}")
        print(f"{'Matrix Completion':<20s} {'IRED (paper)':<25s} {'0.0174':>15s} {'0.2054':>17s}")
        print(f"{'Matrix Inverse':<20s} {'IRED (paper)':<25s} {'0.0095':>15s} {'0.2063':>17s}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
    
    def _save_results(self):
        """Save results to JSON file"""
        results_file = self.base_dir / 'continuous_results_with_anm_diagnostics.json'
        
        # Add metadata
        results_with_meta = {
            'metadata': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'train_iterations': TRAIN_ITERATIONS,
                'diffusion_steps': DIFFUSION_STEPS,
                'rank': RANK,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_types': ['baseline', 'anm'],
                'anm_note': 'ANM always uses curriculum learning',
                'curriculum': {
                    'type': 'AGGRESSIVE_CURRICULUM',
                    'warmup': '0-10% (100% clean, ε=0.0)',
                    'rapid_introduction': '10-25% (50% clean, 40% adversarial, ε=0.3)',
                    'aggressive_ramp': '25-50% (20% clean, 70% adversarial, ε=0.7)',
                    'high_intensity': '50-80% (10% clean, 85% adversarial, ε=1.0)',
                    'extreme_hardening': '80-100% (5% clean, 90% adversarial, ε=1.2)'
                }
            },
            'results': self.results,
            'diagnostics': dict(self.diagnostic_results)  # Include diagnostic results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        
        print(f"Results saved to: {results_file}\n")
        sys.stdout.flush()
    
    def plot_diagnostic_results(self):
        """Create inline visualizations of diagnostic results"""
        if not self.diagnostic_results:
            print("No diagnostic results to plot.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ANM Diagnostic Analysis', fontsize=16, y=1.02)
        
        # Plot 1: Energy distributions for baseline and ANM
        ax = axes[0, 0]
        for model_type in ['baseline', 'anm']:
            key = f'addition_{model_type}'
            if key in self.diagnostic_results and 'energies' in self.diagnostic_results[key]:
                energies = self.diagnostic_results[key]['energies']
                x = ['Clean', 'IRED', 'ANM', 'Gaussian']
                y = [np.mean(energies[k]) for k in ['clean', 'ired_standard', 'anm_adversarial', 'gaussian_noise']]
                label = 'ANM Model' if model_type == 'anm' else 'Baseline Model'
                ax.plot(x, y, marker='o', label=label, linewidth=2)
        
        ax.set_ylabel('Mean Energy')
        ax.set_title('Energy Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Comparative analysis
        ax = axes[0, 1]
        comp_key = 'addition_comparative'
        if comp_key in self.diagnostic_results:
            comp = self.diagnostic_results[comp_key]
            methods = ['IRED\nStandard', 'ANM\nAdversarial']
            energies = [comp['energy_ired'], comp['energy_anm']]
            colors = ['blue', 'red']
            bars = ax.bar(methods, energies, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, energies):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                        f'{val:.4f}', ha='center', va='bottom')
            
            ax.set_ylabel('Energy Score')
            ax.set_title('Direct Energy Comparison')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Energy ratio and verdict
        ax = axes[0, 2]
        ax.axis('off')
        
        if comp_key in self.diagnostic_results:
            comp = self.diagnostic_results[comp_key]
            ratio = comp['energy_ratio']
            improvement = (ratio - 1.0) * 100
            
            verdict_text = "📊 CRITICAL DIAGNOSTIC\n\n"
            if 0.95 <= ratio <= 1.05:
                verdict_text += "❌ ANM ≈ IRED\n\nANM is REDUNDANT!\n\n"
                verdict_text += "ANM finds same negatives\nas standard IRED."
                color = 'red'
            elif ratio < 0.95:
                verdict_text += "❌ ANM < IRED\n\nANM is HARMFUL!\n\n"
                verdict_text += f"ANM reduces energy by\n{-improvement:.1f}%"
                color = 'darkred'
            else:
                verdict_text += f"✓ ANM > IRED\n\n{improvement:.1f}% improvement\n\n"
                verdict_text += "ANM successfully finds\nharder negatives."
                color = 'green'
            
            verdict_text += f"\n\nEnergy Ratio: {ratio:.3f}"
            
            ax.text(0.5, 0.5, verdict_text, transform=ax.transAxes,
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    color=color, weight='bold')
        
        # Plot 4: Energy difference breakdown
        ax = axes[1, 0]
        if 'addition_anm' in self.diagnostic_results and 'energies' in self.diagnostic_results['addition_anm']:
            energies = self.diagnostic_results['addition_anm']['energies']
            
            # Calculate differences from IRED standard
            ired_mean = np.mean(energies['ired_standard'])
            diffs = {
                'Clean': np.mean(energies['clean']) - ired_mean,
                'ANM': np.mean(energies['anm_adversarial']) - ired_mean,
                'Gaussian': np.mean(energies['gaussian_noise']) - ired_mean
            }
            
            colors = ['green' if d < 0 else 'red' for d in diffs.values()]
            bars = ax.bar(diffs.keys(), diffs.values(), color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Energy Difference from IRED')
            ax.set_title('Energy Differences (ANM Model)')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Distance comparison
        ax = axes[1, 1]
        if comp_key in self.diagnostic_results:
            comp = self.diagnostic_results[comp_key]
            methods = ['IRED', 'ANM']
            distances = [comp['distance_ired'], comp['distance_anm']]
            colors = ['blue', 'red']
            bars = ax.bar(methods, distances, color=colors, alpha=0.7)
            
            for bar, val in zip(bars, distances):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                        f'{val:.4f}', ha='center', va='bottom')
            
            ax.set_ylabel('L2 Distance from Original')
            ax.set_title('Sample Movement Distance')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Recommendations
        ax = axes[1, 2]
        ax.axis('off')
        
        recommendations = "📋 RECOMMENDATIONS\n\n"
        if comp_key in self.diagnostic_results:
            ratio = self.diagnostic_results[comp_key]['energy_ratio']
            
            if 0.95 <= ratio <= 1.05:
                recommendations += "1. Increase --anm-adversarial-steps\n   to higher values (50-135)\n\n"
                recommendations += "2. Start ANM earlier\n   (5% instead of 10%)\n\n"
                recommendations += "3. Use more aggressive curriculum"
            elif ratio < 0.95:
                recommendations += "1. Check gradient sign\n   (should maximize energy)\n\n"
                recommendations += "2. Verify energy computation\n\n"
                recommendations += "3. Ensure model.eval() before ANM\n\n"
                recommendations += "4. Check for gradient clipping"
            else:
                recommendations += "✓ Current settings working\n\n"
                recommendations += "Consider:\n"
                recommendations += "• Longer training\n"
                recommendations += "• Fine-tune adversarial steps\n"
                recommendations += "• Adjust curriculum parameters"
        
        ax.text(0.1, 0.9, recommendations, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary with recommendations"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
        print("="*80)
        
        # Check if we have comparative results
        comp_key = 'addition_comparative'
        if comp_key in self.diagnostic_results:
            comp = self.diagnostic_results[comp_key]
            ratio = comp['energy_ratio']
            
            print("\n📊 FINAL VERDICT:")
            if 0.95 <= ratio <= 1.05:
                print("  ❌ ANM is REDUNDANT - provides no improvement over standard IRED\n")
                print("  CRITICAL FINDING: ANM adversarial samples have nearly identical")
                print("  energy to standard IRED negatives, making the additional computation")
                print("  unnecessary.\n")
                
                print("  RECOMMENDED FIXES:")
                print("  1. Increase anm_adversarial_steps to higher values (50-135)")
                print("  2. Start ANM earlier in curriculum (5% instead of 10%)")
                print("  3. Use more aggressive curriculum with higher ANM percentages")
                print("  4. Adjust curriculum temperature and ratios")
                
                print("\n  CODE CHANGES NEEDED:")
                print("  - In train.py: --anm-adversarial-steps 50")
                print("  - In curriculum_config.py: adjust stage timing and ratios")
                
            elif ratio < 0.95:
                print("  ❌ ANM is HARMFUL - actually degrading performance\n")
                degradation = (1.0 - ratio) * 100
                print(f"  CRITICAL FINDING: ANM is reducing energy by {degradation:.1f}%,")
                print("  making negatives EASIER instead of harder!\n")
                
                print("  RECOMMENDED FIXES:")
                print("  1. Check gradient sign (should maximize energy, not minimize)")
                print("  2. Verify energy function computation")
                print("  3. Check for gradient clipping issues")
                print("  4. Ensure model is in eval mode during ANM generation")
                
                print("\n  CODE CHANGES NEEDED:")
                print("  - In adversarial_corruption.py: verify gradient sign")
                print("  - Check model.eval() is called before ANM")
                print("  - Review energy_score implementation")
                
            else:
                improvement = (ratio - 1) * 100
                print(f"  ✓ ANM provides {improvement:.1f}% improvement\n")
                print("  SUCCESS: ANM is successfully finding harder negatives than IRED.\n")
                
                print("  OPTIMIZATION SUGGESTIONS:")
                print("  1. Current settings are working")
                print("  2. Could try increasing training steps")
                print("  3. Fine-tune adversarial steps")
                print("  4. Consider more aggressive curriculum")
        
        # Print energy analysis
        if 'addition_anm' in self.diagnostic_results and 'energies' in self.diagnostic_results['addition_anm']:
            energies = self.diagnostic_results['addition_anm']['energies']
            
            print("\n📈 ENERGY LANDSCAPE ANALYSIS:")
            print("  " + "-"*50)
            
            mean_ired = np.mean(energies['ired_standard'])
            mean_anm = np.mean(energies['anm_adversarial'])
            mean_gaussian = np.mean(energies['gaussian_noise'])
            mean_clean = np.mean(energies['clean'])
            
            print(f"  Clean samples:     {mean_clean:.4f} (baseline)")
            print(f"  Gaussian noise:    {mean_gaussian:.4f} ({(mean_gaussian/mean_clean-1)*100:+.1f}% vs clean)")
            print(f"  IRED standard:     {mean_ired:.4f} ({(mean_ired/mean_clean-1)*100:+.1f}% vs clean)")
            print(f"  ANM adversarial:   {mean_anm:.4f} ({(mean_anm/mean_clean-1)*100:+.1f}% vs clean)")
            
            print("\n  INTERPRETATION:")
            if mean_anm < mean_gaussian:
                print("  ⚠️  ANM samples have LOWER energy than gaussian noise!")
                print("     This indicates a serious problem with the adversarial process.")
            elif mean_anm < mean_ired * 1.05:
                print("  ⚠️  ANM barely improves over standard IRED.")
                print("     The adversarial process needs stronger hyperparameters.")
            else:
                print("  ✓  ANM successfully creates harder negatives than IRED.")
        
        print("\n" + "="*80)
        print("END OF DIAGNOSTIC SUMMARY")
        print("="*80 + "\n")

class Phase1ExperimentRunner(ExperimentRunner):
    """
    Extended ExperimentRunner for Phase 1 Statistical Viability Testing
    
    Implements the complete Phase 1 workflow:
    - 2 configurations × 5 seeds = 10 experiments
    - Statistical analysis with Bonferroni correction
    - Go/no-go decision based on p-values and effect sizes
    - CSV logging for experiment tracking
    """
    
    def __init__(self, base_dir='phase1_results'):
        super().__init__(base_dir)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.phase1_results = []
        self.csv_log_path = self.base_dir / 'phase1_experiment_log.csv'
        
        # Validate Phase 1 configurations on initialization
        validate_phase1_configs()
        
        # Create CSV header if file doesn't exist
        self._initialize_csv_logging()
    
    def _initialize_csv_logging(self):
        """Initialize CSV logging for Phase 1 experiments"""
        if not self.csv_log_path.exists():
            header = [
                'timestamp', 'experiment_id', 'config_name', 'dataset', 'seed', 
                'train_steps', 'mse_same_difficulty', 'mse_harder_difficulty',
                'training_time_minutes', 'model_path', 'success',
                'anm_adversarial_steps'
            ]
            
            with open(self.csv_log_path, 'w') as f:
                f.write(','.join(header) + '\n')
        
        print(f"📊 Phase 1 CSV logging: {self.csv_log_path}")
    
    def _log_experiment_to_csv(self, experiment_spec, results, training_time_minutes, success):
        """Log a single experiment to CSV"""
        config = experiment_spec['config']
        
        row = [
            time.strftime('%Y-%m-%d %H:%M:%S'),
            experiment_spec['experiment_id'],
            experiment_spec['config_name'],
            'addition',  # Fixed for Phase 1
            experiment_spec['seed'],
            experiment_spec['train_steps'],
            results.get('mse_same', '') if results else '',
            results.get('mse_harder', '') if results else '',
            f"{training_time_minutes:.2f}",
            # FIX: Use proper model_type ('anm'/'baseline') instead of config_name
            # This was causing all ANM experiments to log baseline paths
            self.get_result_dir('addition', 'anm' if config.use_anm else 'baseline', 
                              config.anm_adversarial_steps if config.use_anm else None,
                              experiment_spec['seed']),
            'success' if success else 'failed',
            config.anm_adversarial_steps if config.use_anm else ''
        ]
        
        with open(self.csv_log_path, 'a') as f:
            f.write(','.join(str(x) for x in row) + '\n')
    
    def train_model_with_phase1_config(self, dataset, config, seed, force_retrain=False):
        """
        Train a model with specific Phase 1 configuration
        
        Args:
            dataset: Dataset name (typically 'addition' for Phase 1)
            config: Phase1Config object
            seed: Random seed for this experiment
            force_retrain: Force retraining even if model exists
            
        Returns:
            (success, training_time_minutes, results)
        """
        start_time = time.time()
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"PHASE 1 EXPERIMENT: {config.name.upper()} (seed={seed})")
        print(f"{'='*80}")
        print(f"Description: {config.description}")
        print(f"Random seed: {seed}")
        
        if config.use_anm:
            print(f"ANM parameters:")
            print(f"  • Adversarial steps: {config.anm_adversarial_steps}")
        
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        # Use existing train_model method with appropriate parameters
        success = self.train_model(
            dataset=dataset,
            model_type='anm' if config.use_anm else 'baseline',
            force_retrain=force_retrain,
            adv_steps=config.anm_adversarial_steps if config.use_anm else None,
            train_steps=PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment'],
            seed=seed
        )
        
        training_time_minutes = (time.time() - start_time) / 60
        
        # Evaluate model if training succeeded
        results = {}
        if success:
            print(f"\n✅ Training completed in {training_time_minutes:.1f} minutes")
            print("Running evaluation...")
            
            # Evaluate on same and harder difficulty
            mse_same = self._evaluate_phase1_model(dataset, config, seed, ood=False)
            mse_harder = self._evaluate_phase1_model(dataset, config, seed, ood=True)
            
            results = {
                'mse_same': mse_same,
                'mse_harder': mse_harder
            }
            
            print(f"  MSE (same difficulty): {mse_same:.6f}" if mse_same is not None else "  MSE (same difficulty): N/A")
            print(f"  MSE (harder difficulty): {mse_harder:.6f}" if mse_harder is not None else "  MSE (harder difficulty): N/A")
        else:
            print(f"\n❌ Training failed after {training_time_minutes:.1f} minutes")
        
        return success, training_time_minutes, results
    
    
    def _evaluate_phase1_model(self, dataset, config, seed, ood=False):
        """Evaluate a Phase 1 model and return MSE"""
        print(f"\n[PHASE1_DEBUG] Evaluating config: {config.name}, ANM: {config.use_anm}, Steps: {getattr(config, 'anm_adversarial_steps', 'N/A')}, Seed: {seed}")
        
        # Use existing evaluation method
        if config.use_anm:
            return self.evaluate_model_with_config(
                dataset,
                adv_steps=config.anm_adversarial_steps,
                ood=ood,
                train_steps=PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment'],
                seed=seed
            )
        else:
            return self.evaluate_model(dataset, 'baseline', ood=ood, seed=seed)
    
    def run_phase1_viability_test(self, dataset='addition', force_retrain=False):
        """
        Main Phase 1 orchestration: Statistical viability testing
        
        Runs 2 configurations × 5 seeds = 10 experiments with statistical analysis
        
        Args:
            dataset: Dataset name (default: 'addition')  
            force_retrain: Force retraining even if models exist
            
        Returns:
            (go_decision, statistical_results, phase1_report)
        """
        print(f"\n{'#'*80}")
        print(f"# PHASE 1 STATISTICAL VIABILITY TESTING")
        print(f"# Dataset: {dataset}")
        print(f"# Total experiments: {PHASE1_EXPERIMENTAL_DESIGN['total_experiments']}")
        print(f"# Configurations: {PHASE1_EXPERIMENTAL_DESIGN['num_configs']}")
        print(f"# Seeds per config: {PHASE1_EXPERIMENTAL_DESIGN['seeds_per_config']}")
        print(f"{'#'*80}\n")
        
        # Generate complete experiment matrix
        experiments = generate_experiment_matrix()
        print(f"📋 Generated {len(experiments)} experiments")
        
        # Group experiments by configuration for organized execution
        config_groups = {}
        for exp in experiments:
            config_name = exp['config_name']
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(exp)
        
        # Execute experiments configuration by configuration
        all_experiment_results = {}
        total_start_time = time.time()
        
        for config_name, config_experiments in config_groups.items():
            print(f"\n🧪 TESTING CONFIGURATION: {config_name.upper()}")
            print(f"   Experiments: {len(config_experiments)}")
            
            config_results = []
            
            for exp in config_experiments:
                print(f"\n   → Experiment: {exp['experiment_id']}")
                
                success, training_time, results = self.train_model_with_phase1_config(
                    dataset=dataset,
                    config=exp['config'],
                    seed=exp['seed'],
                    force_retrain=force_retrain
                )
                
                # Log to CSV
                self._log_experiment_to_csv(exp, results, training_time, success)
                
                if success and results.get('mse_harder') is not None:
                    config_results.append(results['mse_harder'])
                    print(f"     ✅ MSE: {results['mse_harder']:.6f}")
                else:
                    print(f"     ❌ Experiment failed")
            
            all_experiment_results[config_name] = config_results
            
            print(f"\n   📊 {config_name} summary: {len(config_results)}/{len(config_experiments)} successful")
        
        total_time_hours = (time.time() - total_start_time) / 3600
        print(f"\n🕒 Total Phase 1 execution time: {total_time_hours:.2f} hours")
        
        # DEBUG: Print checkpoint path summary for all configurations
        print(f"\n{'='*80}")
        print("CHECKPOINT PATH VERIFICATION SUMMARY")
        print("="*80)
        configs = get_all_phase1_configs()
        seeds = PHASE1_EXPERIMENTAL_DESIGN['random_seeds']
        
        for config_name, config in configs.items():
            print(f"\n{config_name.upper()}:")
            if config.use_anm:
                adv_steps = config.anm_adversarial_steps
                for seed in seeds:
                    result_dir = self.get_result_dir('addition', 'anm', adv_steps, seed)
                    print(f"  Seed {seed}: {result_dir}/model-1.pt")
            else:
                for seed in seeds:
                    result_dir = self.get_result_dir('addition', 'baseline', None, seed)
                    print(f"  Seed {seed}: {result_dir}/model-1.pt")
        print("="*80)
        
        # Statistical analysis
        return self._analyze_phase1_results(all_experiment_results, dataset, total_time_hours)
    
    def _analyze_phase1_results(self, all_results, dataset, total_time_hours):
        """
        Perform statistical analysis of Phase 1 results
        
        Args:
            all_results: Dict mapping config_name -> list of MSE scores
            dataset: Dataset name
            total_time_hours: Total execution time
            
        Returns:
            (go_decision, statistical_results, phase1_report)
        """
        print(f"\n{'='*80}")
        print("PHASE 1 STATISTICAL ANALYSIS")
        print("="*80)
        
        # Ensure we have baseline results for comparison
        if 'ired_baseline' not in all_results or len(all_results['ired_baseline']) == 0:
            print("❌ ERROR: No baseline results available for comparison")
            return False, [], "Phase 1 failed: No baseline results"
        
        baseline_scores = all_results['ired_baseline']
        print(f"📊 Baseline (IRED) results: {len(baseline_scores)} runs")
        print(f"   MSE: {np.mean(baseline_scores):.6f} ± {np.std(baseline_scores):.6f}")
        
        # Analyze each non-baseline configuration
        statistical_results = []
        
        for config_name, scores in all_results.items():
            if config_name == 'ired_baseline' or len(scores) == 0:
                continue
                
            print(f"\n🔬 Analyzing {config_name}...")
            print(f"   Results: {len(scores)} runs")
            print(f"   MSE: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
            
            # Statistical analysis vs baseline
            result = self.statistical_analyzer.statistical_analysis_single_config(
                config_name=config_name,
                baseline_scores=baseline_scores,
                comparison_scores=scores,
                num_comparisons=len(all_results) - 1  # Exclude baseline from count
            )
            
            statistical_results.append(result)
            
            # Print immediate verdict
            if result.significant and result.effect_size_adequate:
                print(f"   ✅ VIABLE: p={result.p_value_corrected:.4f}, d={result.cohens_d:.3f}")
            else:
                print(f"   ❌ NOT VIABLE: p={result.p_value_corrected:.4f}, d={result.cohens_d:.3f}")
        
        # Generate statistical summary table
        if statistical_results:
            print(f"\n📋 STATISTICAL SUMMARY:")
            summary_table = self.statistical_analyzer.statistical_summary_table(statistical_results)
            print(summary_table.to_string(index=False))
        
        # Make go/no-go decision
        go_decision, rationale = self.statistical_analyzer.go_no_go_decision(statistical_results)
        
        print(f"\n{'='*80}")
        print("PHASE 1 DECISION")
        print("="*80)
        print(rationale)
        print("="*80)
        
        # Generate comprehensive report
        total_experiments = sum(len(scores) for scores in all_results.values())
        phase1_report = self.statistical_analyzer.generate_phase1_report(
            statistical_results, dataset, total_experiments, total_time_hours
        )
        
        # Save report to file
        report_path = self.base_dir / f'phase1_decision_report_{dataset}.md'
        with open(report_path, 'w') as f:
            f.write(phase1_report)
        print(f"\n📄 Phase 1 report saved: {report_path}")
        
        return go_decision, statistical_results, phase1_report

# Initialize runner with base directory
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1 Statistical Viability Testing for ANM')
    parser.add_argument('--base-dir', default='phase1_results',
                        help='Base directory for experiments and logs')
    parser.add_argument('--force', action='store_true', help='Force retrain even if models exist')
    parser.add_argument('--phase1', action='store_true', help='Run Phase 1 statistical viability testing')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep instead of full training')
    parser.add_argument('--dataset', default='addition', help='Dataset for testing')
    args = parser.parse_args()

    # Make base_dir absolute and ensure it exists
    import os
    base_dir = os.path.abspath(args.base_dir or 'phase1_results')
    os.makedirs(base_dir, exist_ok=True)
    try:
        test_path = os.path.join(base_dir, ".__write_test")
        with open(test_path, "w") as f:
            f.write("test\n")
        os.remove(test_path)
    except OSError as e:
        print(f"FATAL: Cannot write to base_dir={base_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.phase1:
        print("🚀 Starting Phase 1 Statistical Viability Testing")
        print("   This implements the optimal ANM algorithm from the final synthesis")
        print("   2 configurations × 5 seeds = 10 experiments with rigorous statistics\n")
        
        runner = Phase1ExperimentRunner(base_dir=base_dir)
        
        go_decision, statistical_results, phase1_report = runner.run_phase1_viability_test(
            dataset=args.dataset, 
            force_retrain=args.force
        )
        
        print(f"\n🏁 PHASE 1 COMPLETE")
        print(f"Decision: {'✅ GO TO PHASE 2' if go_decision else '❌ NO-GO - STOP'}")
        print(f"Configurations tested: {len(PHASE1_CONFIGURATIONS)}")
        if statistical_results:
            viable_count = sum(1 for r in statistical_results if r.significant and r.effect_size_adequate)
            print(f"Statistically viable configs: {viable_count}")
        print(f"\nSee full analysis in: {runner.base_dir}/phase1_decision_report_{args.dataset}.md")
    
    elif args.sweep:
        # Run hyperparameter sweep (original functionality)
        runner = ExperimentRunner(base_dir=args.base_dir)
        runner.run_hyperparameter_sweep(dataset=args.dataset, force_retrain=args.force)
    else:
        # Train all models (original functionality)
        runner = ExperimentRunner(base_dir=args.base_dir)
        success = runner.train_all(force_retrain=args.force)

        # Evaluate if training succeeded
        if success:
            # Evaluate all models (comparative diagnostics run automatically)
            results = runner.evaluate_all()
            
            # Plot diagnostic visualizations
            print("\n" + "#"*80)
            print("# DIAGNOSTIC VISUALIZATIONS")
            print("#"*80)
            runner.plot_diagnostic_results()
            
            # Print comprehensive diagnostic summary with recommendations
            runner.print_diagnostic_summary()
        else:
            print("\nSome training jobs failed. Skipping evaluation.")
            sys.stdout.flush()
