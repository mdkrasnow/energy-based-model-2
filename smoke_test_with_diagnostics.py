# File: /Users/mkrasnow/Desktop/energy-based-model-2/smoke_test_with_diagnostics.py
# Changes: Added hyperparameter sweep functionality for ANM model
#   - Added run_hyperparameter_sweep() method to test 9 ANM configurations
#   - Added evaluate_model_with_config() for evaluating specific hyperparameter configs  
#   - Added _print_sweep_summary() and _save_sweep_results() for analysis
#   - Modified get_result_dir() to support epsilon/adv_steps specific directories
#   - Modified train_model() to accept custom hyperparameters
#   - Added --sweep, --dataset CLI arguments to enable sweep mode
#   - Sweep tests epsilon=[0.1, 0.3, 0.5] × adv_steps=[5, 20, 50] at 20k steps

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


# Hyperparameters from the paper (Appendix A)
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 20000  
DIFFUSION_STEPS = 10
RANK = 20  # For 20x20 matrices
DISTANCE_PENALTY = 0.001

# Tasks to run
TASKS = ['addition']

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
        
    def get_result_dir(self, dataset, model_type='baseline', epsilon=None, adv_steps=None, distance_penalty=None):
        """Get the results directory for a given dataset and model type"""
        base = f'results/ds_{dataset}/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        if model_type == 'anm':
            if epsilon is not None and adv_steps is not None:
                base += f'_anm_eps{epsilon}_steps{adv_steps}'
                if distance_penalty is not None:
                    base += f'_dp{distance_penalty}'
            else:
                base += '_anm_curriculum'  # Default ANM
        return base

    def load_model_for_diagnostics(self, model_dir, device='cuda'):
        """Load model checkpoint for diagnostic purposes with robust prefix handling"""
        checkpoint_path = Path(model_dir) / 'model-20.pt'
        if not checkpoint_path.exists():
            return None
            
        device = device if torch.cuda.is_available() else 'cpu'
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
        
        # Setup diffusion
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=DIFFUSION_STEPS,
            sampling_timesteps=DIFFUSION_STEPS,
            continuous=True,
            show_inference_tqdm=False
        )
        
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

    def _simulate_anm_output(self, x_clean, y_clean, t, diffusion, num_steps=5, eps=0.1):
        """Simulate ANM adversarial corruption on output y given input x"""
        y_adv = y_clean.clone().requires_grad_(True)
        
        # Data scale check - only print once per batch
        if not hasattr(self, '_data_scale_printed'):
            print("=" * 60)
            print("DATA SCALE CHECK:")
            print(f"Clean output norm: {y_clean.norm(dim=-1).mean().item():.6f}")
            print(f"Clean output range: {y_clean.min().item():.6f} to {y_clean.max().item():.6f}")
            print(f"DISTANCE_PENALTY: {DISTANCE_PENALTY}")
            print(f"Relative distance penalty: {DISTANCE_PENALTY / y_clean.norm(dim=-1).mean().item():.3f} of mean norm")
            print("=" * 60)
            self._data_scale_printed = True

        for _ in range(num_steps):
            energy = diffusion.energy_score(x_clean, y_adv, t)
            grad = torch.autograd.grad(energy.sum(), y_adv)[0]

            with torch.no_grad():
                y_adv = y_adv + eps * grad.sign()
                # Constrain adversarial samples to stay within distance_penalty of clean
                movement = y_adv - y_clean
                movement_norm = torch.norm(movement, dim=-1, keepdim=True)
                max_movement = DISTANCE_PENALTY
                y_adv = y_clean + movement * torch.clamp(max_movement / (movement_norm + 1e-8), max=1.0)

            y_adv.requires_grad_(True)

        return y_adv.detach()

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
                   epsilon=None, adv_steps=None, train_steps=None, distance_penalty=None):
        """Train a model for a specific dataset and model type
        
        Args:
            dataset: Dataset name
            model_type: One of 'baseline', 'anm' (which always uses curriculum)
            force_retrain: Force retraining even if model exists
            epsilon: ANM epsilon override (for hyperparameter search)
            adv_steps: ANM adversarial steps override (for hyperparameter search)
            train_steps: Training steps override (default: TRAIN_ITERATIONS)
            distance_penalty: ANM distance penalty override (for hyperparameter search)
        """
        result_dir = self.get_result_dir(dataset, model_type, epsilon, adv_steps, distance_penalty)
        actual_train_steps = train_steps if train_steps is not None else TRAIN_ITERATIONS
        
        # Check if model already exists
        if not force_retrain and os.path.exists(f'{result_dir}/model-20.pt'):
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
        if epsilon is not None and adv_steps is not None:
            print(f"ANM Hyperparameters: epsilon={epsilon}, adversarial_steps={adv_steps}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Training iterations: {actual_train_steps}")
        print(f"Diffusion steps: {DIFFUSION_STEPS}")
        print(f"Matrix rank: {RANK}")
        print(f"Result directory: {result_dir}")
        
        if model_type == 'anm':
            print(f"\nANM with AGGRESSIVE Curriculum Schedule (% of {TRAIN_ITERATIONS} steps):")
            print(f"  Warmup (0-10%): 100% clean, 0% adversarial, ε=0.0")
            print(f"  Rapid Introduction (10-25%): 50% clean, 40% adversarial, 10% gaussian, ε=0.3")
            print(f"  Aggressive Ramp (25-50%): 20% clean, 70% adversarial, 10% gaussian, ε=0.7")
            print(f"  High Intensity (50-80%): 10% clean, 85% adversarial, 5% gaussian, ε=1.0")
            print(f"  Extreme Hardening (80-100%): 5% clean, 90% adversarial, 5% gaussian, ε=1.2")
            
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
        
        # Add model-specific parameters
        if model_type == 'anm':
            # Use custom hyperparameters if provided
            anm_adv_steps = adv_steps if adv_steps is not None else 5
            anm_epsilon = epsilon if epsilon is not None else 0.1
            anm_distance_penalty = distance_penalty if distance_penalty is not None else DISTANCE_PENALTY
            
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', str(anm_adv_steps),
                '--anm-epsilon', str(anm_epsilon),
                '--anm-distance-penalty', str(anm_distance_penalty),
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
        """Run systematic hyperparameter sweep for ANM"""
        
        # Define sweep parameters
        epsilon_values = [0.1, 0.3, 0.5]
        adv_steps_values = [5, 20, 50]
        distance_penalty_values = [0.01, 0.001]  # Test two distance penalties
        train_steps = 20000  # Shorter runs for sweep
        
        print(f"\n{'#'*80}")
        print(f"# HYPERPARAMETER SWEEP FOR ANM")
        print(f"# Dataset: {dataset}")
        print(f"# Epsilon values: {epsilon_values}")
        print(f"# Adversarial steps: {adv_steps_values}")
        print(f"# Distance penalties: {distance_penalty_values}")
        print(f"# Training steps: {train_steps} (~12 min each)")
        print(f"# Total configs: {len(epsilon_values) * len(adv_steps_values) * len(distance_penalty_values)}")
        print(f"# Fixed parameters:")
        print(f"#   - Learning rate: {LEARNING_RATE}")
        print(f"#   - Temperature: 1.0")
        print(f"#   - Clean ratio: 0.1")
        print(f"#   - Adversarial ratio: 0.8")
        print(f"#   - Gaussian ratio: 0.1")
        print(f"{'#'*80}\n")
        
        # Train baseline once if needed
        baseline_dir = self.get_result_dir(dataset, 'baseline')
        if not os.path.exists(f'{baseline_dir}/model-20.pt') or force_retrain:
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
            'epsilon': None,
            'adv_steps': None,
            'distance_penalty': None,
            'energy_gap_percent': None,           # not applicable
            'energy_ratio': None,                 # not applicable
            'mse_same': baseline_mse_same,
            'mse_harder': baseline_mse_harder,
            'energies': baseline_energies,
            'comparative': None                   # not applicable
        }]
        self.sweep_results.append(sweep_results[0])
        
        # Run sweep
        total_configs = len(epsilon_values) * len(adv_steps_values) * len(distance_penalty_values)
        config_num = 0
        
        for epsilon in epsilon_values:
            for adv_steps in adv_steps_values:
                for distance_penalty in distance_penalty_values:
                    config_num += 1
                    config_name = f"eps{epsilon}_steps{adv_steps}_dp{distance_penalty}"
                    
                    print(f"\n{'='*80}")
                    print(f"CONFIG {config_num}/{total_configs}: epsilon={epsilon}, adversarial_steps={adv_steps}, distance_penalty={distance_penalty}")
                    print(f"{'='*80}")
                    
                    # Train model with these hyperparameters
                    success = self.train_model(
                        dataset, 
                        model_type='anm',
                        force_retrain=force_retrain,
                        epsilon=epsilon,
                        adv_steps=adv_steps,
                        train_steps=train_steps,
                        distance_penalty=distance_penalty
                    )
                    
                    if success:
                        # Run diagnostics
                        anm_dir = self.get_result_dir(dataset, 'anm', epsilon, adv_steps, distance_penalty)
                        
                        # Energy diagnostics
                        energies = self.run_energy_diagnostics(anm_dir, dataset)
                        
                        # Comparative diagnostics
                        comp_result = self.run_comparative_diagnostics(
                            baseline_dir, anm_dir, dataset, config_name
                        )
                        
                        # Evaluate on test set
                        mse_same = self.evaluate_model_with_config(
                            dataset, epsilon, adv_steps, ood=False, train_steps=train_steps, distance_penalty=distance_penalty
                        )
                        mse_harder = self.evaluate_model_with_config(
                            dataset, epsilon, adv_steps, ood=True, train_steps=train_steps, distance_penalty=distance_penalty
                        )
                        
                        # Store results
                        result = {
                            'config': config_name,
                            'epsilon': epsilon,
                            'adv_steps': adv_steps,
                            'distance_penalty': distance_penalty,
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
    
    def evaluate_model_with_config(self, dataset, epsilon, adv_steps, ood=False, train_steps=None, distance_penalty=None):
        """Evaluate a model trained with specific hyperparameters"""
        result_dir = self.get_result_dir(dataset, 'anm', epsilon, adv_steps, distance_penalty)
        actual_train_steps = train_steps if train_steps is not None else TRAIN_ITERATIONS
        
        if not os.path.exists(f'{result_dir}/model-20.pt'):
            return None
        
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
            '--anm-epsilon', str(epsilon),
            '--anm-distance-penalty', str(distance_penalty if distance_penalty is not None else DISTANCE_PENALTY),
        ]
        
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
        
        print(f"{'Config':<20s} {'Epsilon':>8s} {'Steps':>6s} {'Dist Pen':>10s} {'Energy Gap':>12s} {'MSE (same)':>12s} {'MSE (harder)':>13s} {'Status':>10s}")
        print("-" * 115)
        
        # Print Baseline row first (if present)
        baseline = next((r for r in sweep_results if r.get('config') == 'baseline'), None)
        if baseline is not None:
            mse_same_str = f"{baseline['mse_same']:.4f}" if baseline['mse_same'] is not None else "N/A"
            mse_harder_str = f"{baseline['mse_harder']:.4f}" if baseline['mse_harder'] is not None else "N/A"
            print(f"{'BASELINE (IRED)':<20s} {'-':>8s} {'-':>6s} {'-':>10s} {'-':>12s} {mse_same_str:>12s} {mse_harder_str:>13s} {'REF':>10s}")
        
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
            
            dp_str = f"{r.get('distance_penalty', 'N/A'):>10}" if 'distance_penalty' in r else 'N/A'
            print(f"{r['config']:<20s} {r['epsilon']:>8.1f} {r['adv_steps']:>6d} {dp_str:>10s} {gap_str:>12s} {mse_same_str:>12s} {mse_harder_str:>13s} {status:>10s}")
        
        # Highlight best configs
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Filter configs with good energy gaps (5-20%)
        good_configs = [r for r in sorted_results if r['energy_gap_percent'] and 5 <= r['energy_gap_percent'] <= 20]
        
        if good_configs:
            print("\n✓ GOOD CONFIGURATIONS (energy gap 5-20%):")
            for r in good_configs[:3]:  # Top 3
                dp = r.get('distance_penalty', DISTANCE_PENALTY)
                print(f"  {r['config']}: epsilon={r['epsilon']}, steps={r['adv_steps']}, distance_penalty={dp}")
                print(f"    Energy gap: {r['energy_gap_percent']:+.1f}%")
                if r['mse_harder']:
                    print(f"    Test MSE: {r['mse_harder']:.4f}")
            
            print("\n→ Run these configs at 50k steps to validate:")
            for r in good_configs[:2]:  # Top 2
                dp = r.get('distance_penalty', DISTANCE_PENALTY)
                print(f"   python train.py --use-anm --anm-epsilon {r['epsilon']} --anm-adversarial-steps {r['adv_steps']} --anm-distance-penalty {dp} --train-steps 50000")
        else:
            print("\n❌ NO GOOD CONFIGURATIONS FOUND")
            print("   All configs have either too weak (<5%) or too strong (>20%) energy gaps.")
            print("   Consider:")
            print("   - Testing wider epsilon range: [0.5, 1.0, 2.0]")
            print("   - Testing more adversarial steps: [50, 100, 200]")
            print("   - Already testing distance penalties: 0.01 and 0.001")
        
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
                'distance_penalties_tested': [0.01, 0.001],
                'includes_baseline': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'sweep_results': sweep_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Sweep results saved to: {results_file}")
    
    def evaluate_model(self, dataset, model_type='baseline', ood=False):
        """Evaluate a trained model on same or harder difficulty"""
        result_dir = self.get_result_dir(dataset, model_type)
        
        # Check if model exists
        if not os.path.exists(f'{result_dir}/model-20.pt'):
            print(f"\n{'='*80}")
            print(f"ERROR: No trained model found for {dataset} ({model_type})")
            print(f"Expected location: {result_dir}/model-20.pt")
            print(f"Please train the model first.")
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
        
        # Add model-specific parameters for evaluation
        if model_type == 'anm':
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', '5',
                '--anm-distance-penalty', str(DISTANCE_PENALTY),
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
                recommendations += "1. Increase --anm-adversarial-steps\n   from 5 to 20-50\n\n"
                recommendations += "2. Increase epsilon to 0.5-1.0\n\n"
                recommendations += "3. Reduce distance penalty\n   from 0.1 to 0.01\n\n"
                recommendations += "4. Start ANM earlier\n   (5% instead of 10%)"
            elif ratio < 0.95:
                recommendations += "1. Check gradient sign\n   (should maximize energy)\n\n"
                recommendations += "2. Verify energy computation\n\n"
                recommendations += "3. Ensure model.eval() before ANM\n\n"
                recommendations += "4. Check for gradient clipping"
            else:
                recommendations += "✓ Current settings working\n\n"
                recommendations += "Consider:\n"
                recommendations += "• Longer training\n"
                recommendations += "• Fine-tune epsilon\n"
                recommendations += "• Adjust adversarial steps"
        
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
                print("  1. Increase anm_adversarial_steps from 5 to 20-50")
                print("  2. Increase epsilon from 0.1 to 0.5-1.0")
                print("  3. Reduce anm_distance_penalty from 0.1 to 0.01")
                print("  4. Start ANM earlier in curriculum (5% instead of 10%)")
                print("  5. Use more aggressive curriculum with higher ANM percentages")
                
                print("\n  CODE CHANGES NEEDED:")
                print("  - In train.py: --anm-adversarial-steps 20")
                print("  - In adversarial_corruption.py: increase eps_iter")
                print("  - In denoising_diffusion_pytorch_1d.py: reduce distance_penalty weight")
                
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
                print("  3. Fine-tune epsilon and adversarial steps")
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
                print("  ⚠️  ANM samples have LOWER energy than random noise!")
                print("     This indicates a serious problem with the adversarial process.")
            elif mean_anm < mean_ired * 1.05:
                print("  ⚠️  ANM barely improves over standard IRED.")
                print("     The adversarial process needs stronger hyperparameters.")
            else:
                print("  ✓  ANM successfully creates harder negatives than IRED.")
        
        print("\n" + "="*80)
        print("END OF DIAGNOSTIC SUMMARY")
        print("="*80 + "\n")

# Initialize runner with base directory
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='experiments', help='Base directory for experiments')
    parser.add_argument('--force', action='store_true', help='Force retrain even if models exist')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep instead of full training')
    parser.add_argument('--dataset', default='addition', help='Dataset for sweep')
    args = parser.parse_args()

    runner = ExperimentRunner(base_dir=args.base_dir)

    if args.sweep:
        # Run hyperparameter sweep
        runner.run_hyperparameter_sweep(dataset=args.dataset, force_retrain=args.force)
    else:
        # Train all models (diagnostics run automatically after each training)
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
