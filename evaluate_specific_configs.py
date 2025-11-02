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


# Hyperparameters - Updated to 50k iterations
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 50000  # Updated from 20k to 50k
DIFFUSION_STEPS = 10
RANK = 20  # For 20x20 matrices

# Tasks to run
TASKS = ['addition']

# Specific ANM configurations to test
ANM_CONFIGS = [
    {
        'name': 'anm_eps0.3_steps20_dp0.01',
        'epsilon': 0.3,
        'adv_steps': 20,
        'distance_penalty': 0.01
    },
    {
        'name': 'anm_eps0.3_steps5_dp0.001',
        'epsilon': 0.3,
        'adv_steps': 5,
        'distance_penalty': 0.001
    }
]


class ExperimentRunner:
    def __init__(self, base_dir='experiments'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.results = {}
        self.diagnostic_results = defaultdict(dict)
    
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
                base += '_anm_curriculum'
        return base

    def load_model_for_diagnostics(self, model_dir, device='cuda'):
        """Load model checkpoint for diagnostic purposes with robust prefix handling"""
        checkpoint_path = Path(model_dir) / 'model-50.pt'
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
        
        # Initialize model
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
            state_dict = checkpoint['ema']['ema_model']
        else:
            state_dict = checkpoint
        
        # Robust prefix handling
        if state_dict:
            model_state = model.state_dict()
            fixed_state_dict = {}
            
            sample_checkpoint_key = next(iter(state_dict.keys()))
            sample_model_key = next(iter(model_state.keys()))
            
            prefixes_to_remove = ['model.', 'module.']
            prefix_removed = False
            
            for prefix in prefixes_to_remove:
                if sample_checkpoint_key.startswith(prefix) and not sample_model_key.startswith(prefix):
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            new_key = k[len(prefix):]
                            if new_key in model_state:
                                fixed_state_dict[new_key] = v
                    prefix_removed = True
                    print(f"Info: Removed '{prefix}' prefix from checkpoint keys")
                    break
            
            if not prefix_removed:
                for k, v in state_dict.items():
                    if any(k.startswith(skip) for skip in ['betas', 'alphas', 'sqrt', 'log', 'posterior', 'loss', 'opt_step']):
                        continue
                    if k in model_state:
                        fixed_state_dict[k] = v
            
            state_dict = fixed_state_dict
        
        # Load the cleaned state dict
        if state_dict:
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Checkpoint loading issues detected, loading with strict=False")
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
            batch = self._sample_batch(data, 256)
            x_clean = batch['x'].to(device)
            y_clean = batch['y'].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (len(x_clean),), device=device)
            
            # 1. Clean samples energy
            with torch.no_grad():
                energy_clean = diffusion.energy_score(x_clean, y_clean, t)
                energies['clean'].append(energy_clean.mean().item())
            
            # 2. Standard IRED corruption
            noise = torch.randn_like(y_clean)
            alpha = 1.0 - (t.float() / diffusion.num_timesteps).view(-1, 1)
            y_ired = alpha * y_clean + (1 - alpha) * noise
            with torch.no_grad():
                energy_ired = diffusion.energy_score(x_clean, y_ired, t)
                energies['ired_standard'].append(energy_ired.mean().item())
            
            # 3. ANM adversarial corruption
            y_anm = self._simulate_anm_output(x_clean, y_clean.clone(), t, diffusion, num_steps=5)
            with torch.no_grad():
                energy_anm = diffusion.energy_score(x_clean, y_anm, t)
                energies['anm_adversarial'].append(energy_anm.mean().item())
            
            # 4. Gaussian noise corruption
            y_gaussian = y_clean + 0.1 * torch.randn_like(y_clean)
            with torch.no_grad():
                energy_gaussian = diffusion.energy_score(x_clean, y_gaussian, t)
                energies['gaussian_noise'].append(energy_gaussian.mean().item())
        
        return energies

    def _simulate_anm_output(self, x_clean, y_clean, t, diffusion, num_steps=5, eps=0.1):
        """Simulate ANM adversarial corruption on output y given input x"""
        y_adv = y_clean.clone().requires_grad_(True)
        
        for _ in range(num_steps):
            energy = diffusion.energy_score(x_clean, y_adv, t)
            grad = torch.autograd.grad(energy.sum(), y_adv)[0]

            with torch.no_grad():
                y_adv = y_adv + eps * grad.sign()
                movement = y_adv - y_clean
                movement_norm = torch.norm(movement, dim=-1, keepdim=True)
                max_movement = 0.001  # Default distance penalty
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
        
        batch = self._sample_batch(data, 100)
        x_clean = batch['x'].to(device)
        y_clean = batch['y'].to(device)
        
        t = torch.randint(0, baseline_diffusion.num_timesteps, (len(x_clean),), device=device)
        
        # Generate negatives using both methods
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
                   epsilon=None, adv_steps=None, distance_penalty=None):
        """Train a model for a specific dataset and model type"""
        result_dir = self.get_result_dir(dataset, model_type, epsilon, adv_steps, distance_penalty)
        
        # Check if model already exists
        if not force_retrain and os.path.exists(f'{result_dir}/model-50.pt'):
            print(f"\n{'='*80}")
            print(f"Model for {dataset} ({model_type}) already exists. Skipping training.")
            print(f"Use --force to retrain.")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            
            # Run diagnostics on existing model
            print(f"Running diagnostics on existing {model_type} model...")
            energies = self.run_energy_diagnostics(result_dir, dataset)
            if energies:
                config_key = f'{dataset}_{model_type}'
                if epsilon is not None and adv_steps is not None:
                    config_key += f'_eps{epsilon}_steps{adv_steps}'
                self.diagnostic_results[config_key]['energies'] = energies
                self._print_energy_summary(energies, model_type)
            
            return True
            
        print(f"\n{'='*80}")
        print(f"Training IRED ({model_type.upper()}) on {dataset.upper()} task")
        print(f"{'='*80}")
        print(f"Model Type: {model_type}")
        if epsilon is not None and adv_steps is not None:
            print(f"ANM Hyperparameters: epsilon={epsilon}, adversarial_steps={adv_steps}, distance_penalty={distance_penalty}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
        print(f"Diffusion steps: {DIFFUSION_STEPS}")
        print(f"Matrix rank: {RANK}")
        print(f"Result directory: {result_dir}")
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
            '--train-steps', str(TRAIN_ITERATIONS),
        ]
        
        # Add model-specific parameters
        if model_type == 'anm':
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', str(adv_steps),
                '--anm-epsilon', str(epsilon),
                '--anm-distance-penalty', str(distance_penalty),
                '--anm-temperature', '1.0',
                '--anm-clean-ratio', '0.1',
                '--anm-adversarial-ratio', '0.8',
                '--anm-gaussian-ratio', '0.1',
            ])
        
        # Run training with real-time output
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    sys.stdout.flush()
            
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
                    config_key = f'{dataset}_{model_type}'
                    if epsilon is not None and adv_steps is not None:
                        config_key += f'_eps{epsilon}_steps{adv_steps}'
                    self.diagnostic_results[config_key]['energies'] = energies
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
    
    def evaluate_model(self, dataset, model_type='baseline', epsilon=None, adv_steps=None, 
                      distance_penalty=None, ood=False):
        """Evaluate a trained model on same or harder difficulty"""
        result_dir = self.get_result_dir(dataset, model_type, epsilon, adv_steps, distance_penalty)
        
        # Check if model exists
        if not os.path.exists(f'{result_dir}/model-50.pt'):
            print(f"\n{'='*80}")
            print(f"ERROR: No trained model found for {dataset} ({model_type})")
            print(f"Expected location: {result_dir}/model-50.pt")
            print(f"Please train the model first.")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return None
        
        difficulty = "Harder Difficulty (OOD)" if ood else "Same Difficulty"
        config_desc = model_type
        if epsilon is not None and adv_steps is not None:
            config_desc += f" (eps={epsilon}, steps={adv_steps}, dp={distance_penalty})"
        
        print(f"\n{'='*80}")
        print(f"Evaluating IRED ({config_desc.upper()}) on {dataset.upper()} - {difficulty}")
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
            '--train-steps', str(TRAIN_ITERATIONS),
            '--load-milestone', '1',
            '--evaluate',
        ]
        
        # Add model-specific parameters for evaluation
        if model_type == 'anm':
            cmd.extend([
                '--use-anm',
                '--anm-adversarial-steps', str(adv_steps),
                '--anm-epsilon', str(epsilon),
                '--anm-distance-penalty', str(distance_penalty),
            ])
        
        if ood:
            cmd.append('--ood')
        
        # Run evaluation with real-time output
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
                    print(line.rstrip())
                    sys.stdout.flush()
                    output_lines.append(line)
            
            result = process.wait()
            
            if result == 0:
                # Parse output to extract MSE
                output_text = ''.join(output_lines)
                mse = self._parse_mse_from_output(output_text, '')
                
                print(f"\n{'='*80}")
                print(f"Evaluation completed for {dataset} ({config_desc}) - {difficulty}")
                if mse is not None:
                    print(f"MSE: {mse:.4f}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                return mse
            else:
                print(f"\n{'='*80}")
                print(f"ERROR: Evaluation failed for {dataset} ({config_desc}) - {difficulty} with exit code {result}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return None
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Evaluation failed for {dataset} ({config_desc}) - {difficulty}: {e}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return None
    
    def _parse_mse_from_output(self, stdout, stderr):
        """Parse MSE from training/evaluation output"""
        output = stdout + stderr
        lines = output.split('\n')
        
        mse_value = None
        for i, line in enumerate(lines):
            if line.startswith('mse') and '  ' in line:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == 'mse':
                    try:
                        mse_value = float(parts[1])
                    except (ValueError, IndexError):
                        pass
        
        if mse_value is None:
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
    
    def train_specific_configs(self, force_retrain=False):
        """Train baseline and specific ANM configurations"""
        print(f"\n{'#'*80}")
        print(f"# TRAINING SPECIFIC CONFIGURATIONS AT 50K ITERATIONS")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"# Configurations:")
        print(f"#   1. Baseline (IRED)")
        for config in ANM_CONFIGS:
            print(f"#   2. ANM: epsilon={config['epsilon']}, adv_steps={config['adv_steps']}, distance_penalty={config['distance_penalty']}")
        print(f"# Training Steps: {TRAIN_ITERATIONS}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        success = {}
        
        for dataset in TASKS:
            # Train baseline
            key = f"{dataset}_baseline"
            success[key] = self.train_model(dataset, 'baseline', force_retrain)
            
            # Train each ANM config
            for config in ANM_CONFIGS:
                key = f"{dataset}_{config['name']}"
                success[key] = self.train_model(
                    dataset, 
                    'anm', 
                    force_retrain,
                    epsilon=config['epsilon'],
                    adv_steps=config['adv_steps'],
                    distance_penalty=config['distance_penalty']
                )
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING SUMMARY")
        print(f"{'#'*80}")
        for dataset in TASKS:
            print(f"\n{dataset.upper()}:")
            print(f"  {'Baseline':<40s}: {'✓ SUCCESS' if success.get(f'{dataset}_baseline', False) else '✗ FAILED'}")
            for config in ANM_CONFIGS:
                key = f"{dataset}_{config['name']}"
                config_desc = f"ANM (eps={config['epsilon']}, steps={config['adv_steps']}, dp={config['distance_penalty']})"
                status = '✓ SUCCESS' if success.get(key, False) else '✗ FAILED'
                print(f"  {config_desc:<40s}: {status}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        return all(success.values())
    
    def evaluate_specific_configs(self):
        """Evaluate baseline and specific ANM configurations"""
        print(f"\n{'#'*80}")
        print(f"# EVALUATING SPECIFIC CONFIGURATIONS")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        results = {}
        
        for dataset in TASKS:
            results[dataset] = {}
            
            # Evaluate baseline
            results[dataset]['baseline'] = {
                'same_difficulty': self.evaluate_model(dataset, 'baseline', ood=False),
                'harder_difficulty': self.evaluate_model(dataset, 'baseline', ood=True)
            }
            
            # Evaluate each ANM config
            for config in ANM_CONFIGS:
                results[dataset][config['name']] = {
                    'same_difficulty': self.evaluate_model(
                        dataset, 'anm', 
                        epsilon=config['epsilon'],
                        adv_steps=config['adv_steps'],
                        distance_penalty=config['distance_penalty'],
                        ood=False
                    ),
                    'harder_difficulty': self.evaluate_model(
                        dataset, 'anm',
                        epsilon=config['epsilon'],
                        adv_steps=config['adv_steps'],
                        distance_penalty=config['distance_penalty'],
                        ood=True
                    )
                }
        
        self.results = results
        
        # Run comparative diagnostics after evaluation
        print(f"\n{'#'*80}")
        print(f"# RUNNING COMPARATIVE DIAGNOSTICS")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            baseline_dir = self.get_result_dir(dataset, 'baseline')
            
            for config in ANM_CONFIGS:
                anm_dir = self.get_result_dir(
                    dataset, 'anm', 
                    config['epsilon'], 
                    config['adv_steps'],
                    config['distance_penalty']
                )
                
                comp_result = self.run_comparative_diagnostics(
                    baseline_dir, anm_dir, dataset, config['name']
                )
                if comp_result:
                    self.diagnostic_results[f"{dataset}_{config['name']}_comparative"] = comp_result
        
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
        print(f"{'Task':<20s} {'Configuration':<50s} {'Same Difficulty':>15s} {'Harder Difficulty':>17s}")
        print(f"{'-'*20} {'-'*50} {'-'*15} {'-'*17}")
        
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
            print(f"{task_name:<20s} {'IRED (baseline)':<50s} {baseline_same_str:>15s} {baseline_harder_str:>17s}")
            
            # Each ANM config
            for config in ANM_CONFIGS:
                config_desc = f"ANM (eps={config['epsilon']}, steps={config['adv_steps']}, dp={config['distance_penalty']})"
                anm_same = self.results.get(dataset, {}).get(config['name'], {}).get('same_difficulty')
                anm_harder = self.results.get(dataset, {}).get(config['name'], {}).get('harder_difficulty')
                anm_same_str = f"{anm_same:.4f}" if anm_same is not None else "N/A"
                anm_harder_str = f"{anm_harder:.4f}" if anm_harder is not None else "N/A"
                print(f"{'':<20s} {config_desc:<50s} {anm_same_str:>15s} {anm_harder_str:>17s}")
            
            print()
        
        # Print improvement percentages
        print(f"\n{'#'*80}")
        print(f"# RELATIVE IMPROVEMENTS vs BASELINE")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            task_name = task_display.get(dataset, dataset)
            baseline_same = self.results.get(dataset, {}).get('baseline', {}).get('same_difficulty')
            baseline_harder = self.results.get(dataset, {}).get('baseline', {}).get('harder_difficulty')
            
            if baseline_same and baseline_harder:
                print(f"{task_name}:")
                
                for config in ANM_CONFIGS:
                    anm_same = self.results.get(dataset, {}).get(config['name'], {}).get('same_difficulty')
                    anm_harder = self.results.get(dataset, {}).get(config['name'], {}).get('harder_difficulty')
                    
                    if anm_same and anm_harder:
                        same_imp = ((baseline_same - anm_same) / baseline_same) * 100
                        harder_imp = ((baseline_harder - anm_harder) / baseline_harder) * 100
                        config_desc = f"eps={config['epsilon']}, steps={config['adv_steps']}, dp={config['distance_penalty']}"
                        print(f"  {config_desc}: {same_imp:+.1f}% (same), {harder_imp:+.1f}% (harder)")
                
                # Print comparative diagnostics
                for config in ANM_CONFIGS:
                    comp_key = f"{dataset}_{config['name']}_comparative"
                    if comp_key in self.diagnostic_results:
                        comp = self.diagnostic_results[comp_key]
                        print(f"  {config['name']} energy gap: {comp['energy_gap_percent']:+.1f}%")
        
        print(f"\n{'#'*80}")
        print(f"# Paper's reported IRED results for comparison:")
        print(f"{'#'*80}")
        print(f"{'Addition':<20s} {'IRED (paper)':<50s} {'0.0002':>15s} {'0.0020':>17s}")
        print(f"{'Matrix Completion':<20s} {'IRED (paper)':<50s} {'0.0174':>15s} {'0.2054':>17s}")
        print(f"{'Matrix Inverse':<20s} {'IRED (paper)':<50s} {'0.0095':>15s} {'0.2063':>17s}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
    
    def _save_results(self):
        """Save results to JSON file"""
        results_file = self.base_dir / 'specific_configs_results_50k.json'
        
        # Add metadata
        results_with_meta = {
            'metadata': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'train_iterations': TRAIN_ITERATIONS,
                'diffusion_steps': DIFFUSION_STEPS,
                'rank': RANK,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'configurations': [
                    {'type': 'baseline', 'description': 'IRED baseline'},
                    *[{
                        'type': 'anm',
                        'name': config['name'],
                        'epsilon': config['epsilon'],
                        'adv_steps': config['adv_steps'],
                        'distance_penalty': config['distance_penalty']
                    } for config in ANM_CONFIGS]
                ]
            },
            'results': self.results,
            'diagnostics': dict(self.diagnostic_results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        
        print(f"Results saved to: {results_file}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='experiments', help='Base directory for experiments')
    parser.add_argument('--force', action='store_true', help='Force retrain even if models exist')
    args = parser.parse_args()

    runner = ExperimentRunner(base_dir=args.base_dir)

    # Train all specific configurations
    success = runner.train_specific_configs(force_retrain=args.force)

    # Evaluate if training succeeded
    if success:
        results = runner.evaluate_specific_configs()
    else:
        print("\nSome training jobs failed. Skipping evaluation.")
        sys.stdout.flush()