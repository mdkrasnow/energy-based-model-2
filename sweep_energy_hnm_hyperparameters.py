# File: sweep_energy_hnm_hyperparameters.py
# Comprehensive hyperparameter sweep for Energy-Based Hard Negative Mining (HNM)
# This script tests all combinations of num_candidates, refinement_steps, and lambda_weight
# on Matrix Inverse and Matrix Completion tasks with reduced training steps (25k)

import os
import subprocess
import sys

# Clone repository and set up environment
subprocess.run(['rm', '-rf', 'energy-based-model-2'], check=False)
subprocess.run(['git', 'clone', 'https://github.com/mdkrasnow/energy-based-model-2.git'], check=True)
os.chdir('energy-based-model-2')
sys.path.insert(0, os.getcwd())

import argparse
import json
import time
import re
import errno
from pathlib import Path
from collections import defaultdict
from itertools import product

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import model components
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Addition, Inverse, LowRankDataset
from diffusion_lib.energy_hard_negatives import energy_based_hard_negative_mining

# Training hyperparameters
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 25000  # Reduced for rapid iteration
DIFFUSION_STEPS = 10
RANK = 20

# Hyperparameter sweep space for Energy-Based HNM
NUM_CANDIDATES_OPTIONS = [3, 5, 7, 10]
REFINEMENT_STEPS_OPTIONS = [1, 3, 5]
LAMBDA_WEIGHT_OPTIONS = [0.25, 0.5, 1.0, 2.0]

# Tasks to test (Matrix Inverse and Matrix Completion)
TASKS = ['inverse', 'lowrank']


class EnergyHNMSweepRunner:
    """Experimental runner for Energy-Based Hard Negative Mining hyperparameter sweep"""
    
    def __init__(self, base_dir=None):
        """
        base_dir:
          - If provided, use it as-is.
          - If None, default to:
              $ENERGY_HNM_EXPERIMENT_DIR or 'experiments_energy_hnm'
        """
        if base_dir is None:
            base_dir = os.environ.get("ENERGY_HNM_EXPERIMENT_DIR",
                                      "experiments_energy_hnm")

        self.base_dir = Path(base_dir)

        if not self._robust_mkdir(self.base_dir):
            raise OSError(f"Failed to create base directory: {self.base_dir}")

        self.sweep_results = []
        self.diagnostic_data = defaultdict(dict)
    
    def _robust_file_exists(self, file_path, max_retries=3):
        """Robustly check if file exists, handling stale file handles on network filesystems"""
        if file_path is None:
            return False
            
        path = Path(file_path)
        
        for attempt in range(max_retries):
            try:
                return path.exists()
            except OSError as e:
                if e.errno == errno.ESTALE:  # Stale file handle
                    if attempt < max_retries - 1:
                        # Refresh the path object and retry with exponential backoff
                        try:
                            path = Path(str(file_path)).resolve()
                        except OSError:
                            # If resolve() also fails, just recreate the path
                            path = Path(str(file_path))
                        
                        wait_time = 0.1 * (2 ** attempt)
                        print(f"Warning: Stale file handle for {file_path}, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: Failed to check file existence after {max_retries} retries: {file_path}")
                        return False
                else:
                    # Re-raise other OSErrors (permissions, disk errors, etc.)
                    raise
        
        return False  # Should not reach here, but safe fallback
    
    def _robust_mkdir(self, dir_path, max_retries=3):
        """Robustly create directory, handling stale file handles on network filesystems"""
        if dir_path is None:
            return False
            
        path = Path(dir_path)
        
        for attempt in range(max_retries):
            try:
                path.mkdir(parents=True, exist_ok=True)
                return True
            except OSError as e:
                if e.errno == errno.ESTALE:  # Stale file handle
                    if attempt < max_retries - 1:
                        # Refresh the path object and retry with exponential backoff
                        try:
                            path = Path(str(dir_path)).resolve()
                        except OSError:
                            # If resolve() also fails, just recreate the path
                            path = Path(str(dir_path))
                        
                        wait_time = 0.1 * (2 ** attempt)
                        print(f"Warning: Stale file handle for directory {dir_path}, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: Failed to create directory after {max_retries} retries: {dir_path}")
                        return False
                else:
                    # Re-raise other OSErrors (permissions, disk errors, etc.)
                    raise
        
        return False  # Should not reach here, but safe fallback
    
    def _robust_write_file(self, file_path, content, mode='w', max_retries=3, **kwargs):
        """Robustly write file, handling stale file handles on network filesystems"""
        if file_path is None:
            return False
            
        path = Path(file_path)
        
        # Ensure parent directory exists with robust mkdir
        if not self._robust_mkdir(path.parent):
            print(f"Error: Failed to create parent directory for {file_path}")
            return False
        
        for attempt in range(max_retries):
            try:
                with open(path, mode, **kwargs) as f:
                    if isinstance(content, str):
                        f.write(content)
                    elif hasattr(content, 'items'):  # Dictionary for JSON
                        import json
                        json.dump(content, f, indent=2)
                    else:
                        f.write(content)
                return True
            except OSError as e:
                if e.errno == errno.ESTALE:  # Stale file handle
                    if attempt < max_retries - 1:
                        # Refresh the path object and retry with exponential backoff
                        try:
                            path = Path(str(file_path)).resolve()
                        except OSError:
                            # If resolve() also fails, just recreate the path
                            path = Path(str(file_path))
                        
                        wait_time = 0.1 * (2 ** attempt)
                        print(f"Warning: Stale file handle for file {file_path}, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: Failed to write file after {max_retries} retries: {file_path}")
                        return False
                else:
                    # Re-raise other OSErrors (permissions, disk errors, etc.)
                    raise
        
        return False  # Should not reach here, but safe fallback
    
    def _robust_savefig(self, file_path, max_retries=3, **kwargs):
        """Robustly save matplotlib figure, handling stale file handles on network filesystems"""
        if file_path is None:
            return False
            
        path = Path(file_path)
        
        # Ensure parent directory exists with robust mkdir
        if not self._robust_mkdir(path.parent):
            print(f"Error: Failed to create parent directory for {file_path}")
            return False
        
        for attempt in range(max_retries):
            try:
                import matplotlib.pyplot as plt
                plt.savefig(path, **kwargs)
                return True
            except OSError as e:
                if e.errno == errno.ESTALE:  # Stale file handle
                    if attempt < max_retries - 1:
                        # Refresh the path object and retry with exponential backoff
                        try:
                            path = Path(str(file_path)).resolve()
                        except OSError:
                            # If resolve() also fails, just recreate the path
                            path = Path(str(file_path))
                        
                        wait_time = 0.1 * (2 ** attempt)
                        print(f"Warning: Stale file handle for plot {file_path}, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: Failed to save plot after {max_retries} retries: {file_path}")
                        return False
                else:
                    # Re-raise other OSErrors (permissions, disk errors, etc.)
                    raise
        
        return False  # Should not reach here, but safe fallback
        
    def _sample_batch(self, dataset, batch_size):
        """Sample a batch from PyTorch dataset"""
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
    
    def get_result_dir(self, dataset, model_type='baseline', num_candidates=None, 
                      refinement_steps=None, lambda_weight=None):
        """Get results directory for a specific configuration"""
        base = f'results/ds_{dataset}/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        if model_type == 'energy_hnm' and all(x is not None for x in [num_candidates, refinement_steps, lambda_weight]):
            base += f'_hnm_c{num_candidates}_r{refinement_steps}_l{lambda_weight}'
        elif model_type == 'energy_hnm':
            base += '_hnm_default'
        return base
    
    def train_baseline(self, dataset, force_retrain=False):
        """Train baseline IRED model without HNM"""
        result_dir = self.get_result_dir(dataset, 'baseline')
        
        if not force_retrain and self._robust_file_exists(f'{result_dir}/model-20.pt'):
            print(f"\nBaseline for {dataset} already exists. Skipping.")
            return True
        
        print(f"\n{'='*80}")
        print(f"Training BASELINE IRED on {dataset.upper()}")
        print(f"{'='*80}")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
        print(f"Result directory: {result_dir}")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(TRAIN_ITERATIONS),
        ]
        
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
                print(f"\nBaseline training completed in {elapsed/60:.2f} minutes\n")
                return True
            else:
                print(f"\nERROR: Baseline training failed with exit code {result}\n")
                return False
                
        except Exception as e:
            print(f"\nERROR: Baseline training failed: {e}\n")
            return False
    
    def train_energy_hnm_model(self, dataset, num_candidates, refinement_steps, 
                              lambda_weight, force_retrain=False):
        """Train model with specific Energy-Based HNM configuration"""
        result_dir = self.get_result_dir(dataset, 'energy_hnm', num_candidates, 
                                        refinement_steps, lambda_weight)
        
        config_name = f"c{num_candidates}_r{refinement_steps}_l{lambda_weight}"
        
        if not force_retrain and self._robust_file_exists(f'{result_dir}/model-20.pt'):
            print(f"\nModel for {dataset} ({config_name}) already exists. Skipping.")
            return True
        
        print(f"\n{'='*80}")
        print(f"Training Energy-Based HNM: {config_name} on {dataset.upper()}")
        print(f"{'='*80}")
        print(f"HNM Parameters:")
        print(f"  num_candidates: {num_candidates} (landscape sampling diversity)")
        print(f"  refinement_steps: {refinement_steps} (energy descent iterations)")
        print(f"  lambda_weight: {lambda_weight} (deception score balance)")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
        print(f"Result directory: {result_dir}")
        print(f"\nExpected computation cost: {num_candidates * refinement_steps}x baseline")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(TRAIN_ITERATIONS),
            '--use-anm',  # Enable corruption replacement
            '--anm-use-energy-hnm',  # Use energy-based HNM instead of adversarial
            '--hnm-num-candidates', str(num_candidates),
            '--hnm-refinement-steps', str(refinement_steps),
            '--hnm-lambda-weight', str(lambda_weight),
            '--anm-clean-ratio', '0.1',
            '--anm-adversarial-ratio', '0.0',  # Disable adversarial corruption
            '--anm-hard-negative-ratio', '0.8',  # Use HNM for 80% of samples
            '--anm-gaussian-ratio', '0.1',
            '--anm-warmup-steps', str(int(TRAIN_ITERATIONS * 0.05)),  # 5% warmup
        ]
        
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
                print(f"\nTraining completed in {elapsed/60:.2f} minutes\n")
                return True
            else:
                print(f"\nERROR: Training failed with exit code {result}\n")
                return False
                
        except Exception as e:
            print(f"\nERROR: Training failed: {e}\n")
            return False
    
    def evaluate_model(self, dataset, model_type='baseline', ood=False,
                      num_candidates=None, refinement_steps=None, lambda_weight=None):
        """Evaluate a trained model"""
        result_dir = self.get_result_dir(dataset, model_type, num_candidates, 
                                        refinement_steps, lambda_weight)
        
        if not self._robust_file_exists(f'{result_dir}/model-20.pt'):
            print(f"\nERROR: No trained model found at {result_dir}/model-20.pt\n")
            return None
        
        difficulty = "Harder (OOD)" if ood else "Same"
        config_str = f"c{num_candidates}_r{refinement_steps}_l{lambda_weight}" if model_type == 'energy_hnm' else 'baseline'
        
        print(f"\nEvaluating {dataset} ({config_str}) - {difficulty} difficulty...")
        sys.stdout.flush()
        
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
        
        if model_type == 'energy_hnm':
            cmd.extend([
                '--use-anm',
                '--anm-use-energy-hnm',
                '--hnm-num-candidates', str(num_candidates),
                '--hnm-refinement-steps', str(refinement_steps),
                '--hnm-lambda-weight', str(lambda_weight),
            ])
        
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
                mse = self._parse_mse_from_output(output_text)
                print(f"  MSE: {mse:.6f}" if mse else "  MSE: N/A")
                return mse
            else:
                print(f"  Evaluation failed with exit code {result}")
                return None
                
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            return None
    
    def _parse_mse_from_output(self, output):
        """Parse MSE from evaluation output"""
        lines = output.split('\n')
        mse_value = None
        
        for line in lines:
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
    
    def load_model_for_diagnostics(self, model_dir, dataset_name, device='cuda'):
        """Load trained model for diagnostic analysis"""
        checkpoint_path = Path(model_dir) / 'model-20.pt'
        if not self._robust_file_exists(checkpoint_path):
            return None
        
        device = device if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get dataset
        if dataset_name == 'inverse':
            dataset = Inverse("train", RANK, False)
        elif dataset_name == 'lowrank':
            dataset = LowRankDataset("train", RANK, False)
        else:
            dataset = Addition("train", RANK, False)
        
        # Initialize model
        model = EBM(
            inp_dim=dataset.inp_dim,
            out_dim=dataset.out_dim,
        )
        model = DiffusionWrapper(model)
        
        # Load state dict with robust prefix handling
        state_dict = None
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint and 'ema_model' in checkpoint['ema']:
            state_dict = checkpoint['ema']['ema_model']
        else:
            state_dict = checkpoint
        
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
                    break
            
            if not prefix_removed:
                for k, v in state_dict.items():
                    if any(k.startswith(skip) for skip in ['betas', 'alphas', 'sqrt', 'log', 'posterior', 'loss', 'opt_step']):
                        continue
                    if k in model_state:
                        fixed_state_dict[k] = v
            
            state_dict = fixed_state_dict
        
        if state_dict:
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                model.load_state_dict(state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=DIFFUSION_STEPS,
            sampling_timesteps=DIFFUSION_STEPS,
            continuous=True,
            show_inference_tqdm=False,
            use_adversarial_corruption=False,
            sudoku=False,
            shortest_path=False,
        )
        
        return model, diffusion, device, dataset
    
    def run_hnm_diagnostics(self, model_dir, dataset_name, num_candidates=10, 
                           refinement_steps=5, lambda_weight=1.0, num_batches=5):
        """Run comprehensive diagnostics on Energy-Based HNM model"""
        result = self.load_model_for_diagnostics(model_dir, dataset_name)
        if result is None:
            return None
        
        model, diffusion, device, data = result
        
        metrics = {
            'selected_energies': [],
            'selected_errors': [],
            'deception_scores': [],
            'candidate_energy_means': [],
            'candidate_error_means': [],
            'clean_energies': [],
        }
        
        print(f"  Running HNM diagnostics (c={num_candidates}, r={refinement_steps}, λ={lambda_weight})...")
        
        for batch_idx in range(num_batches):
            batch = self._sample_batch(data, 256)
            x_clean = batch['x'].to(device)
            y_clean = batch['y'].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (len(x_clean),), device=device)
            
            # Get clean sample energies
            with torch.no_grad():
                energy_clean = diffusion.energy_score(x_clean, y_clean, t)
                metrics['clean_energies'].extend(energy_clean.cpu().numpy())
            
            # Generate HNM negatives with detailed tracking
            try:
                y_hnm, hnm_metrics = self._generate_hnm_with_metrics(
                    x_clean, y_clean, t, diffusion, 
                    num_candidates, refinement_steps, lambda_weight
                )
                
                # Record metrics
                metrics['selected_energies'].extend(hnm_metrics['selected_energies'])
                metrics['selected_errors'].extend(hnm_metrics['selected_errors'])
                metrics['deception_scores'].extend(hnm_metrics['deception_scores'])
                metrics['candidate_energy_means'].append(hnm_metrics['candidate_energy_mean'])
                metrics['candidate_error_means'].append(hnm_metrics['candidate_error_mean'])
                
            except Exception as e:
                print(f"    Warning: HNM generation failed for batch {batch_idx}: {e}")
                continue
        
        # Compute aggregate statistics
        diagnostics = {
            'avg_selected_energy': float(np.mean(metrics['selected_energies'])),
            'std_selected_energy': float(np.std(metrics['selected_energies'])),
            'avg_selected_error': float(np.mean(metrics['selected_errors'])),
            'std_selected_error': float(np.std(metrics['selected_errors'])),
            'avg_deception_score': float(np.mean(metrics['deception_scores'])),
            'std_deception_score': float(np.std(metrics['deception_scores'])),
            'avg_candidate_energy': float(np.mean(metrics['candidate_energy_means'])),
            'avg_candidate_error': float(np.mean(metrics['candidate_error_means'])),
            'avg_clean_energy': float(np.mean(metrics['clean_energies'])),
            'std_clean_energy': float(np.std(metrics['clean_energies'])),
            'energy_vs_clean': float(np.mean(metrics['selected_energies'])) / (float(np.mean(metrics['clean_energies'])) + 1e-8),
        }
        
        return diagnostics
    
    def _generate_hnm_with_metrics(self, x_inp, y_clean, t, diffusion, 
                                   num_candidates, refinement_steps, lambda_weight):
        """Generate HNM negatives and return detailed metrics"""
        batch_size = x_inp.shape[0]
        device = x_inp.device
        
        # Storage for all candidates
        all_candidates = []
        all_energies = []
        all_errors = []
        
        # Generate candidates by sampling different landscape indices
        available_timesteps = list(range(diffusion.num_timesteps))
        
        for cand_idx in range(num_candidates):
            # Cycle through timesteps for landscape diversity
            landscape_t = torch.full((batch_size,), available_timesteps[cand_idx % len(available_timesteps)], 
                                    device=device, dtype=torch.long)
            
            # Initialize with q_sample (standard forward process)
            noise = torch.randn_like(y_clean)
            alpha = 1.0 - (landscape_t.float() / diffusion.num_timesteps).view(-1, 1)
            y_candidate = alpha * y_clean + (1 - alpha) * noise
            
            # Run energy descent refinement (pure gradient descent, no distance penalty)
            y_candidate.requires_grad_(True)
            
            for step in range(refinement_steps):
                if y_candidate.grad is not None:
                    y_candidate.grad.zero_()
                
                energy = diffusion.energy_score(x_inp, y_candidate, t)
                energy_sum = energy.sum()
                energy_sum.backward()
                
                with torch.no_grad():
                    # Pure energy descent: move in direction of increasing energy
                    grad = y_candidate.grad
                    y_candidate = y_candidate - 0.01 * grad  # Small step size
                    y_candidate = y_candidate.detach()
                    y_candidate.requires_grad_(True)
            
            y_candidate = y_candidate.detach()
            
            # Evaluate candidate
            with torch.no_grad():
                cand_energy = diffusion.energy_score(x_inp, y_candidate, t)
                cand_error = F.mse_loss(y_candidate, y_clean, reduction='none').mean(dim=-1)
                
                all_candidates.append(y_candidate)
                all_energies.append(cand_energy)
                all_errors.append(cand_error)
        
        # Stack all candidates
        all_candidates = torch.stack(all_candidates, dim=0)  # [num_candidates, batch_size, dim]
        all_energies = torch.stack(all_energies, dim=0)  # [num_candidates, batch_size]
        all_errors = torch.stack(all_errors, dim=0)  # [num_candidates, batch_size]
        
        # Normalize energies and errors to [0, 1] per batch item
        energies_transposed = all_energies.T  # [batch_size, num_candidates]
        errors_transposed = all_errors.T  # [batch_size, num_candidates]
        
        energy_min = energies_transposed.min(dim=1, keepdim=True)[0]
        energy_max = energies_transposed.max(dim=1, keepdim=True)[0]
        energy_norm = (energies_transposed - energy_min) / (energy_max - energy_min + 1e-8)
        
        error_min = errors_transposed.min(dim=1, keepdim=True)[0]
        error_max = errors_transposed.max(dim=1, keepdim=True)[0]
        error_norm = (errors_transposed - error_min) / (error_max - error_min + 1e-8)
        
        # Compute deception scores: score = -energy + λ * error
        # We want LOW energy (easy to fool) + HIGH error (very wrong)
        deception_scores = -energy_norm + lambda_weight * error_norm  # [batch_size, num_candidates]
        
        # Select highest-scoring candidate per batch item
        best_indices = deception_scores.argmax(dim=1)  # [batch_size]
        
        # Extract selected candidates
        batch_indices = torch.arange(batch_size, device=device)
        y_selected = all_candidates[best_indices, batch_indices]  # [batch_size, dim]
        selected_energies = all_energies[best_indices, batch_indices]  # [batch_size]
        selected_errors = all_errors[best_indices, batch_indices]  # [batch_size]
        selected_deception = deception_scores[batch_indices, best_indices]  # [batch_size]
        
        # Gather metrics
        metrics = {
            'selected_energies': selected_energies.cpu().numpy().tolist(),
            'selected_errors': selected_errors.cpu().numpy().tolist(),
            'deception_scores': selected_deception.cpu().numpy().tolist(),
            'candidate_energy_mean': all_energies.mean().item(),
            'candidate_error_mean': all_errors.mean().item(),
        }
        
        return y_selected, metrics
    
    def run_complete_sweep(self, force_retrain=False):
        """Run complete hyperparameter sweep across all configurations"""
        print(f"\n{'#'*80}")
        print(f"# ENERGY-BASED HNM HYPERPARAMETER SWEEP")
        print(f"{'#'*80}")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"# Training steps: {TRAIN_ITERATIONS} (reduced for rapid iteration)")
        print(f"# Hyperparameter space:")
        print(f"#   num_candidates: {NUM_CANDIDATES_OPTIONS}")
        print(f"#   refinement_steps: {REFINEMENT_STEPS_OPTIONS}")
        print(f"#   lambda_weight: {LAMBDA_WEIGHT_OPTIONS}")
        print(f"# Total configurations per task: {len(NUM_CANDIDATES_OPTIONS) * len(REFINEMENT_STEPS_OPTIONS) * len(LAMBDA_WEIGHT_OPTIONS)}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        # Train baselines first
        for dataset in TASKS:
            print(f"\n{'='*80}")
            print(f"Training baseline for {dataset.upper()}")
            print(f"{'='*80}")
            success = self.train_baseline(dataset, force_retrain)
            if not success:
                print(f"Baseline training failed for {dataset}. Continuing anyway...")
        
        # Evaluate baselines
        baseline_results = {}
        for dataset in TASKS:
            print(f"\n{'='*80}")
            print(f"Evaluating baseline for {dataset.upper()}")
            print(f"{'='*80}")
            
            baseline_results[dataset] = {
                'mse_same': self.evaluate_model(dataset, 'baseline', ood=False),
                'mse_harder': self.evaluate_model(dataset, 'baseline', ood=True),
            }
            
            # Run diagnostics on baseline
            baseline_dir = self.get_result_dir(dataset, 'baseline')
            baseline_diag = self.run_baseline_diagnostics(baseline_dir, dataset)
            if baseline_diag:
                baseline_results[dataset]['diagnostics'] = baseline_diag
        
        # Store baseline results
        for dataset in TASKS:
            self.sweep_results.append({
                'dataset': dataset,
                'config_type': 'baseline',
                'num_candidates': None,
                'refinement_steps': None,
                'lambda_weight': None,
                'mse_same': baseline_results[dataset]['mse_same'],
                'mse_harder': baseline_results[dataset]['mse_harder'],
                'diagnostics': baseline_results[dataset].get('diagnostics'),
            })
        
        # Run sweep for each dataset
        total_configs = len(NUM_CANDIDATES_OPTIONS) * len(REFINEMENT_STEPS_OPTIONS) * len(LAMBDA_WEIGHT_OPTIONS)
        
        for dataset in TASKS:
            print(f"\n{'#'*80}")
            print(f"# SWEEPING {dataset.upper()}")
            print(f"# {total_configs} configurations")
            print(f"{'#'*80}\n")
            
            config_num = 0
            for num_cand, ref_steps, lambda_w in product(NUM_CANDIDATES_OPTIONS, 
                                                         REFINEMENT_STEPS_OPTIONS, 
                                                         LAMBDA_WEIGHT_OPTIONS):
                config_num += 1
                config_name = f"c{num_cand}_r{ref_steps}_l{lambda_w}"
                
                print(f"\n{'='*80}")
                print(f"CONFIG {config_num}/{total_configs}: {config_name}")
                print(f"  Dataset: {dataset}")
                print(f"  num_candidates: {num_cand}")
                print(f"  refinement_steps: {ref_steps}")
                print(f"  lambda_weight: {lambda_w}")
                print(f"  Estimated compute: {num_cand * ref_steps}x baseline")
                print(f"{'='*80}")
                
                # Train model
                success = self.train_energy_hnm_model(
                    dataset, num_cand, ref_steps, lambda_w, force_retrain
                )
                
                if not success:
                    print(f"Training failed for {config_name}, skipping evaluation...")
                    self.sweep_results.append({
                        'dataset': dataset,
                        'config_type': 'energy_hnm',
                        'config_name': config_name,
                        'num_candidates': num_cand,
                        'refinement_steps': ref_steps,
                        'lambda_weight': lambda_w,
                        'mse_same': None,
                        'mse_harder': None,
                        'diagnostics': None,
                        'training_status': 'failed',
                    })
                    continue
                
                # Evaluate model
                mse_same = self.evaluate_model(dataset, 'energy_hnm', ood=False,
                                              num_candidates=num_cand,
                                              refinement_steps=ref_steps,
                                              lambda_weight=lambda_w)
                
                mse_harder = self.evaluate_model(dataset, 'energy_hnm', ood=True,
                                                num_candidates=num_cand,
                                                refinement_steps=ref_steps,
                                                lambda_weight=lambda_w)
                
                # Run diagnostics
                model_dir = self.get_result_dir(dataset, 'energy_hnm', num_cand, ref_steps, lambda_w)
                diagnostics = self.run_hnm_diagnostics(model_dir, dataset, num_cand, ref_steps, lambda_w)
                
                # Store results
                result = {
                    'dataset': dataset,
                    'config_type': 'energy_hnm',
                    'config_name': config_name,
                    'num_candidates': num_cand,
                    'refinement_steps': ref_steps,
                    'lambda_weight': lambda_w,
                    'mse_same': mse_same,
                    'mse_harder': mse_harder,
                    'diagnostics': diagnostics,
                    'training_status': 'success',
                }
                self.sweep_results.append(result)
                
                # Print quick summary
                print(f"\n✓ Config {config_name} complete:")
                if mse_same:
                    print(f"  MSE (same): {mse_same:.6f}")
                if mse_harder:
                    print(f"  MSE (harder): {mse_harder:.6f}")
                if diagnostics:
                    print(f"  Avg deception score: {diagnostics['avg_deception_score']:.4f}")
                    print(f"  Selected energy vs clean: {diagnostics['energy_vs_clean']:.2f}x")
        
        # Generate comprehensive analysis
        self.print_sweep_summary()
        self.save_results_json()
        self.generate_heatmaps()
        self.generate_markdown_report()
        
        return self.sweep_results
    
    def run_baseline_diagnostics(self, model_dir, dataset_name, num_batches=5):
        """Run diagnostics on baseline IRED model"""
        result = self.load_model_for_diagnostics(model_dir, dataset_name)
        if result is None:
            return None
        
        model, diffusion, device, data = result
        
        clean_energies = []
        ired_energies = []
        
        print(f"  Running baseline diagnostics...")
        
        for _ in range(num_batches):
            batch = self._sample_batch(data, 256)
            x_clean = batch['x'].to(device)
            y_clean = batch['y'].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (len(x_clean),), device=device)
            
            with torch.no_grad():
                # Clean samples
                energy_clean = diffusion.energy_score(x_clean, y_clean, t)
                clean_energies.extend(energy_clean.cpu().numpy())
                
                # IRED standard corruption
                noise = torch.randn_like(y_clean)
                alpha = 1.0 - (t.float() / diffusion.num_timesteps).view(-1, 1)
                y_ired = alpha * y_clean + (1 - alpha) * noise
                energy_ired = diffusion.energy_score(x_clean, y_ired, t)
                ired_energies.extend(energy_ired.cpu().numpy())
        
        return {
            'avg_clean_energy': float(np.mean(clean_energies)),
            'std_clean_energy': float(np.std(clean_energies)),
            'avg_ired_energy': float(np.mean(ired_energies)),
            'std_ired_energy': float(np.std(ired_energies)),
        }
    
    def print_sweep_summary(self):
        """Print comprehensive summary of sweep results"""
        print(f"\n{'#'*80}")
        print(f"# SWEEP RESULTS SUMMARY")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            print(f"\n{'='*80}")
            print(f"{dataset.upper()} RESULTS")
            print(f"{'='*80}")
            
            # Filter results for this dataset
            dataset_results = [r for r in self.sweep_results if r['dataset'] == dataset]
            baseline = [r for r in dataset_results if r['config_type'] == 'baseline'][0]
            hnm_results = [r for r in dataset_results if r['config_type'] == 'energy_hnm']
            
            # Print baseline
            print(f"\nBASELINE:")
            print(f"  MSE (same): {baseline['mse_same']:.6f}" if baseline['mse_same'] else "  MSE (same): N/A")
            print(f"  MSE (harder): {baseline['mse_harder']:.6f}" if baseline['mse_harder'] else "  MSE (harder): N/A")
            if baseline.get('diagnostics'):
                diag = baseline['diagnostics']
                print(f"  Avg clean energy: {diag['avg_clean_energy']:.4f}")
                print(f"  Avg IRED energy: {diag['avg_ired_energy']:.4f}")
            
            # Sort HNM results by MSE (harder)
            hnm_results_sorted = sorted([r for r in hnm_results if r['mse_harder'] is not None],
                                       key=lambda x: x['mse_harder'])
            
            # Print table header
            print(f"\n{'Config':<15s} {'Cand':>5s} {'Ref':>4s} {'Lambda':>7s} {'MSE(same)':>11s} {'MSE(harder)':>12s} {'Deception':>11s} {'Energy vs Clean':>16s}")
            print("-" * 90)
            
            # Print all HNM results
            for r in hnm_results_sorted:
                config = r['config_name']
                mse_s = f"{r['mse_same']:.6f}" if r['mse_same'] else "N/A"
                mse_h = f"{r['mse_harder']:.6f}" if r['mse_harder'] else "N/A"
                
                if r.get('diagnostics'):
                    deception = f"{r['diagnostics']['avg_deception_score']:.4f}"
                    energy_ratio = f"{r['diagnostics']['energy_vs_clean']:.2f}x"
                else:
                    deception = "N/A"
                    energy_ratio = "N/A"
                
                print(f"{config:<15s} {r['num_candidates']:>5d} {r['refinement_steps']:>4d} {r['lambda_weight']:>7.2f} {mse_s:>11s} {mse_h:>12s} {deception:>11s} {energy_ratio:>16s}")
            
            # Highlight top 3 configurations
            print(f"\n{'='*80}")
            print(f"TOP 3 CONFIGURATIONS BY TEST MSE (HARDER):")
            print(f"{'='*80}")
            
            for i, r in enumerate(hnm_results_sorted[:3], 1):
                improvement = ((baseline['mse_harder'] - r['mse_harder']) / baseline['mse_harder'] * 100) if baseline['mse_harder'] and r['mse_harder'] else 0
                print(f"\n{i}. {r['config_name']}:")
                print(f"   MSE (harder): {r['mse_harder']:.6f} ({improvement:+.1f}% vs baseline)")
                print(f"   MSE (same): {r['mse_same']:.6f}" if r['mse_same'] else "   MSE (same): N/A")
                print(f"   Parameters: c={r['num_candidates']}, r={r['refinement_steps']}, λ={r['lambda_weight']}")
                if r.get('diagnostics'):
                    print(f"   Deception score: {r['diagnostics']['avg_deception_score']:.4f}")
                    print(f"   Energy vs clean: {r['diagnostics']['energy_vs_clean']:.2f}x")
    
    def save_results_json(self, filename="energy_hnm_sweep_results.json"):
        """
        Save sweep results to JSON using robust file writing.
        Called from run_complete_sweep().
        """
        output_path = Path(self.base_dir) / filename
        if not self._robust_write_file(output_path, self.sweep_results):
            raise OSError(f"Failed to write sweep results to {output_path}")
    
    def generate_heatmaps(self):
        """Generate heatmap visualizations of hyperparameter performance"""
        print(f"\n{'='*80}")
        print("Generating heatmap visualizations...")
        print(f"{'='*80}")
        
        for dataset in TASKS:
            # Filter results for this dataset
            dataset_results = [r for r in self.sweep_results 
                             if r['dataset'] == dataset and r['config_type'] == 'energy_hnm'
                             and r['mse_harder'] is not None]
            
            if not dataset_results:
                print(f"No valid results for {dataset}, skipping heatmap")
                continue
            
            # Create figure with multiple heatmaps
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle(f'Energy-Based HNM Hyperparameter Analysis: {dataset.upper()}', 
                        fontsize=16, y=0.995)
            
            # Prepare data for heatmaps
            # We'll create 2D heatmaps by fixing one parameter
            
            # 1. MSE (harder) vs num_candidates and refinement_steps (averaged over lambda)
            self._plot_heatmap_2d(axes[0, 0], dataset_results, 'num_candidates', 
                                'refinement_steps', 'mse_harder', 
                                'MSE (Harder) vs Candidates & Refinement')
            
            # 2. MSE (harder) vs num_candidates and lambda_weight (averaged over refinement)
            self._plot_heatmap_2d(axes[0, 1], dataset_results, 'num_candidates', 
                                'lambda_weight', 'mse_harder',
                                'MSE (Harder) vs Candidates & Lambda')
            
            # 3. Deception score vs refinement_steps and lambda_weight
            self._plot_heatmap_2d(axes[1, 0], dataset_results, 'refinement_steps',
                                'lambda_weight', 'avg_deception_score',
                                'Deception Score vs Refinement & Lambda',
                                use_diagnostics=True)
            
            # 4. Energy ratio vs num_candidates and refinement_steps
            self._plot_heatmap_2d(axes[1, 1], dataset_results, 'num_candidates',
                                'refinement_steps', 'energy_vs_clean',
                                'Energy Ratio vs Candidates & Refinement',
                                use_diagnostics=True)
            
            plt.tight_layout()
            output_file = self.base_dir / f'heatmaps_{dataset}.png'
            if self._robust_savefig(output_file, dpi=150, bbox_inches='tight'):
                print(f"  ✓ Saved heatmap: {output_file}")
            else:
                print(f"  ✗ Failed to save heatmap: {output_file}")
            plt.close()
    
    def _plot_heatmap_2d(self, ax, results, x_param, y_param, metric, title, use_diagnostics=False):
        """Helper to plot 2D heatmap"""
        # Get unique values for each parameter
        x_values = sorted(list(set(r[x_param] for r in results)))
        y_values = sorted(list(set(r[y_param] for r in results)))
        
        # Create matrix for heatmap
        matrix = np.full((len(y_values), len(x_values)), np.nan)
        
        for r in results:
            x_idx = x_values.index(r[x_param])
            y_idx = y_values.index(r[y_param])
            
            if use_diagnostics and r.get('diagnostics'):
                value = r['diagnostics'].get(metric)
            else:
                value = r.get(metric)
            
            if value is not None:
                if np.isnan(matrix[y_idx, x_idx]):
                    matrix[y_idx, x_idx] = value
                else:
                    # Average if multiple values (shouldn't happen but just in case)
                    matrix[y_idx, x_idx] = (matrix[y_idx, x_idx] + value) / 2
        
        # Plot heatmap
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap='RdYlGn_r' if 'mse' in metric.lower() else 'RdYlGn',
                   xticklabels=x_values, yticklabels=y_values, ax=ax, cbar_kws={'label': metric})
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_title(title)
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report with recommendations"""
        output_file = self.base_dir / 'energy_hnm_sweep_report.md'
        
        lines = [
            "# Energy-Based Hard Negative Mining (HNM) Hyperparameter Sweep Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Training Steps:** {TRAIN_ITERATIONS}",
            f"**Tasks:** {', '.join(TASKS)}",
            "",
            "## Hyperparameter Space",
            "",
            f"- **num_candidates:** {NUM_CANDIDATES_OPTIONS}",
            f"- **refinement_steps:** {REFINEMENT_STEPS_OPTIONS}",
            f"- **lambda_weight:** {LAMBDA_WEIGHT_OPTIONS}",
            f"- **Total configurations per task:** {len(NUM_CANDIDATES_OPTIONS) * len(REFINEMENT_STEPS_OPTIONS) * len(LAMBDA_WEIGHT_OPTIONS)}",
            "",
            "---",
            "",
        ]
        
        for dataset in TASKS:
            lines.extend(self._generate_dataset_section(dataset))
        
        # Add recommendations section
        lines.extend([
            "## Overall Recommendations",
            "",
            self._generate_recommendations(),
            "",
            "---",
            "",
            "## Next Steps",
            "",
            "### Configurations for Full 50k Training",
            "",
            self._generate_next_steps(),
            "",
        ])
        
        if self._robust_write_file(output_file, '\n'.join(lines)):
            print(f"✓ Saved markdown report: {output_file}")
        else:
            print(f"✗ Failed to save markdown report: {output_file}")
    
    def _generate_dataset_section(self, dataset):
        """Generate markdown section for a specific dataset"""
        lines = [
            f"## {dataset.upper()} Results",
            "",
        ]
        
        # Get results
        dataset_results = [r for r in self.sweep_results if r['dataset'] == dataset]
        baseline = [r for r in dataset_results if r['config_type'] == 'baseline'][0]
        hnm_results = [r for r in dataset_results if r['config_type'] == 'energy_hnm']
        
        # Baseline section
        lines.extend([
            "### Baseline Performance",
            "",
            f"- **MSE (same difficulty):** {baseline['mse_same']:.6f}" if baseline['mse_same'] else "- **MSE (same difficulty):** N/A",
            f"- **MSE (harder difficulty):** {baseline['mse_harder']:.6f}" if baseline['mse_harder'] else "- **MSE (harder difficulty):** N/A",
            "",
        ])
        
        if baseline.get('diagnostics'):
            diag = baseline['diagnostics']
            lines.extend([
                "**Baseline Diagnostics:**",
                f"- Average clean energy: {diag['avg_clean_energy']:.4f}",
                f"- Average IRED energy: {diag['avg_ired_energy']:.4f}",
                "",
            ])
        
        # Top configurations
        hnm_results_sorted = sorted([r for r in hnm_results if r['mse_harder'] is not None],
                                   key=lambda x: x['mse_harder'])
        
        lines.extend([
            "### Top 3 Configurations (by MSE harder)",
            "",
        ])
        
        for i, r in enumerate(hnm_results_sorted[:3], 1):
            improvement = ((baseline['mse_harder'] - r['mse_harder']) / baseline['mse_harder'] * 100) if baseline['mse_harder'] and r['mse_harder'] else 0
            
            lines.extend([
                f"#### {i}. Configuration: `{r['config_name']}`",
                "",
                f"- **Parameters:**",
                f"  - num_candidates: {r['num_candidates']}",
                f"  - refinement_steps: {r['refinement_steps']}",
                f"  - lambda_weight: {r['lambda_weight']}",
                f"- **Performance:**",
                f"  - MSE (harder): {r['mse_harder']:.6f} ({improvement:+.1f}% vs baseline)",
                f"  - MSE (same): {r['mse_same']:.6f}" if r['mse_same'] else "  - MSE (same): N/A",
            ])
            
            if r.get('diagnostics'):
                diag = r['diagnostics']
                lines.extend([
                    f"- **Diagnostics:**",
                    f"  - Avg deception score: {diag['avg_deception_score']:.4f}",
                    f"  - Selected energy vs clean: {diag['energy_vs_clean']:.2f}x",
                    f"  - Avg selected energy: {diag['avg_selected_energy']:.4f}",
                    f"  - Avg selected error: {diag['avg_selected_error']:.4f}",
                ])
            
            lines.append("")
        
        # Performance analysis
        lines.extend([
            "### Performance Analysis",
            "",
        ])
        
        # Analyze patterns
        improvements = []
        for r in hnm_results:
            if r['mse_harder'] and baseline['mse_harder']:
                imp = ((baseline['mse_harder'] - r['mse_harder']) / baseline['mse_harder'] * 100)
                improvements.append((r['config_name'], imp, r))
        
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        if improvements:
            best_imp = improvements[0][1]
            worst_imp = improvements[-1][1]
            avg_imp = np.mean([x[1] for x in improvements])
            
            lines.extend([
                f"- **Best improvement:** {best_imp:+.1f}% ({improvements[0][0]})",
                f"- **Worst improvement:** {worst_imp:+.1f}% ({improvements[-1][0]})",
                f"- **Average improvement:** {avg_imp:+.1f}%",
                "",
            ])
            
            # Check if any configs beat baseline
            beating_baseline = [x for x in improvements if x[1] > 0]
            if beating_baseline:
                lines.extend([
                    f"✅ **{len(beating_baseline)}/{len(improvements)} configurations beat baseline**",
                    "",
                ])
            else:
                lines.extend([
                    f"⚠️ **No configurations beat baseline on {dataset}**",
                    "",
                ])
        
        lines.extend([
            "---",
            "",
        ])
        
        return lines
    
    def _generate_recommendations(self):
        """Generate overall recommendations based on sweep results"""
        lines = []
        
        # Analyze patterns across all tasks
        all_hnm_results = [r for r in self.sweep_results if r['config_type'] == 'energy_hnm']
        
        if not all_hnm_results:
            return "No HNM results available for recommendations."
        
        # Group by parameters
        by_candidates = defaultdict(list)
        by_refinement = defaultdict(list)
        by_lambda = defaultdict(list)
        
        for r in all_hnm_results:
            if r['mse_harder']:
                by_candidates[r['num_candidates']].append(r['mse_harder'])
                by_refinement[r['refinement_steps']].append(r['mse_harder'])
                by_lambda[r['lambda_weight']].append(r['mse_harder'])
        
        # Find best values
        best_candidates = min(by_candidates.items(), key=lambda x: np.mean(x[1]))[0] if by_candidates else None
        best_refinement = min(by_refinement.items(), key=lambda x: np.mean(x[1]))[0] if by_refinement else None
        best_lambda = min(by_lambda.items(), key=lambda x: np.mean(x[1]))[0] if by_lambda else None
        
        lines.append("Based on the sweep results across all tasks:\n")
        
        if best_candidates:
            lines.append(f"- **Optimal num_candidates:** {best_candidates}")
            lines.append(f"  - Average MSE: {np.mean(by_candidates[best_candidates]):.6f}")
        
        if best_refinement:
            lines.append(f"- **Optimal refinement_steps:** {best_refinement}")
            lines.append(f"  - Average MSE: {np.mean(by_refinement[best_refinement]):.6f}")
        
        if best_lambda:
            lines.append(f"- **Optimal lambda_weight:** {best_lambda}")
            lines.append(f"  - Average MSE: {np.mean(by_lambda[best_lambda]):.6f}")
        
        lines.append("\n### Key Insights\n")
        
        # Compute cost analysis
        if all_hnm_results:
            costs = [(r['num_candidates'] * r['refinement_steps'], r['mse_harder'], r['config_name']) 
                    for r in all_hnm_results if r['mse_harder']]
            costs.sort(key=lambda x: x[1])  # Sort by MSE
            
            best_performance = costs[0]
            
            # Find cheapest config with similar performance (within 5%)
            best_mse = best_performance[1]
            affordable_configs = [c for c in costs if c[1] <= best_mse * 1.05]
            cheapest_good = min(affordable_configs, key=lambda x: x[0])
            
            lines.append(f"- **Best performance:** {best_performance[2]} (MSE: {best_performance[1]:.6f}, cost: {best_performance[0]}x)")
            lines.append(f"- **Best value:** {cheapest_good[2]} (MSE: {cheapest_good[1]:.6f}, cost: {cheapest_good[0]}x)")
        
        return '\n'.join(lines)
    
    def _generate_next_steps(self):
        """Generate recommended next steps for full training"""
        lines = []
        
        # Find top performers for each task
        for dataset in TASKS:
            dataset_results = [r for r in self.sweep_results 
                             if r['dataset'] == dataset and r['config_type'] == 'energy_hnm'
                             and r['mse_harder'] is not None]
            
            if not dataset_results:
                continue
            
            # Sort by MSE (harder)
            dataset_results.sort(key=lambda x: x['mse_harder'])
            top_config = dataset_results[0]
            
            lines.extend([
                f"### {dataset.upper()}",
                "",
                f"Run full 50k training with: `{top_config['config_name']}`",
                "",
                "```bash",
                f"python train.py \\",
                f"  --dataset {dataset} \\",
                f"  --model mlp \\",
                f"  --batch_size {BATCH_SIZE} \\",
                f"  --diffusion_steps {DIFFUSION_STEPS} \\",
                f"  --rank {RANK} \\",
                f"  --train-steps 50000 \\",
                f"  --use-anm \\",
                f"  --anm-use-energy-hnm \\",
                f"  --hnm-num-candidates {top_config['num_candidates']} \\",
                f"  --hnm-refinement-steps {top_config['refinement_steps']} \\",
                f"  --hnm-lambda-weight {top_config['lambda_weight']}",
                "```",
                "",
                f"Expected improvement over baseline: {((dataset_results[0]['mse_harder'] / next(r['mse_harder'] for r in self.sweep_results if r['dataset'] == dataset and r['config_type'] == 'baseline') - 1) * 100):+.1f}%",
                "",
            ])
        
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Energy-Based HNM Hyperparameter Sweep')
    parser.add_argument('--base-dir', default=None, 
                       help='Base directory for experiments (default: $ENERGY_HNM_EXPERIMENT_DIR or experiments_energy_hnm)')
    parser.add_argument('--force', action='store_true', 
                       help='Force retrain even if models exist')
    args = parser.parse_args()
    
    runner = EnergyHNMSweepRunner(base_dir=args.base_dir)
    results = runner.run_complete_sweep(force_retrain=args.force)
    
    print(f"\n{'#'*80}")
    print("# SWEEP COMPLETE")
    print(f"# Total configurations tested: {len(results)}")
    print(f"# Results saved to: {runner.base_dir}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()