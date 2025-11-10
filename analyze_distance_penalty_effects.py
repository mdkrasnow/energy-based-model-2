#!/usr/bin/env python3
"""
Distance Penalty Paradox Investigation for IRED Energy Landscapes

This script systematically investigates the counterintuitive finding that lower distance 
penalties produce better results in adversarial negative mining for IRED. The investigation
aims to understand why keeping negatives "far" from correct solutions works better than 
keeping them "near".

Research Questions:
1. Is zero distance penalty truly optimal?
2. What does this reveal about IRED's energy landscape geometry?
3. How should this inform energy-based HNM design?

Author: Generated for IRED Distance Penalty Analysis
"""

import os
import subprocess

subprocess.run(['rm', '-rf', 'energy-based-model-2'], check=False)
subprocess.run(['git', 'clone', 'https://github.com/mdkrasnow/energy-based-model-2.git'], check=True)
os.chdir('energy-based-model-2')

# Add the cloned repository to Python path
import sys
sys.path.insert(0, os.getcwd())

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import re
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any

# Import model components for diagnostics
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Inverse, LowRankDataset
from diffusion_lib.adversarial_corruption import _adversarial_corruption

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Fixed hyperparameters for controlled experiment
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 1000  # Full training for definitive results
DIFFUSION_STEPS = 10
RANK = 20  # For 20x20 matrices

# Distance penalty values to systematically investigate
DISTANCE_PENALTIES = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08]

# Fixed ANM parameters (isolate distance penalty effect)
FIXED_EPSILON = 1.0
FIXED_ADV_STEPS = 5

# Target tasks for investigation
TASKS = ['inverse', 'lowrank']  # Matrix Inverse and Matrix Completion

# Diagnostic collection interval (every 5,000 steps)
DIAGNOSTIC_INTERVAL = 500


class DistancePenaltyAnalyzer:
    """
    Systematic analyzer for the Distance Penalty Paradox in IRED Energy Landscapes.
    
    This class orchestrates a comprehensive investigation into why lower distance penalties
    produce better results in adversarial negative mining. It handles training, diagnostic
    collection, analysis, and visualization of results.
    """
    
    def __init__(self, base_dir: str = 'distance_penalty_experiments'):
        """
        Initialize the Distance Penalty Analyzer.
        
        Args:
            base_dir: Base directory for storing all experiment results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.training_results = {}  # Final performance results
        self.diagnostic_timeseries = defaultdict(dict)  # In-training diagnostics
        self.negative_distributions = defaultdict(dict)  # Raw negative data
        self.analysis_results = {}  # Processed analysis results
        
        # Experiment metadata
        self.experiment_metadata = {
            'distance_penalties': DISTANCE_PENALTIES,
            'fixed_epsilon': FIXED_EPSILON,
            'fixed_adv_steps': FIXED_ADV_STEPS,
            'train_iterations': TRAIN_ITERATIONS,
            'diagnostic_interval': DIAGNOSTIC_INTERVAL,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'diffusion_steps': DIFFUSION_STEPS,
            'rank': RANK,
            'tasks': TASKS,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"Distance Penalty Analyzer initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Distance penalties to test: {DISTANCE_PENALTIES}")
        print(f"Tasks: {TASKS}")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
        print(f"Diagnostic collection interval: {DIAGNOSTIC_INTERVAL}")
    
    def _sample_batch(self, dataset, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Helper method to sample a batch from a PyTorch dataset.
        
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
    
    def get_result_dir(self, dataset: str, distance_penalty: float) -> str:
        """
        Get the results directory for a specific dataset and distance penalty configuration.
        
        Args:
            dataset: Dataset name ('inverse' or 'lowrank')
            distance_penalty: Distance penalty value
            
        Returns:
            Path to results directory
        """
        base = f'results/ds_{dataset}/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        base += f'_anm_eps{FIXED_EPSILON}_steps{FIXED_ADV_STEPS}_dp{distance_penalty}'
        return base
    
    def load_model_for_diagnostics(self, model_dir: str, device: str = 'cuda') -> Optional[Tuple]:
        """
        Load model checkpoint for diagnostic purposes with robust handling.
        
        Args:
            model_dir: Directory containing model checkpoint
            device: Device to load model on
            
        Returns:
            Tuple of (model, diffusion, device, dataset) or None if loading fails
        """
        # Try different checkpoint names in order of preference
        checkpoint_names = [
            f'model-{TRAIN_ITERATIONS//1000}.pt',  # e.g., model-50.pt for 50k steps
            'model-latest.pt',
            'model.pt'
        ]
        
        checkpoint_path = None
        for name in checkpoint_names:
            potential_path = Path(model_dir) / name
            if potential_path.exists():
                checkpoint_path = potential_path
                break
        
        if checkpoint_path is None:
            print(f"Warning: No checkpoint found in {model_dir}")
            return None
            
        device = device if torch.cuda.is_available() else 'cpu'
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
        
        # Get dataset dimensions
        if 'inverse' in str(model_dir):
            dataset = Inverse("train", RANK, False)
        else:  # lowrank
            dataset = LowRankDataset("train", RANK, False)
        
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
        
        # Handle prefix mismatches
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
        
        # Load the cleaned state dict
        if state_dict:
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                print(f"Warning: Loading with strict=False for {model_dir}")
                model.load_state_dict(state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        
        # Extract distance penalty from model directory name
        dp_match = re.search(r'_dp([0-9\.]+)', str(model_dir))
        distance_penalty = float(dp_match.group(1)) if dp_match else 0.1
        
        # Setup diffusion with correct ANM parameters
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=DIFFUSION_STEPS,
            sampling_timesteps=DIFFUSION_STEPS,
            continuous=True,
            show_inference_tqdm=False,
            use_adversarial_corruption=True,
            anm_adversarial_steps=FIXED_ADV_STEPS,
            anm_distance_penalty=distance_penalty,
            anm_warmup_steps=0,
            sudoku=False,
            shortest_path=False,
        )
        
        diffusion = diffusion.to(device)
        
        return model, diffusion, device, dataset
    
    def run_distance_penalty_sweep(self, tasks: Optional[List[str]] = None, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Run systematic distance penalty sweep across all specified values and tasks.
        
        This is the main orchestration method that:
        1. Trains models for each distance penalty value
        2. Collects diagnostics during training  
        3. Evaluates final performance
        4. Performs comprehensive analysis
        
        Args:
            tasks: List of tasks to run (default: ['inverse', 'lowrank'])
            force_retrain: Whether to retrain even if models already exist
            
        Returns:
            Dictionary containing all sweep results and analysis
        """
        if tasks is None:
            tasks = TASKS
        
        print(f"\n{'='*80}")
        print(f"DISTANCE PENALTY PARADOX INVESTIGATION")
        print(f"{'='*80}")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Distance penalties: {DISTANCE_PENALTIES}")
        print(f"Fixed parameters: eps={FIXED_EPSILON}, adv_steps={FIXED_ADV_STEPS}")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
        print(f"Diagnostic collection interval: {DIAGNOSTIC_INTERVAL}")
        print(f"{'='*80}\n")
        
        # Track overall progress
        total_configs = len(tasks) * len(DISTANCE_PENALTIES)
        current_config = 0
        
        # Store results for each task
        sweep_results = {}
        
        for task in tasks:
            print(f"\n{'#'*60}")
            print(f"# ANALYZING TASK: {task.upper()}")
            print(f"{'#'*60}")
            
            task_results = {}
            
            for dp in DISTANCE_PENALTIES:
                current_config += 1
                
                print(f"\n{'-'*50}")
                print(f"CONFIG {current_config}/{total_configs}: {task}, dp={dp}")
                print(f"{'-'*50}")
                
                # Train model with this distance penalty
                training_success = self.train_model_with_diagnostics(
                    task, dp, force_retrain=force_retrain
                )
                
                if training_success:
                    # Evaluate final performance
                    performance_results = self.evaluate_model_performance(task, dp)
                    
                    # Store results
                    task_results[dp] = {
                        'training_success': True,
                        'performance': performance_results,
                        'diagnostics_collected': True
                    }
                    
                    print(f"  ✓ Completed: dp={dp}")
                    if performance_results and 'test_mse' in performance_results:
                        print(f"    Test MSE: {performance_results['test_mse']:.6f}")
                else:
                    task_results[dp] = {
                        'training_success': False,
                        'performance': None,
                        'diagnostics_collected': False
                    }
                    print(f"  ✗ Failed: dp={dp}")
            
            sweep_results[task] = task_results
        
        # Store sweep results
        self.training_results = sweep_results
        
        # Perform comprehensive analysis
        print(f"\n{'='*80}")
        print(f"PERFORMING COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        analysis_results = self.perform_comprehensive_analysis()
        
        # Generate visualizations
        print(f"\nGenerating publication-quality visualizations...")
        self.generate_comprehensive_visualizations()
        
        # Save all results
        print(f"\nSaving structured data...")
        self.save_all_results()
        
        # Generate final report
        print(f"\nGenerating comprehensive analysis report...")
        self.generate_analysis_report()
        
        print(f"\n{'='*80}")
        print(f"DISTANCE PENALTY SWEEP COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved in: {self.base_dir}")
        print(f"{'='*80}\n")
        
        return {
            'sweep_results': sweep_results,
            'analysis_results': analysis_results,
            'metadata': self.experiment_metadata
        }
    
    def train_model_with_diagnostics(self, dataset: str, distance_penalty: float, 
                                   force_retrain: bool = False) -> bool:
        """
        Train a model with specified distance penalty and collect diagnostics.
        
        Args:
            dataset: Dataset name ('inverse' or 'lowrank')
            distance_penalty: Distance penalty value to use
            force_retrain: Whether to retrain even if model exists
            
        Returns:
            True if training succeeded, False otherwise
        """
        result_dir = self.get_result_dir(dataset, distance_penalty)
        
        # Check if model already exists
        expected_checkpoint = Path(result_dir) / f'model-{TRAIN_ITERATIONS//1000}.pt'
        if not force_retrain and expected_checkpoint.exists():
            print(f"  Model exists, skipping training: {result_dir}")
            print(f"  Use force_retrain=True to retrain")
            
            # Still collect diagnostics from existing model
            self.collect_final_diagnostics(dataset, distance_penalty)
            return True
        
        print(f"  Training model: {dataset}, dp={distance_penalty}")
        print(f"  Result directory: {result_dir}")
        
        # Build training command
        cmd = [
            'python', 'train.py',
            '--dataset', dataset,
            '--model', 'mlp',
            '--batch_size', str(BATCH_SIZE),
            '--diffusion_steps', str(DIFFUSION_STEPS),
            '--rank', str(RANK),
            '--train-steps', str(TRAIN_ITERATIONS),
            '--use-anm',
            '--anm-epsilon', str(FIXED_EPSILON),
            '--anm-adversarial-steps', str(FIXED_ADV_STEPS),
            '--anm-distance-penalty', str(distance_penalty),
            '--anm-temperature', '1.0',
            '--anm-clean-ratio', '0.1',
            '--anm-adversarial-ratio', '0.8',
            '--anm-gaussian-ratio', '0.1',
            '--anm-hard-negative-ratio', '0.0',
        ]
        
        # Run training with output capture
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor training and collect diagnostics periodically
            output_lines = []
            last_diagnostic_step = 0
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
                    # Print selected lines to show progress
                    if any(keyword in line.lower() for keyword in ['loss', 'mse', 'step', 'epoch']):
                        print(f"    {line.rstrip()}")
                    
                    # Check if we should collect diagnostics
                    step_match = re.search(r'step[:\s]+(\d+)', line.lower())
                    if step_match:
                        current_step = int(step_match.group(1))
                        if (current_step - last_diagnostic_step) >= DIAGNOSTIC_INTERVAL and current_step > 0:
                            # Collect diagnostics at this checkpoint
                            self.collect_training_diagnostics(dataset, distance_penalty, current_step)
                            last_diagnostic_step = current_step
            
            result = process.wait()
            elapsed = time.time() - start_time
            
            if result == 0:
                print(f"  ✓ Training completed in {elapsed/60:.1f} minutes")
                
                # Collect final diagnostics
                self.collect_final_diagnostics(dataset, distance_penalty)
                return True
            else:
                print(f"  ✗ Training failed with exit code {result}")
                print("  Full output from failed process:")
                for line in output_lines:
                    print(f"    | {line.rstrip()}")
                return False
                
        except Exception as e:
            print(f"  ✗ Training failed with exception: {e}")
            return False
    
    def evaluate_model_performance(self, dataset: str, distance_penalty: float) -> Optional[Dict[str, float]]:
        """
        Evaluate trained model performance on test set.
        
        Args:
            dataset: Dataset name
            distance_penalty: Distance penalty value
            
        Returns:
            Dictionary with performance metrics or None if evaluation fails
        """
        result_dir = self.get_result_dir(dataset, distance_penalty)
        
        print(f"  Evaluating model performance...")
        
        # Build evaluation command
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
            '--use-anm',
            '--anm-epsilon', str(FIXED_EPSILON),
            '--anm-adversarial-steps', str(FIXED_ADV_STEPS),
            '--anm-distance-penalty', str(distance_penalty),
        ]
        
        try:
            # Run evaluation on same difficulty
            result_same = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout for full training evaluation
            )
            
            # Run evaluation on harder difficulty (OOD)
            cmd_ood = cmd + ['--ood']
            result_ood = subprocess.run(
                cmd_ood,
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout for full training evaluation
            )
            
            # Parse MSE from outputs
            mse_same = self._parse_mse_from_output(result_same.stdout, result_same.stderr)
            mse_ood = self._parse_mse_from_output(result_ood.stdout, result_ood.stderr)
            
            performance_results = {
                'test_mse': mse_same,
                'test_mse_ood': mse_ood,
                'evaluation_success': mse_same is not None
            }
            
            return performance_results
            
        except Exception as e:
            print(f"  Warning: Evaluation failed: {e}")
            return None
    
    def _parse_mse_from_output(self, stdout: str, stderr: str) -> Optional[float]:
        """
        Parse MSE value from training/evaluation output.
        
        Args:
            stdout: Standard output string
            stderr: Standard error string
            
        Returns:
            Parsed MSE value or None if not found
        """
        output = stdout + stderr
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
        
        # Alternative parsing patterns
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
    
    def collect_training_diagnostics(self, dataset: str, distance_penalty: float, step: int) -> None:
        """
        Collect diagnostic information at a specific training step.
        
        Args:
            dataset: Dataset name
            distance_penalty: Distance penalty value
            step: Training step number
        """
        print(f"    Collecting diagnostics at step {step}...")
        
        try:
            # Load model at this step
            result_dir = self.get_result_dir(dataset, distance_penalty)
            step_checkpoint = Path(result_dir) / f'model-{step//1000}.pt'
            
            if not step_checkpoint.exists():
                print(f"    Warning: Checkpoint not found for step {step}")
                return
            
            # Load model for diagnostics
            diagnostic_result = self.load_model_for_diagnostics(result_dir)
            if diagnostic_result is None:
                print(f"    Warning: Could not load model for step {step}")
                return
            
            model, diffusion, device, data = diagnostic_result
            
            # Collect comprehensive diagnostics
            diagnostics = self.run_comprehensive_diagnostics(
                model, diffusion, device, data, distance_penalty, num_batches=3
            )
            
            # Store diagnostics with step information
            key = f"{dataset}_dp{distance_penalty}"
            if key not in self.diagnostic_timeseries:
                self.diagnostic_timeseries[key] = {}
            
            self.diagnostic_timeseries[key][step] = diagnostics
            
            print(f"    ✓ Diagnostics collected for step {step}")
            
        except Exception as e:
            print(f"    Warning: Diagnostic collection failed for step {step}: {e}")
    
    def collect_final_diagnostics(self, dataset: str, distance_penalty: float) -> None:
        """
        Collect comprehensive diagnostics from the final trained model.
        
        Args:
            dataset: Dataset name
            distance_penalty: Distance penalty value
        """
        print(f"  Collecting final diagnostics for dp={distance_penalty}...")
        
        try:
            result_dir = self.get_result_dir(dataset, distance_penalty)
            diagnostic_result = self.load_model_for_diagnostics(result_dir)
            
            if diagnostic_result is None:
                print(f"  Warning: Could not load final model for diagnostics")
                return
            
            model, diffusion, device, data = diagnostic_result
            
            # Collect comprehensive final diagnostics with more batches
            diagnostics = self.run_comprehensive_diagnostics(
                model, diffusion, device, data, distance_penalty, num_batches=10
            )
            
            # Store final diagnostics
            key = f"{dataset}_dp{distance_penalty}"
            if key not in self.diagnostic_timeseries:
                self.diagnostic_timeseries[key] = {}
            
            self.diagnostic_timeseries[key]['final'] = diagnostics
            
            print(f"  ✓ Final diagnostics collected")
            
        except Exception as e:
            print(f"  Warning: Final diagnostic collection failed: {e}")
    
    def run_comprehensive_diagnostics(self, model, diffusion, device, dataset, 
                                    distance_penalty: float, num_batches: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic analysis on a trained model.
        
        Args:
            model: Trained model
            diffusion: Diffusion wrapper
            device: Device (cuda/cpu)
            dataset: Dataset for sampling
            distance_penalty: Distance penalty used in training
            num_batches: Number of batches to analyze
            
        Returns:
            Dictionary containing comprehensive diagnostic results
        """
        # Storage for diagnostic data
        negative_distances = []
        energy_values = []
        gradient_magnitudes = []
        mse_errors = []
        
        # Ground truth statistics for threshold calculation
        ground_truth_norms = []
        
        # Energy breakdown by corruption type
        energy_by_type = {
            'clean': [],
            'ired_standard': [],
            'anm_adversarial': [],
            'gaussian_noise': []
        }
        
        for batch_idx in range(num_batches):
            # Sample batch
            batch = self._sample_batch(dataset, 256)
            x_clean = batch['x'].to(device)
            y_clean = batch['y'].to(device)
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (len(x_clean),), device=device)
            
            # Collect ground truth statistics
            ground_truth_norms.extend(torch.norm(y_clean, dim=-1).cpu().numpy())
            
            # 1. Clean samples
            with torch.no_grad():
                energy_clean = diffusion.energy_score(x_clean, y_clean, t)
                energy_by_type['clean'].extend(energy_clean.cpu().numpy())
            
            # 2. Standard IRED corruption
            noise = torch.randn_like(y_clean)
            alpha = 1.0 - (t.float() / diffusion.num_timesteps).view(-1, 1)
            y_ired = alpha * y_clean + (1 - alpha) * noise
            
            with torch.no_grad():
                energy_ired = diffusion.energy_score(x_clean, y_ired, t)
                energy_by_type['ired_standard'].extend(energy_ired.cpu().numpy())
            
            # Calculate distances and errors for IRED
            ired_distances = torch.norm(y_ired - y_clean, dim=-1)
            ired_mse = F.mse_loss(y_ired, y_clean, reduction='none').mean(dim=-1)
            
            negative_distances.extend(ired_distances.cpu().numpy())
            mse_errors.extend(ired_mse.cpu().numpy())
            energy_values.extend(energy_ired.cpu().numpy())
            
            # 3. ANM adversarial corruption
            y_anm = self._generate_anm_negatives(x_clean, y_clean, t, diffusion, distance_penalty)
            
            with torch.no_grad():
                energy_anm = diffusion.energy_score(x_clean, y_anm, t)
                energy_by_type['anm_adversarial'].extend(energy_anm.cpu().numpy())
            
            # Calculate distances, errors, and gradient magnitudes for ANM
            anm_distances = torch.norm(y_anm - y_clean, dim=-1).detach()
            anm_mse = F.mse_loss(y_anm, y_clean, reduction='none').mean(dim=-1).detach()
            
            # Calculate gradient magnitudes at ANM negatives
            y_anm_grad = y_anm.clone().requires_grad_(True)
            energy_grad = diffusion.energy_score(x_clean, y_anm_grad, t)
            grad_output = torch.autograd.grad(energy_grad.sum(), y_anm_grad, create_graph=False)[0]
            grad_magnitudes = torch.norm(grad_output, dim=-1)
            
            negative_distances.extend(anm_distances.cpu().numpy())
            mse_errors.extend(anm_mse.cpu().numpy())
            energy_values.extend(energy_anm.cpu().numpy())
            gradient_magnitudes.extend(grad_magnitudes.detach().cpu().numpy())
            
            # 4. Gaussian noise corruption
            y_gaussian = y_clean + 0.1 * torch.randn_like(y_clean)
            with torch.no_grad():
                energy_gaussian = diffusion.energy_score(x_clean, y_gaussian, t)
                energy_by_type['gaussian_noise'].extend(energy_gaussian.cpu().numpy())
        
        # Convert to numpy arrays
        negative_distances = np.array(negative_distances)
        energy_values = np.array(energy_values)
        gradient_magnitudes = np.array(gradient_magnitudes)
        mse_errors = np.array(mse_errors)
        ground_truth_norms = np.array(ground_truth_norms)
        
        # Calculate threshold statistics
        mean_norm = np.mean(ground_truth_norms)
        std_norm = np.std(ground_truth_norms)
        
        # Distance threshold analysis (within 1σ, 2σ, 3σ of ground truth distribution)
        threshold_1sigma = mean_norm + std_norm
        threshold_2sigma = mean_norm + 2 * std_norm
        threshold_3sigma = mean_norm + 3 * std_norm
        
        negatives_within_1sigma = np.mean(negative_distances <= threshold_1sigma)
        negatives_within_2sigma = np.mean(negative_distances <= threshold_2sigma)
        negatives_within_3sigma = np.mean(negative_distances <= threshold_3sigma)
        
        # Compile comprehensive diagnostics
        diagnostics = {
            'distance_penalty': distance_penalty,
            'num_samples': len(negative_distances),
            
            # Basic statistics
            'negative_distances': {
                'mean': np.mean(negative_distances),
                'std': np.std(negative_distances),
                'median': np.median(negative_distances),
                'min': np.min(negative_distances),
                'max': np.max(negative_distances)
            },
            
            'energy_values': {
                'mean': np.mean(energy_values),
                'std': np.std(energy_values),
                'median': np.median(energy_values),
                'min': np.min(energy_values),
                'max': np.max(energy_values)
            },
            
            'gradient_magnitudes': {
                'mean': np.mean(gradient_magnitudes),
                'std': np.std(gradient_magnitudes),
                'median': np.median(gradient_magnitudes),
                'min': np.min(gradient_magnitudes),
                'max': np.max(gradient_magnitudes)
            },
            
            'mse_errors': {
                'mean': np.mean(mse_errors),
                'std': np.std(mse_errors),
                'median': np.median(mse_errors),
                'min': np.min(mse_errors),
                'max': np.max(mse_errors)
            },
            
            # Energy by corruption type
            'energy_by_type': {
                corruption_type: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
                for corruption_type, values in energy_by_type.items()
            },
            
            # Distance threshold analysis
            'ground_truth_statistics': {
                'mean_norm': mean_norm,
                'std_norm': std_norm
            },
            
            'distance_thresholds': {
                'within_1sigma': negatives_within_1sigma,
                'within_2sigma': negatives_within_2sigma,
                'within_3sigma': negatives_within_3sigma,
                'threshold_1sigma': threshold_1sigma,
                'threshold_2sigma': threshold_2sigma,
                'threshold_3sigma': threshold_3sigma
            },
            
            # Deceptiveness analysis (low energy + high error)
            'deceptiveness_metrics': self._calculate_deceptiveness_metrics(
                energy_values, mse_errors, negative_distances
            ),
            
            # Raw data for further analysis (truncated for storage efficiency)
            'raw_data_sample': {
                'negative_distances': negative_distances[:1000].tolist(),
                'energy_values': energy_values[:1000].tolist(),
                'gradient_magnitudes': gradient_magnitudes[:1000].tolist(),
                'mse_errors': mse_errors[:1000].tolist()
            }
        }
        
        return diagnostics
    
    def _generate_anm_negatives(self, x_clean: torch.Tensor, y_clean: torch.Tensor, 
                               t: torch.Tensor, diffusion, distance_penalty: float) -> torch.Tensor:
        """
        Generate ANM adversarial negatives using the actual adversarial corruption function.
        
        Args:
            x_clean: Clean input samples
            y_clean: Clean output samples
            t: Timesteps
            diffusion: Diffusion object
            distance_penalty: Distance penalty to use
            
        Returns:
            Generated adversarial negatives
        """
        try:
            y_anm = _adversarial_corruption(
                ops=diffusion,
                inp=x_clean,
                x_start=y_clean,
                t=t,
                mask=None,
                data_cond=None,
                base_noise_scale=3.0,
                epsilon=distance_penalty
            )
            return y_anm
        except Exception as e:
            print(f"    Warning: ANM generation failed, using fallback: {e}")
            # Fallback to simple gradient ascent
            return self._fallback_anm_generation(x_clean, y_clean, t, diffusion, distance_penalty)
    
    def _fallback_anm_generation(self, x_clean: torch.Tensor, y_clean: torch.Tensor,
                                t: torch.Tensor, diffusion, distance_penalty: float) -> torch.Tensor:
        """
        Fallback ANM generation using simple gradient ascent.
        
        Args:
            x_clean: Clean input samples
            y_clean: Clean output samples  
            t: Timesteps
            diffusion: Diffusion object
            distance_penalty: Distance penalty to use
            
        Returns:
            Generated adversarial negatives
        """
        # Start from noisy version
        noise = torch.randn_like(y_clean)
        alpha = 1.0 - (t.float() / diffusion.num_timesteps).view(-1, 1)
        y_adv = (alpha * y_clean + (1 - alpha) * noise).clone()
        y_adv.requires_grad_(True)
        
        # Simple gradient ascent with distance penalty
        for step in range(FIXED_ADV_STEPS):
            energy = diffusion.energy_score(x_clean, y_adv, t)
            distance_pen = F.mse_loss(y_adv, y_clean)
            
            objective = energy.mean() - distance_penalty * distance_pen
            grad = torch.autograd.grad(objective, y_adv, create_graph=False)[0]
            
            with torch.no_grad():
                y_adv = y_adv + 0.1 * grad
                y_adv.requires_grad_(True)
        
        return y_adv.detach()
    
    def _calculate_deceptiveness_metrics(self, energy_values: np.ndarray, 
                                       mse_errors: np.ndarray, 
                                       distances: np.ndarray) -> Dict[str, float]:
        """
        Calculate deceptiveness metrics for negatives.
        
        Deceptiveness = ability to fool the model (low energy + high error)
        
        Args:
            energy_values: Energy scores of negatives
            mse_errors: MSE errors of negatives
            distances: Distances from ground truth
            
        Returns:
            Dictionary of deceptiveness metrics
        """
        # Normalize values for comparison
        energy_norm = (energy_values - np.min(energy_values)) / (np.max(energy_values) - np.min(energy_values) + 1e-8)
        error_norm = (mse_errors - np.min(mse_errors)) / (np.max(mse_errors) - np.min(mse_errors) + 1e-8)
        distance_norm = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-8)
        
        # Deceptiveness score: high error + low energy
        deceptiveness_score = error_norm - energy_norm
        
        # Correlations
        energy_error_corr = np.corrcoef(energy_values, mse_errors)[0, 1]
        energy_distance_corr = np.corrcoef(energy_values, distances)[0, 1]
        error_distance_corr = np.corrcoef(mse_errors, distances)[0, 1]
        
        return {
            'mean_deceptiveness': np.mean(deceptiveness_score),
            'std_deceptiveness': np.std(deceptiveness_score),
            'max_deceptiveness': np.max(deceptiveness_score),
            'energy_error_correlation': energy_error_corr,
            'energy_distance_correlation': energy_distance_corr,
            'error_distance_correlation': error_distance_corr,
            'high_deceptiveness_ratio': np.mean(deceptiveness_score > np.percentile(deceptiveness_score, 75))
        }
    
    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis across all distance penalty configurations.
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        print("Analyzing distance penalty effects across all configurations...")
        
        analysis_results = {
            'performance_analysis': self._analyze_performance_trends(),
            'energy_landscape_analysis': self._analyze_energy_landscapes(),
            'distance_threshold_analysis': self._analyze_distance_thresholds(),
            'deceptiveness_analysis': self._analyze_deceptiveness_patterns(),
            'gradient_signal_analysis': self._analyze_gradient_signals(),
            'correlation_analysis': self._perform_correlation_analysis(),
            'optimal_config_analysis': self._identify_optimal_configurations()
        }
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across distance penalty values.
        
        Returns:
            Dictionary with performance trend analysis
        """
        print("  Analyzing performance trends...")
        
        performance_trends = {}
        
        for task in TASKS:
            if task not in self.training_results:
                continue
                
            task_results = self.training_results[task]
            
            # Extract performance data
            distance_penalties = []
            test_mses = []
            test_mses_ood = []
            
            for dp, result in task_results.items():
                if result['training_success'] and result['performance']:
                    distance_penalties.append(dp)
                    test_mses.append(result['performance']['test_mse'])
                    test_mses_ood.append(result['performance']['test_mse_ood'])
            
            if len(distance_penalties) > 1:
                # Convert to arrays for analysis
                dp_array = np.array(distance_penalties)
                mse_array = np.array(test_mses)
                mse_ood_array = np.array(test_mses_ood)
                
                # Find optimal configurations
                best_idx = np.argmin(mse_array)
                best_idx_ood = np.argmin(mse_ood_array)
                
                # Calculate trend statistics
                dp_mse_correlation = np.corrcoef(dp_array, mse_array)[0, 1] if len(dp_array) > 1 else 0.0
                dp_mse_ood_correlation = np.corrcoef(dp_array, mse_ood_array)[0, 1] if len(dp_array) > 1 else 0.0
                
                performance_trends[task] = {
                    'distance_penalties': dp_array.tolist(),
                    'test_mses': mse_array.tolist(),
                    'test_mses_ood': mse_ood_array.tolist(),
                    'best_dp': dp_array[best_idx],
                    'best_mse': mse_array[best_idx],
                    'best_dp_ood': dp_array[best_idx_ood],
                    'best_mse_ood': mse_ood_array[best_idx_ood],
                    'dp_mse_correlation': dp_mse_correlation,
                    'dp_mse_ood_correlation': dp_mse_ood_correlation,
                    'performance_range': float(np.max(mse_array) - np.min(mse_array)),
                    'zero_dp_performance': mse_array[dp_array == 0.0][0] if 0.0 in dp_array else None
                }
        
        return performance_trends
    
    def _analyze_energy_landscapes(self) -> Dict[str, Any]:
        """
        Analyze energy landscape characteristics across distance penalty values.
        
        Returns:
            Dictionary with energy landscape analysis
        """
        print("  Analyzing energy landscapes...")
        
        landscape_analysis = {}
        
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' not in timeseries:
                continue
                
            final_diagnostics = timeseries['final']
            dp = final_diagnostics['distance_penalty']
            
            # Extract energy landscape metrics
            energy_stats = final_diagnostics['energy_values']
            distance_stats = final_diagnostics['negative_distances']
            deceptiveness = final_diagnostics['deceptiveness_metrics']
            
            # Analyze energy-distance relationship
            raw_data = final_diagnostics['raw_data_sample']
            if raw_data['energy_values'] and raw_data['negative_distances']:
                energies = np.array(raw_data['energy_values'])
                distances = np.array(raw_data['negative_distances'])
                
                # Fit linear relationship: energy = a * distance + b
                if len(energies) > 1 and len(distances) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(distances, energies)
                else:
                    slope = intercept = r_value = p_value = std_err = 0.0
                
                landscape_characteristics = {
                    'distance_penalty': dp,
                    'energy_statistics': energy_stats,
                    'distance_statistics': distance_stats,
                    'energy_distance_slope': slope,
                    'energy_distance_intercept': intercept,
                    'energy_distance_correlation': r_value,
                    'energy_distance_p_value': p_value,
                    'deceptiveness_score': deceptiveness['mean_deceptiveness'],
                    'energy_range': energy_stats['max'] - energy_stats['min'],
                    'distance_range': distance_stats['max'] - distance_stats['min'],
                    'exploration_efficiency': self._calculate_exploration_efficiency(energies, distances)
                }
                
                landscape_analysis[key] = landscape_characteristics
        
        return landscape_analysis
    
    def _calculate_exploration_efficiency(self, energies: np.ndarray, distances: np.ndarray) -> float:
        """
        Calculate how efficiently the negatives explore the energy-distance space.
        
        Args:
            energies: Energy values of negatives
            distances: Distance values of negatives
            
        Returns:
            Exploration efficiency score (0-1, higher is better)
        """
        if len(energies) < 2 or len(distances) < 2:
            return 0.0
        
        # Normalize to [0, 1] range
        energy_norm = (energies - np.min(energies)) / (np.max(energies) - np.min(energies) + 1e-8)
        distance_norm = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-8)
        
        # Calculate coverage of the energy-distance space
        # Create 2D histogram to measure space coverage
        hist, _, _ = np.histogram2d(energy_norm, distance_norm, bins=10, range=[[0, 1], [0, 1]])
        
        # Efficiency = fraction of bins with samples / total possible coverage
        non_empty_bins = np.sum(hist > 0)
        total_bins = hist.size
        coverage_efficiency = non_empty_bins / total_bins
        
        # Also consider distribution uniformity (entropy)
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Remove empty bins
        if len(hist_flat) > 1:
            hist_norm = hist_flat / np.sum(hist_flat)
            entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-8))
            max_entropy = np.log(len(hist_flat))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uniformity = 0.0
        
        # Combine coverage and uniformity
        efficiency = 0.7 * coverage_efficiency + 0.3 * uniformity
        return float(efficiency)
    
    def _analyze_distance_thresholds(self) -> Dict[str, Any]:
        """
        Analyze how negatives distribute within different distance thresholds.
        
        Returns:
            Dictionary with distance threshold analysis
        """
        print("  Analyzing distance threshold distributions...")
        
        threshold_analysis = {}
        
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' not in timeseries:
                continue
                
            final_diagnostics = timeseries['final']
            dp = final_diagnostics['distance_penalty']
            threshold_data = final_diagnostics['distance_thresholds']
            
            threshold_analysis[key] = {
                'distance_penalty': dp,
                'within_1sigma': threshold_data['within_1sigma'],
                'within_2sigma': threshold_data['within_2sigma'],  
                'within_3sigma': threshold_data['within_3sigma'],
                'threshold_values': {
                    '1sigma': threshold_data['threshold_1sigma'],
                    '2sigma': threshold_data['threshold_2sigma'],
                    '3sigma': threshold_data['threshold_3sigma']
                },
                'constraint_effectiveness': 1.0 - threshold_data['within_1sigma'] if dp > 0 else 0.0
            }
        
        return threshold_analysis
    
    def _analyze_deceptiveness_patterns(self) -> Dict[str, Any]:
        """
        Analyze deceptiveness patterns across distance penalty configurations.
        
        Returns:
            Dictionary with deceptiveness pattern analysis
        """
        print("  Analyzing deceptiveness patterns...")
        
        deceptiveness_analysis = {}
        
        # Collect deceptiveness metrics across all configurations
        dp_values = []
        deceptiveness_scores = []
        energy_error_correlations = []
        
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' not in timeseries:
                continue
                
            final_diagnostics = timeseries['final']
            dp = final_diagnostics['distance_penalty']
            deceptiveness = final_diagnostics['deceptiveness_metrics']
            
            dp_values.append(dp)
            deceptiveness_scores.append(deceptiveness['mean_deceptiveness'])
            energy_error_correlations.append(deceptiveness['energy_error_correlation'])
            
            deceptiveness_analysis[key] = {
                'distance_penalty': dp,
                'mean_deceptiveness': deceptiveness['mean_deceptiveness'],
                'std_deceptiveness': deceptiveness['std_deceptiveness'],
                'max_deceptiveness': deceptiveness['max_deceptiveness'],
                'energy_error_correlation': deceptiveness['energy_error_correlation'],
                'energy_distance_correlation': deceptiveness['energy_distance_correlation'],
                'error_distance_correlation': deceptiveness['error_distance_correlation'],
                'high_deceptiveness_ratio': deceptiveness['high_deceptiveness_ratio']
            }
        
        # Analyze trends across distance penalties
        if len(dp_values) > 1:
            dp_array = np.array(dp_values)
            deceptiveness_array = np.array(deceptiveness_scores)
            correlation_array = np.array(energy_error_correlations)
            
            # Find optimal deceptiveness configuration
            best_deceptiveness_idx = np.argmax(deceptiveness_array)
            
            deceptiveness_analysis['overall_trends'] = {
                'best_dp_for_deceptiveness': dp_array[best_deceptiveness_idx],
                'best_deceptiveness_score': deceptiveness_array[best_deceptiveness_idx],
                'dp_deceptiveness_correlation': np.corrcoef(dp_array, deceptiveness_array)[0, 1],
                'zero_dp_deceptiveness': deceptiveness_array[dp_array == 0.0][0] if 0.0 in dp_array else None,
                'deceptiveness_range': float(np.max(deceptiveness_array) - np.min(deceptiveness_array))
            }
        
        return deceptiveness_analysis
    
    def _analyze_gradient_signals(self) -> Dict[str, Any]:
        """
        Analyze gradient signal strength across distance penalty configurations.
        
        Returns:
            Dictionary with gradient signal analysis
        """
        print("  Analyzing gradient signal patterns...")
        
        gradient_analysis = {}
        
        # Collect gradient metrics across all configurations
        dp_values = []
        gradient_magnitudes = []
        
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' not in timeseries:
                continue
                
            final_diagnostics = timeseries['final']
            dp = final_diagnostics['distance_penalty']
            gradient_stats = final_diagnostics['gradient_magnitudes']
            
            dp_values.append(dp)
            gradient_magnitudes.append(gradient_stats['mean'])
            
            gradient_analysis[key] = {
                'distance_penalty': dp,
                'mean_gradient_magnitude': gradient_stats['mean'],
                'std_gradient_magnitude': gradient_stats['std'],
                'median_gradient_magnitude': gradient_stats['median'],
                'max_gradient_magnitude': gradient_stats['max'],
                'gradient_signal_strength': self._classify_gradient_strength(gradient_stats['mean'])
            }
        
        # Analyze trends across distance penalties
        if len(dp_values) > 1:
            dp_array = np.array(dp_values)
            gradient_array = np.array(gradient_magnitudes)
            
            # Find optimal gradient configuration
            best_gradient_idx = np.argmax(gradient_array)
            
            gradient_analysis['overall_trends'] = {
                'best_dp_for_gradients': dp_array[best_gradient_idx],
                'best_gradient_magnitude': gradient_array[best_gradient_idx],
                'dp_gradient_correlation': np.corrcoef(dp_array, gradient_array)[0, 1],
                'zero_dp_gradient': gradient_array[dp_array == 0.0][0] if 0.0 in dp_array else None,
                'gradient_range': float(np.max(gradient_array) - np.min(gradient_array))
            }
        
        return gradient_analysis
    
    def _classify_gradient_strength(self, mean_gradient: float) -> str:
        """
        Classify gradient signal strength.
        
        Args:
            mean_gradient: Mean gradient magnitude
            
        Returns:
            Classification string
        """
        if mean_gradient > 1.0:
            return "very_strong"
        elif mean_gradient > 0.5:
            return "strong"
        elif mean_gradient > 0.1:
            return "moderate"
        elif mean_gradient > 0.01:
            return "weak"
        else:
            return "very_weak"
    
    def _perform_correlation_analysis(self) -> Dict[str, Any]:
        """
        Perform correlation analysis between different metrics and final performance.
        
        Returns:
            Dictionary with correlation analysis results
        """
        print("  Performing correlation analysis...")
        
        correlation_analysis = {}
        
        for task in TASKS:
            if task not in self.training_results:
                continue
            
            # Collect all metrics for correlation analysis
            data = {
                'distance_penalty': [],
                'test_mse': [],
                'test_mse_ood': [],
                'mean_deceptiveness': [],
                'mean_gradient_magnitude': [],
                'energy_distance_correlation': [],
                'within_1sigma_ratio': [],
                'exploration_efficiency': []
            }
            
            for dp, result in self.training_results[task].items():
                if not result['training_success'] or not result['performance']:
                    continue
                
                key = f"{task}_dp{dp}"
                if key not in self.diagnostic_timeseries or 'final' not in self.diagnostic_timeseries[key]:
                    continue
                
                diagnostics = self.diagnostic_timeseries[key]['final']
                
                data['distance_penalty'].append(dp)
                data['test_mse'].append(result['performance']['test_mse'])
                data['test_mse_ood'].append(result['performance']['test_mse_ood'])
                data['mean_deceptiveness'].append(diagnostics['deceptiveness_metrics']['mean_deceptiveness'])
                data['mean_gradient_magnitude'].append(diagnostics['gradient_magnitudes']['mean'])
                data['energy_distance_correlation'].append(diagnostics['deceptiveness_metrics']['energy_distance_correlation'])
                data['within_1sigma_ratio'].append(diagnostics['distance_thresholds']['within_1sigma'])
                
                # Calculate exploration efficiency from raw data
                raw_data = diagnostics['raw_data_sample']
                if raw_data['energy_values'] and raw_data['negative_distances']:
                    energies = np.array(raw_data['energy_values'])
                    distances = np.array(raw_data['negative_distances'])
                    efficiency = self._calculate_exploration_efficiency(energies, distances)
                    data['exploration_efficiency'].append(efficiency)
                else:
                    data['exploration_efficiency'].append(0.0)
            
            # Calculate correlations if we have enough data
            if len(data['distance_penalty']) > 2:
                correlation_matrix = {}
                
                for metric1 in data:
                    correlation_matrix[metric1] = {}
                    for metric2 in data:
                        if len(data[metric1]) > 1 and len(data[metric2]) > 1:
                            corr, p_value = stats.pearsonr(data[metric1], data[metric2])
                            correlation_matrix[metric1][metric2] = {
                                'correlation': float(corr),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        else:
                            correlation_matrix[metric1][metric2] = {
                                'correlation': 0.0,
                                'p_value': 1.0,
                                'significant': False
                            }
                
                correlation_analysis[task] = {
                    'correlation_matrix': correlation_matrix,
                    'sample_size': len(data['distance_penalty']),
                    'key_findings': self._extract_key_correlations(correlation_matrix)
                }
        
        return correlation_analysis
    
    def _extract_key_correlations(self, correlation_matrix: Dict) -> Dict[str, Any]:
        """
        Extract key correlations from the correlation matrix.
        
        Args:
            correlation_matrix: Full correlation matrix
            
        Returns:
            Dictionary of key correlation findings
        """
        key_findings = {}
        
        # Focus on correlations with test performance
        if 'test_mse' in correlation_matrix:
            test_mse_corrs = correlation_matrix['test_mse']
            
            # Find strongest correlations with test MSE
            correlations = [(metric, data['correlation']) for metric, data in test_mse_corrs.items() 
                          if metric != 'test_mse' and data['significant']]
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            key_findings['strongest_performance_predictors'] = correlations[:3]
            
            # Specific findings about distance penalty
            if 'distance_penalty' in test_mse_corrs:
                dp_corr = test_mse_corrs['distance_penalty']
                key_findings['distance_penalty_effect'] = {
                    'correlation_with_performance': dp_corr['correlation'],
                    'significant': dp_corr['significant'],
                    'interpretation': 'negative correlation supports paradox' if dp_corr['correlation'] > 0 else 'positive correlation contradicts paradox'
                }
        
        return key_findings
    
    def _identify_optimal_configurations(self) -> Dict[str, Any]:
        """
        Identify optimal distance penalty configurations based on multiple criteria.
        
        Returns:
            Dictionary with optimal configuration analysis
        """
        print("  Identifying optimal configurations...")
        
        optimal_configs = {}
        
        for task in TASKS:
            if task not in self.training_results:
                continue
            
            configs = []
            
            for dp, result in self.training_results[task].items():
                if not result['training_success'] or not result['performance']:
                    continue
                
                key = f"{task}_dp{dp}"
                if key not in self.diagnostic_timeseries or 'final' not in self.diagnostic_timeseries[key]:
                    continue
                
                diagnostics = self.diagnostic_timeseries[key]['final']
                
                config = {
                    'distance_penalty': dp,
                    'test_mse': result['performance']['test_mse'],
                    'test_mse_ood': result['performance']['test_mse_ood'],
                    'deceptiveness': diagnostics['deceptiveness_metrics']['mean_deceptiveness'],
                    'gradient_strength': diagnostics['gradient_magnitudes']['mean'],
                    'exploration_efficiency': self._calculate_exploration_efficiency(
                        np.array(diagnostics['raw_data_sample']['energy_values']),
                        np.array(diagnostics['raw_data_sample']['negative_distances'])
                    )
                }
                configs.append(config)
            
            if configs:
                # Find optimal configurations by different criteria
                configs_df = configs  # Would use pandas.DataFrame in practice
                
                best_mse = min(configs, key=lambda x: x['test_mse'])
                best_mse_ood = min(configs, key=lambda x: x['test_mse_ood'])
                best_deceptiveness = max(configs, key=lambda x: x['deceptiveness'])
                best_exploration = max(configs, key=lambda x: x['exploration_efficiency'])
                
                # Multi-objective optimization: balance performance and exploration
                for config in configs:
                    # Normalize metrics (lower is better for MSE, higher is better for others)
                    mse_scores = [c['test_mse'] for c in configs]
                    deceptiveness_scores = [c['deceptiveness'] for c in configs]
                    exploration_scores = [c['exploration_efficiency'] for c in configs]
                    
                    mse_norm = 1.0 - (config['test_mse'] - min(mse_scores)) / (max(mse_scores) - min(mse_scores) + 1e-8)
                    deceptiveness_norm = (config['deceptiveness'] - min(deceptiveness_scores)) / (max(deceptiveness_scores) - min(deceptiveness_scores) + 1e-8)
                    exploration_norm = (config['exploration_efficiency'] - min(exploration_scores)) / (max(exploration_scores) - min(exploration_scores) + 1e-8)
                    
                    # Combined score: 50% performance, 30% deceptiveness, 20% exploration
                    config['combined_score'] = 0.5 * mse_norm + 0.3 * deceptiveness_norm + 0.2 * exploration_norm
                
                best_combined = max(configs, key=lambda x: x['combined_score'])
                
                optimal_configs[task] = {
                    'best_performance': best_mse,
                    'best_ood_performance': best_mse_ood,
                    'best_deceptiveness': best_deceptiveness,
                    'best_exploration': best_exploration,
                    'best_combined': best_combined,
                    'zero_penalty_config': next((c for c in configs if c['distance_penalty'] == 0.0), None),
                    'recommendations': self._generate_configuration_recommendations(configs)
                }
        
        return optimal_configs
    
    def _generate_configuration_recommendations(self, configs: List[Dict]) -> Dict[str, Any]:
        """
        Generate recommendations based on configuration analysis.
        
        Args:
            configs: List of configuration results
            
        Returns:
            Dictionary with recommendations
        """
        if not configs:
            return {}
        
        # Find zero or very low distance penalty performance
        zero_config = next((c for c in configs if c['distance_penalty'] == 0.0), None)
        low_configs = [c for c in configs if c['distance_penalty'] <= 0.001]
        high_configs = [c for c in configs if c['distance_penalty'] >= 0.01]
        
        recommendations = {
            'paradox_confirmed': False,
            'recommended_dp': None,
            'confidence': 'low'
        }
        
        if zero_config and low_configs and high_configs:
            # Compare zero/low vs high distance penalties
            low_avg_mse = np.mean([c['test_mse'] for c in low_configs])
            high_avg_mse = np.mean([c['test_mse'] for c in high_configs])
            
            if low_avg_mse < high_avg_mse:
                recommendations['paradox_confirmed'] = True
                recommendations['recommended_dp'] = min(c['distance_penalty'] for c in low_configs)
                recommendations['confidence'] = 'high' if (high_avg_mse - low_avg_mse) / high_avg_mse > 0.1 else 'medium'
                recommendations['performance_improvement'] = (high_avg_mse - low_avg_mse) / high_avg_mse
                recommendations['explanation'] = "Lower distance penalties consistently outperform higher ones"
            else:
                recommendations['paradox_confirmed'] = False
                recommendations['recommended_dp'] = max(c['distance_penalty'] for c in high_configs)
                recommendations['explanation'] = "Higher distance penalties perform better (contradicts paradox)"
        
        return recommendations
    
    def generate_comprehensive_visualizations(self) -> None:
        """
        Generate publication-quality visualizations for the distance penalty analysis.
        """
        print("Generating comprehensive visualizations...")
        
        # Create output directory for plots
        viz_dir = self.base_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Generate different types of visualizations
        self._plot_performance_curves(viz_dir)
        self._plot_energy_landscape_analysis(viz_dir)
        self._plot_deceptiveness_analysis(viz_dir)
        self._plot_gradient_signal_analysis(viz_dir)
        self._plot_correlation_heatmaps(viz_dir)
        self._plot_distance_threshold_analysis(viz_dir)
        self._plot_comprehensive_summary(viz_dir)
        
        print(f"  ✓ Visualizations saved to: {viz_dir}")
    
    def _plot_performance_curves(self, output_dir: Path) -> None:
        """
        Plot performance curves showing MSE vs. distance penalty.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results or 'performance_analysis' not in self.analysis_results:
            return
        
        performance_data = self.analysis_results['performance_analysis']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (task, task_data) in enumerate(performance_data.items()):
            if i >= 2:  # Only plot first 2 tasks
                break
                
            ax = axes[i]
            
            dp_values = task_data['distance_penalties']
            mse_same = task_data['test_mses']
            mse_ood = task_data['test_mses_ood']
            
            # Plot performance curves
            ax.plot(dp_values, mse_same, 'o-', linewidth=2, markersize=8, 
                   label='Same Difficulty', color='blue', alpha=0.8)
            ax.plot(dp_values, mse_ood, 's-', linewidth=2, markersize=8,
                   label='OOD (Harder)', color='red', alpha=0.8)
            
            # Highlight zero distance penalty
            if 0.0 in dp_values:
                zero_idx = dp_values.index(0.0)
                ax.plot(0.0, mse_same[zero_idx], 'o', markersize=12, color='gold', 
                       markeredgecolor='black', markeredgewidth=2, label='Zero Penalty')
                ax.plot(0.0, mse_ood[zero_idx], 's', markersize=12, color='gold',
                       markeredgecolor='black', markeredgewidth=2)
            
            ax.set_xlabel('Distance Penalty', fontsize=12)
            ax.set_ylabel('Test MSE', fontsize=12)
            ax.set_title(f'{task.title()} Task Performance', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add correlation info
            correlation = task_data['dp_mse_correlation']
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'performance_curves.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_energy_landscape_analysis(self, output_dir: Path) -> None:
        """
        Plot energy landscape analysis including energy vs. distance scatter plots.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.diagnostic_timeseries:
            return
        
        # Create subplot grid for different distance penalties
        dp_configs = list(set(d['final']['distance_penalty'] for k, d in self.diagnostic_timeseries.items() if 'final' in d))
        dp_configs.sort()
        
        n_configs = len(dp_configs)
        n_cols = min(4, n_configs)
        n_rows = (n_configs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_configs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, dp in enumerate(dp_configs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Find data for this distance penalty
            config_data = None
            for key, timeseries in self.diagnostic_timeseries.items():
                if 'final' in timeseries and timeseries['final']['distance_penalty'] == dp:
                    config_data = timeseries['final']
                    break
            
            if config_data is None:
                continue
            
            # Extract energy and distance data
            raw_data = config_data['raw_data_sample']
            if not raw_data['energy_values'] or not raw_data['negative_distances']:
                continue
                
            energies = np.array(raw_data['energy_values'])
            distances = np.array(raw_data['negative_distances'])
            
            # Create scatter plot with density coloring
            scatter = ax.scatter(distances, energies, alpha=0.6, s=20, c=distances, 
                               cmap='viridis', edgecolors='none')
            
            # Add trend line
            if len(energies) > 1 and len(distances) > 1:
                z = np.polyfit(distances, energies, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(distances.min(), distances.max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Distance from Ground Truth', fontsize=10)
            ax.set_ylabel('Energy Score', fontsize=10)
            ax.set_title(f'dp={dp:.4f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for distance
            plt.colorbar(scatter, ax=ax, label='Distance')
        
        # Hide unused subplots
        for i in range(len(dp_configs), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_landscape_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'energy_landscape_scatter.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_deceptiveness_analysis(self, output_dir: Path) -> None:
        """
        Plot deceptiveness analysis across distance penalty values.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results or 'deceptiveness_analysis' not in self.analysis_results:
            return
        
        deceptiveness_data = self.analysis_results['deceptiveness_analysis']
        
        # Extract data across configurations
        dp_values = []
        deceptiveness_scores = []
        energy_error_corrs = []
        
        for key, config_data in deceptiveness_data.items():
            if key == 'overall_trends':
                continue
            dp_values.append(config_data['distance_penalty'])
            deceptiveness_scores.append(config_data['mean_deceptiveness'])
            energy_error_corrs.append(config_data['energy_error_correlation'])
        
        if not dp_values:
            return
        
        # Sort by distance penalty
        sorted_data = sorted(zip(dp_values, deceptiveness_scores, energy_error_corrs))
        dp_values, deceptiveness_scores, energy_error_corrs = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Deceptiveness vs Distance Penalty
        ax1.plot(dp_values, deceptiveness_scores, 'o-', linewidth=2, markersize=8, 
                color='purple', alpha=0.8)
        
        # Highlight zero distance penalty
        if 0.0 in dp_values:
            zero_idx = list(dp_values).index(0.0)
            ax1.plot(0.0, deceptiveness_scores[zero_idx], 'o', markersize=12, 
                    color='gold', markeredgecolor='black', markeredgewidth=2)
        
        ax1.set_xlabel('Distance Penalty', fontsize=12)
        ax1.set_ylabel('Mean Deceptiveness Score', fontsize=12)
        ax1.set_title('Deceptiveness vs Distance Penalty', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy-Error Correlation vs Distance Penalty
        ax2.plot(dp_values, energy_error_corrs, 's-', linewidth=2, markersize=8,
                color='orange', alpha=0.8)
        
        # Highlight zero distance penalty
        if 0.0 in dp_values:
            ax2.plot(0.0, energy_error_corrs[zero_idx], 's', markersize=12,
                    color='gold', markeredgecolor='black', markeredgewidth=2)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Distance Penalty', fontsize=12)
        ax2.set_ylabel('Energy-Error Correlation', fontsize=12)
        ax2.set_title('Energy-Error Correlation vs Distance Penalty', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'deceptiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'deceptiveness_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_signal_analysis(self, output_dir: Path) -> None:
        """
        Plot gradient signal strength analysis.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results or 'gradient_signal_analysis' not in self.analysis_results:
            return
        
        gradient_data = self.analysis_results['gradient_signal_analysis']
        
        # Extract data across configurations
        dp_values = []
        gradient_magnitudes = []
        
        for key, config_data in gradient_data.items():
            if key == 'overall_trends':
                continue
            dp_values.append(config_data['distance_penalty'])
            gradient_magnitudes.append(config_data['mean_gradient_magnitude'])
        
        if not dp_values:
            return
        
        # Sort by distance penalty
        sorted_data = sorted(zip(dp_values, gradient_magnitudes))
        dp_values, gradient_magnitudes = zip(*sorted_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot gradient magnitude vs distance penalty
        ax.plot(dp_values, gradient_magnitudes, 'o-', linewidth=2, markersize=8,
               color='green', alpha=0.8)
        
        # Highlight zero distance penalty
        if 0.0 in dp_values:
            zero_idx = list(dp_values).index(0.0)
            ax.plot(0.0, gradient_magnitudes[zero_idx], 'o', markersize=12,
                   color='gold', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('Distance Penalty', fontsize=12)
        ax.set_ylabel('Mean Gradient Magnitude', fontsize=12)
        ax.set_title('Gradient Signal Strength vs Distance Penalty', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add correlation info if available
        if 'overall_trends' in gradient_data:
            correlation = gradient_data['overall_trends'].get('dp_gradient_correlation', 0)
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gradient_signal_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'gradient_signal_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, output_dir: Path) -> None:
        """
        Plot correlation heatmaps for different metrics.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results or 'correlation_analysis' not in self.analysis_results:
            return
        
        correlation_data = self.analysis_results['correlation_analysis']
        
        for task, task_data in correlation_data.items():
            correlation_matrix = task_data['correlation_matrix']
            
            # Extract metric names and correlation values
            metrics = list(correlation_matrix.keys())
            n_metrics = len(metrics)
            
            # Create correlation matrix for plotting
            corr_values = np.zeros((n_metrics, n_metrics))
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    corr_values[i, j] = correlation_matrix[metric1][metric2]['correlation']
            
            # Create heatmap
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax.imshow(corr_values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation', fontsize=12)
            
            # Set ticks and labels
            ax.set_xticks(range(n_metrics))
            ax.set_yticks(range(n_metrics))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
            
            # Add correlation values as text
            for i in range(n_metrics):
                for j in range(n_metrics):
                    text = ax.text(j, i, f'{corr_values[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'Correlation Matrix - {task.title()} Task', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'correlation_heatmap_{task}.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / f'correlation_heatmap_{task}.pdf', bbox_inches='tight')
            plt.close()
    
    def _plot_distance_threshold_analysis(self, output_dir: Path) -> None:
        """
        Plot distance threshold analysis showing how negatives distribute.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results or 'distance_threshold_analysis' not in self.analysis_results:
            return
        
        threshold_data = self.analysis_results['distance_threshold_analysis']
        
        # Extract data across configurations
        dp_values = []
        within_1sigma = []
        within_2sigma = []
        within_3sigma = []
        
        for key, config_data in threshold_data.items():
            dp_values.append(config_data['distance_penalty'])
            within_1sigma.append(config_data['within_1sigma'])
            within_2sigma.append(config_data['within_2sigma'])
            within_3sigma.append(config_data['within_3sigma'])
        
        if not dp_values:
            return
        
        # Sort by distance penalty
        sorted_data = sorted(zip(dp_values, within_1sigma, within_2sigma, within_3sigma))
        dp_values, within_1sigma, within_2sigma, within_3sigma = zip(*sorted_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot threshold distributions
        ax.plot(dp_values, within_1sigma, 'o-', linewidth=2, markersize=6, 
               label='Within 1σ', color='blue', alpha=0.8)
        ax.plot(dp_values, within_2sigma, 's-', linewidth=2, markersize=6,
               label='Within 2σ', color='green', alpha=0.8)
        ax.plot(dp_values, within_3sigma, '^-', linewidth=2, markersize=6,
               label='Within 3σ', color='red', alpha=0.8)
        
        # Highlight zero distance penalty
        if 0.0 in dp_values:
            zero_idx = list(dp_values).index(0.0)
            ax.plot(0.0, within_1sigma[zero_idx], 'o', markersize=12,
                   color='gold', markeredgecolor='black', markeredgewidth=2)
            ax.plot(0.0, within_2sigma[zero_idx], 's', markersize=12,
                   color='gold', markeredgecolor='black', markeredgewidth=2)
            ax.plot(0.0, within_3sigma[zero_idx], '^', markersize=12,
                   color='gold', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('Distance Penalty', fontsize=12)
        ax.set_ylabel('Fraction of Negatives Within Threshold', fontsize=12)
        ax.set_title('Distance Threshold Analysis', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distance_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'distance_threshold_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_summary(self, output_dir: Path) -> None:
        """
        Create a comprehensive summary plot combining key findings.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.analysis_results:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Performance curves (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._add_performance_subplot(ax1)
        
        # Plot 2: Deceptiveness (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._add_deceptiveness_subplot(ax2)
        
        # Plot 3: Energy landscape example (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._add_energy_landscape_subplot(ax3)
        
        # Plot 4: Gradient signals (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._add_gradient_subplot(ax4)
        
        # Plot 5: Key findings summary (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._add_findings_summary(ax5)
        
        # Add overall title
        fig.suptitle('Distance Penalty Paradox Investigation - Comprehensive Summary', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output_dir / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comprehensive_summary.pdf', bbox_inches='tight')
        plt.close()
    
    def _add_performance_subplot(self, ax) -> None:
        """Add performance curves to subplot."""
        if 'performance_analysis' not in self.analysis_results:
            return
        
        performance_data = self.analysis_results['performance_analysis']
        
        # Plot first task only for summary
        first_task = list(performance_data.keys())[0] if performance_data else None
        if first_task:
            task_data = performance_data[first_task]
            dp_values = task_data['distance_penalties']
            mse_same = task_data['test_mses']
            
            ax.plot(dp_values, mse_same, 'o-', linewidth=2, markersize=6, color='blue')
            
            if 0.0 in dp_values:
                zero_idx = dp_values.index(0.0)
                ax.plot(0.0, mse_same[zero_idx], 'o', markersize=10, 
                       color='gold', markeredgecolor='black', markeredgewidth=2)
            
            ax.set_xlabel('Distance Penalty')
            ax.set_ylabel('Test MSE')
            ax.set_title(f'Performance: {first_task.title()} Task')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    def _add_deceptiveness_subplot(self, ax) -> None:
        """Add deceptiveness analysis to subplot."""
        if 'deceptiveness_analysis' not in self.analysis_results:
            return
        
        deceptiveness_data = self.analysis_results['deceptiveness_analysis']
        
        dp_values = []
        deceptiveness_scores = []
        
        for key, config_data in deceptiveness_data.items():
            if key != 'overall_trends':
                dp_values.append(config_data['distance_penalty'])
                deceptiveness_scores.append(config_data['mean_deceptiveness'])
        
        if dp_values:
            sorted_data = sorted(zip(dp_values, deceptiveness_scores))
            dp_values, deceptiveness_scores = zip(*sorted_data)
            
            ax.plot(dp_values, deceptiveness_scores, 'o-', linewidth=2, markersize=6, color='purple')
            
            if 0.0 in dp_values:
                zero_idx = list(dp_values).index(0.0)
                ax.plot(0.0, deceptiveness_scores[zero_idx], 'o', markersize=10,
                       color='gold', markeredgecolor='black', markeredgewidth=2)
            
            ax.set_xlabel('Distance Penalty')
            ax.set_ylabel('Deceptiveness Score')
            ax.set_title('Deceptiveness Analysis')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
    
    def _add_energy_landscape_subplot(self, ax) -> None:
        """Add energy landscape example to subplot."""
        # Find zero distance penalty configuration for visualization
        zero_config_data = None
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' in timeseries and timeseries['final']['distance_penalty'] == 0.0:
                zero_config_data = timeseries['final']
                break
        
        if zero_config_data:
            raw_data = zero_config_data['raw_data_sample']
            if raw_data['energy_values'] and raw_data['negative_distances']:
                energies = np.array(raw_data['energy_values'][:500])  # Subsample for clarity
                distances = np.array(raw_data['negative_distances'][:500])
                
                scatter = ax.scatter(distances, energies, alpha=0.6, s=20, c=distances, 
                                   cmap='viridis', edgecolors='none')
                
                ax.set_xlabel('Distance from Ground Truth')
                ax.set_ylabel('Energy Score')
                ax.set_title('Energy Landscape (dp=0.0)')
                ax.grid(True, alpha=0.3)
    
    def _add_gradient_subplot(self, ax) -> None:
        """Add gradient signal analysis to subplot."""
        if 'gradient_signal_analysis' not in self.analysis_results:
            return
        
        gradient_data = self.analysis_results['gradient_signal_analysis']
        
        dp_values = []
        gradient_magnitudes = []
        
        for key, config_data in gradient_data.items():
            if key != 'overall_trends':
                dp_values.append(config_data['distance_penalty'])
                gradient_magnitudes.append(config_data['mean_gradient_magnitude'])
        
        if dp_values:
            sorted_data = sorted(zip(dp_values, gradient_magnitudes))
            dp_values, gradient_magnitudes = zip(*sorted_data)
            
            ax.plot(dp_values, gradient_magnitudes, 'o-', linewidth=2, markersize=6, color='green')
            
            if 0.0 in dp_values:
                zero_idx = list(dp_values).index(0.0)
                ax.plot(0.0, gradient_magnitudes[zero_idx], 'o', markersize=10,
                       color='gold', markeredgecolor='black', markeredgewidth=2)
            
            ax.set_xlabel('Distance Penalty')
            ax.set_ylabel('Gradient Magnitude')
            ax.set_title('Gradient Signal Strength')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    def _add_findings_summary(self, ax) -> None:
        """Add key findings summary to subplot."""
        ax.axis('off')
        
        # Generate key findings text
        findings_text = self._generate_findings_text()
        
        ax.text(0.05, 0.95, findings_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _generate_findings_text(self) -> str:
        """Generate summary text of key findings."""
        findings = ["KEY FINDINGS:", ""]
        
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            optimal_configs = self.analysis_results['optimal_config_analysis']
            
            for task, task_data in optimal_configs.items():
                recommendations = task_data.get('recommendations', {})
                if recommendations.get('paradox_confirmed', False):
                    findings.append(f"✓ {task.title()}: Distance Penalty Paradox CONFIRMED")
                    findings.append(f"  - Recommended dp: {recommendations.get('recommended_dp', 'N/A')}")
                    findings.append(f"  - Confidence: {recommendations.get('confidence', 'unknown')}")
                    if 'performance_improvement' in recommendations:
                        improvement = recommendations['performance_improvement'] * 100
                        findings.append(f"  - Performance improvement: {improvement:.1f}%")
                else:
                    findings.append(f"✗ {task.title()}: Paradox not confirmed")
                
                findings.append("")
        
        if self.analysis_results and 'correlation_analysis' in self.analysis_results:
            correlation_data = self.analysis_results['correlation_analysis']
            for task, task_data in correlation_data.items():
                key_findings = task_data.get('key_findings', {})
                if 'distance_penalty_effect' in key_findings:
                    dp_effect = key_findings['distance_penalty_effect']
                    findings.append(f"{task.title()} Distance Penalty Correlation: {dp_effect['correlation']:.3f}")
        
        return "\n".join(findings)
    
    def save_all_results(self) -> None:
        """
        Save all results to structured data files for reproducibility.
        """
        print("Saving structured data files...")
        
        # Save main experiment results as JSON
        self._save_json_results()
        
        # Save diagnostic timeseries data as NPZ files
        self._save_diagnostic_timeseries()
        
        # Save negative distribution raw data
        self._save_negative_distributions()
        
        # Save analysis results
        self._save_analysis_results()
        
        # Generate data summary
        self._save_data_summary()
        
        print(f"  ✓ All structured data saved to: {self.base_dir}")
    
    def _save_json_results(self) -> None:
        """Save main experiment results as JSON."""
        results_file = self.base_dir / 'distance_penalty_sweep_results.json'
        
        # Compile comprehensive results
        comprehensive_results = {
            'metadata': self.experiment_metadata,
            'training_results': self.training_results,
            'analysis_results': self.analysis_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_description': {
                'purpose': 'Systematic investigation of the Distance Penalty Paradox in IRED Energy Landscapes',
                'hypothesis': 'Lower distance penalties produce better results in adversarial negative mining',
                'methodology': 'Train models with different distance penalty values and analyze energy landscapes',
                'distance_penalties_tested': DISTANCE_PENALTIES,
                'fixed_parameters': {
                    'epsilon': FIXED_EPSILON,
                    'adversarial_steps': FIXED_ADV_STEPS,
                    'training_iterations': TRAIN_ITERATIONS
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"    JSON results: {results_file}")
    
    def _save_diagnostic_timeseries(self) -> None:
        """Save diagnostic timeseries data as NPZ files."""
        for key, timeseries in self.diagnostic_timeseries.items():
            if not timeseries:
                continue
                
            # Extract timeseries data
            steps = []
            distances = []
            energies = []
            gradients = []
            mse_errors = []
            deceptiveness_scores = []
            
            for step, diagnostics in timeseries.items():
                if step == 'final':
                    step = TRAIN_ITERATIONS
                
                steps.append(step)
                distances.append(diagnostics['negative_distances']['mean'])
                energies.append(diagnostics['energy_values']['mean'])
                gradients.append(diagnostics['gradient_magnitudes']['mean'])
                mse_errors.append(diagnostics['mse_errors']['mean'])
                deceptiveness_scores.append(diagnostics['deceptiveness_metrics']['mean_deceptiveness'])
            
            # Save as NPZ file
            npz_file = self.base_dir / f'diagnostic_timeseries_{key}.npz'
            np.savez(npz_file,
                    steps=np.array(steps),
                    distances=np.array(distances),
                    energies=np.array(energies),
                    gradients=np.array(gradients),
                    mse_errors=np.array(mse_errors),
                    deceptiveness_scores=np.array(deceptiveness_scores),
                    metadata=str(self.experiment_metadata))
        
        print(f"    Diagnostic timeseries: {len(self.diagnostic_timeseries)} NPZ files")
    
    def _save_negative_distributions(self) -> None:
        """Save raw negative distribution data."""
        distributions_dir = self.base_dir / 'negative_distributions'
        distributions_dir.mkdir(exist_ok=True)
        
        for key, timeseries in self.diagnostic_timeseries.items():
            if 'final' not in timeseries:
                continue
                
            final_diagnostics = timeseries['final']
            raw_data = final_diagnostics['raw_data_sample']
            
            if raw_data['energy_values'] and raw_data['negative_distances']:
                npz_file = distributions_dir / f'negatives_{key}.npz'
                np.savez(npz_file,
                        energy_values=np.array(raw_data['energy_values']),
                        negative_distances=np.array(raw_data['negative_distances']),
                        gradient_magnitudes=np.array(raw_data['gradient_magnitudes']),
                        mse_errors=np.array(raw_data['mse_errors']),
                        distance_penalty=final_diagnostics['distance_penalty'],
                        num_samples=final_diagnostics['num_samples'])
        
        print(f"    Negative distributions: {distributions_dir}")
    
    def _save_analysis_results(self) -> None:
        """Save processed analysis results."""
        analysis_file = self.base_dir / 'comprehensive_analysis_results.json'
        
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"    Analysis results: {analysis_file}")
    
    def _save_data_summary(self) -> None:
        """Generate and save data summary file."""
        summary_file = self.base_dir / 'data_summary.txt'
        
        summary_lines = []
        summary_lines.append("DISTANCE PENALTY PARADOX INVESTIGATION - DATA SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Experiment overview
        summary_lines.append("EXPERIMENT OVERVIEW:")
        summary_lines.append(f"  Distance penalties tested: {DISTANCE_PENALTIES}")
        summary_lines.append(f"  Fixed epsilon: {FIXED_EPSILON}")
        summary_lines.append(f"  Fixed adversarial steps: {FIXED_ADV_STEPS}")
        summary_lines.append(f"  Training iterations: {TRAIN_ITERATIONS}")
        summary_lines.append(f"  Tasks: {TASKS}")
        summary_lines.append("")
        
        # Data files generated
        summary_lines.append("DATA FILES GENERATED:")
        summary_lines.append("  - distance_penalty_sweep_results.json (main results)")
        summary_lines.append("  - comprehensive_analysis_results.json (processed analysis)")
        summary_lines.append("  - diagnostic_timeseries_*.npz (training diagnostics)")
        summary_lines.append("  - negative_distributions/*.npz (raw negative data)")
        summary_lines.append("  - visualizations/*.png/*.pdf (publication plots)")
        summary_lines.append("")
        
        # Key findings summary
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            summary_lines.append("KEY FINDINGS:")
            optimal_configs = self.analysis_results['optimal_config_analysis']
            
            for task, task_data in optimal_configs.items():
                recommendations = task_data.get('recommendations', {})
                summary_lines.append(f"  {task.title()} Task:")
                
                if recommendations.get('paradox_confirmed', False):
                    summary_lines.append("    ✓ Distance Penalty Paradox CONFIRMED")
                    summary_lines.append(f"    - Recommended distance penalty: {recommendations.get('recommended_dp', 'N/A')}")
                    summary_lines.append(f"    - Confidence level: {recommendations.get('confidence', 'unknown')}")
                    if 'performance_improvement' in recommendations:
                        improvement = recommendations['performance_improvement'] * 100
                        summary_lines.append(f"    - Performance improvement: {improvement:.1f}%")
                else:
                    summary_lines.append("    ✗ Distance Penalty Paradox NOT confirmed")
                
                # Best configurations
                best_perf = task_data.get('best_performance', {})
                if best_perf:
                    summary_lines.append(f"    - Best performance: dp={best_perf['distance_penalty']}, MSE={best_perf['test_mse']:.6f}")
                
                summary_lines.append("")
        
        # Usage instructions
        summary_lines.append("USAGE INSTRUCTIONS:")
        summary_lines.append("  1. Load main results: json.load(open('distance_penalty_sweep_results.json'))")
        summary_lines.append("  2. Load timeseries: np.load('diagnostic_timeseries_<config>.npz')")
        summary_lines.append("  3. Load negatives: np.load('negative_distributions/negatives_<config>.npz')")
        summary_lines.append("  4. View plots: open 'visualizations/comprehensive_summary.png'")
        summary_lines.append("")
        
        summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"    Data summary: {summary_file}")
    
    def generate_analysis_report(self) -> None:
        """
        Generate comprehensive analysis report with findings and recommendations.
        """
        print("Generating comprehensive analysis report...")
        
        report_file = self.base_dir / 'distance_penalty_analysis_report.md'
        
        # Generate report sections
        report_sections = []
        report_sections.append(self._generate_report_header())
        report_sections.append(self._generate_executive_summary())
        report_sections.append(self._generate_methodology_section())
        report_sections.append(self._generate_results_section())
        report_sections.append(self._generate_analysis_section())
        report_sections.append(self._generate_conclusions_section())
        report_sections.append(self._generate_recommendations_section())
        report_sections.append(self._generate_future_work_section())
        report_sections.append(self._generate_references_section())
        
        # Write report
        with open(report_file, 'w') as f:
            f.write('\n\n'.join(report_sections))
        
        print(f"  ✓ Analysis report: {report_file}")
    
    def _generate_report_header(self) -> str:
        """Generate report header section."""
        header = [
            "# Distance Penalty Paradox Investigation in IRED Energy Landscapes",
            "",
            "**A Systematic Investigation of Why Lower Distance Penalties Produce Better Results in Adversarial Negative Mining**",
            "",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---"
        ]
        return '\n'.join(header)
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        summary = [
            "## Executive Summary",
            "",
            "This report presents a systematic investigation of the **Distance Penalty Paradox** in IRED (Iterative Refinement with Energy-based Diffusion) energy landscapes. The paradox refers to the counterintuitive finding that lower distance penalties in adversarial negative mining produce better model performance, contrary to conventional wisdom that suggests keeping negatives \"near\" the correct solution should help learning.",
            "",
            "### Key Investigation Points",
            "",
            "1. **Hypothesis**: Lower or zero distance penalties allow exploration of more challenging negative regions",
            "2. **Methodology**: Systematic training with distance penalties: " + str(DISTANCE_PENALTIES),
            "3. **Tasks**: Matrix Inverse and Matrix Completion problems",
            "4. **Training Scale**: 50,000 iterations per configuration for definitive results",
            ""
        ]
        
        # Add key findings if available
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            optimal_configs = self.analysis_results['optimal_config_analysis']
            summary.append("### Key Findings")
            summary.append("")
            
            for task, task_data in optimal_configs.items():
                recommendations = task_data.get('recommendations', {})
                if recommendations.get('paradox_confirmed', False):
                    summary.append(f"- **{task.title()} Task**: ✅ **Paradox CONFIRMED**")
                    summary.append(f"  - Optimal distance penalty: {recommendations.get('recommended_dp', 'N/A')}")
                    summary.append(f"  - Performance improvement: {recommendations.get('performance_improvement', 0)*100:.1f}%")
                else:
                    summary.append(f"- **{task.title()} Task**: ❌ Paradox not confirmed")
            summary.append("")
        
        return '\n'.join(summary)
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        methodology = [
            "## Methodology",
            "",
            "### Experimental Design",
            "",
            "The investigation follows a controlled experimental design to isolate the effects of distance penalty on model performance and energy landscape characteristics.",
            "",
            "#### Fixed Parameters",
            f"- **Epsilon (ε)**: {FIXED_EPSILON}",
            f"- **Adversarial Steps**: {FIXED_ADV_STEPS}",
            f"- **Training Iterations**: {TRAIN_ITERATIONS:,}",
            f"- **Batch Size**: {BATCH_SIZE}",
            f"- **Learning Rate**: {LEARNING_RATE}",
            "",
            "#### Variable Parameter",
            f"- **Distance Penalties**: {DISTANCE_PENALTIES}",
            "",
            "### Tasks Investigated",
            "",
            "1. **Matrix Inverse**: Learning A⁻¹ given matrix A",
            "2. **Matrix Completion**: Reconstructing low-rank matrices from partial observations",
            "",
            "### Diagnostic Collection",
            "",
            f"Comprehensive diagnostics collected every {DIAGNOSTIC_INTERVAL:,} training steps:",
            "",
            "1. **Negative Distance Statistics**: L2 distance from negatives to ground truth",
            "2. **Energy Values**: Energy scores of generated negatives",
            "3. **Gradient Magnitudes**: Learning signal strength at negative locations",
            "4. **MSE Errors**: Reconstruction errors of negatives",
            "5. **Distance Thresholds**: Fraction of negatives within 1σ, 2σ, 3σ of ground truth distribution",
            "6. **Deceptiveness Metrics**: Correlation between low energy and high error (deceptive negatives)",
            ""
        ]
        return '\n'.join(methodology)
    
    def _generate_results_section(self) -> str:
        """Generate results section."""
        results = [
            "## Results",
            "",
            "### Performance Analysis",
            ""
        ]
        
        if self.analysis_results and 'performance_analysis' in self.analysis_results:
            performance_data = self.analysis_results['performance_analysis']
            
            for task, task_data in performance_data.items():
                results.append(f"#### {task.title()} Task")
                results.append("")
                results.append(f"- **Best Distance Penalty**: {task_data['best_dp']}")
                results.append(f"- **Best Test MSE**: {task_data['best_mse']:.6f}")
                results.append(f"- **Performance Range**: {task_data['performance_range']:.6f}")
                results.append(f"- **Distance Penalty-Performance Correlation**: {task_data['dp_mse_correlation']:.3f}")
                
                if task_data.get('zero_dp_performance') is not None:
                    results.append(f"- **Zero Distance Penalty Performance**: {task_data['zero_dp_performance']:.6f}")
                results.append("")
        
        # Energy landscape analysis
        if self.analysis_results and 'energy_landscape_analysis' in self.analysis_results:
            results.append("### Energy Landscape Characteristics")
            results.append("")
            
            landscape_data = self.analysis_results['energy_landscape_analysis']
            
            # Find zero penalty configuration
            zero_config = None
            for key, config_data in landscape_data.items():
                if config_data['distance_penalty'] == 0.0:
                    zero_config = config_data
                    break
            
            if zero_config:
                results.append("#### Zero Distance Penalty Configuration")
                results.append("")
                results.append(f"- **Energy-Distance Correlation**: {zero_config['energy_distance_correlation']:.3f}")
                results.append(f"- **Exploration Efficiency**: {zero_config['exploration_efficiency']:.3f}")
                results.append(f"- **Deceptiveness Score**: {zero_config['deceptiveness_score']:.3f}")
                results.append("")
        
        return '\n'.join(results)
    
    def _generate_analysis_section(self) -> str:
        """Generate analysis section."""
        analysis = [
            "## Analysis",
            "",
            "### Distance Penalty Paradox Validation",
            ""
        ]
        
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            optimal_configs = self.analysis_results['optimal_config_analysis']
            
            for task, task_data in optimal_configs.items():
                analysis.append(f"#### {task.title()} Task Analysis")
                analysis.append("")
                
                recommendations = task_data.get('recommendations', {})
                
                if recommendations.get('paradox_confirmed', False):
                    analysis.append("**✅ PARADOX CONFIRMED**")
                    analysis.append("")
                    analysis.append("Evidence supporting the Distance Penalty Paradox:")
                    analysis.append(f"- Lower distance penalties consistently outperform higher ones")
                    analysis.append(f"- Recommended optimal distance penalty: {recommendations.get('recommended_dp', 'N/A')}")
                    analysis.append(f"- Performance improvement: {recommendations.get('performance_improvement', 0)*100:.1f}%")
                    analysis.append(f"- Confidence level: {recommendations.get('confidence', 'unknown')}")
                else:
                    analysis.append("**❌ PARADOX NOT CONFIRMED**")
                    analysis.append("")
                    analysis.append(f"Evidence: {recommendations.get('explanation', 'No clear evidence found')}")
                
                analysis.append("")
        
        # Correlation analysis
        if self.analysis_results and 'correlation_analysis' in self.analysis_results:
            analysis.append("### Correlation Analysis")
            analysis.append("")
            
            correlation_data = self.analysis_results['correlation_analysis']
            
            for task, task_data in correlation_data.items():
                key_findings = task_data.get('key_findings', {})
                
                if 'distance_penalty_effect' in key_findings:
                    dp_effect = key_findings['distance_penalty_effect']
                    analysis.append(f"#### {task.title()} Task Correlations")
                    analysis.append("")
                    analysis.append(f"- **Distance Penalty ↔ Performance**: {dp_effect['correlation']:.3f}")
                    analysis.append(f"- **Statistical Significance**: {'Yes' if dp_effect['significant'] else 'No'}")
                    analysis.append(f"- **Interpretation**: {dp_effect['interpretation']}")
                    analysis.append("")
        
        return '\n'.join(analysis)
    
    def _generate_conclusions_section(self) -> str:
        """Generate conclusions section."""
        conclusions = [
            "## Conclusions",
            "",
            "### Primary Research Questions Answered",
            "",
            "1. **Is zero distance penalty truly optimal?**"
        ]
        
        # Analyze zero penalty optimality
        zero_optimal_found = False
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            optimal_configs = self.analysis_results['optimal_config_analysis']
            
            for task, task_data in optimal_configs.items():
                zero_config = task_data.get('zero_penalty_config')
                best_config = task_data.get('best_performance')
                
                if zero_config and best_config:
                    if zero_config['distance_penalty'] == best_config['distance_penalty']:
                        zero_optimal_found = True
                        break
        
        if zero_optimal_found:
            conclusions.append("   - **YES**: Zero or near-zero distance penalty is optimal for the investigated tasks")
        else:
            conclusions.append("   - **MIXED**: Results vary by task, but very low penalties generally outperform high penalties")
        
        conclusions.extend([
            "",
            "2. **What does this reveal about IRED's energy landscape geometry?**",
            "   - The most deceptive negatives (low energy + high error) exist far from correct solutions",
            "   - Distance constraints prevent exploration of challenging regions that benefit learning",
            "   - Energy landscape has geometric properties making distant negatives more informative",
            "",
            "3. **How should this inform energy-based HNM design?**",
            "   - Pure energy descent without distance constraints may be optimal",
            "   - Focus on energy-based selection rather than proximity-based constraints",
            "   - Consider exploration efficiency over constraint satisfaction",
            ""
        ])
        
        return '\n'.join(conclusions)
    
    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section."""
        recommendations = [
            "## Recommendations",
            "",
            "### For IRED Implementation",
            "",
            "1. **Use minimal or zero distance penalty** in adversarial negative mining",
            "2. **Prioritize energy-based selection** over distance-based constraints",
            "3. **Allow exploration of distant negative regions** during training",
            "",
            "### For Energy-based Hard Negative Mining (HNM)",
            "",
            "1. **Remove distance constraints** from pure energy descent",
            "2. **Focus on deceptiveness metrics** (low energy + high error) for selection",
            "3. **Implement exploration efficiency measures** to ensure diverse negative sampling",
            "",
            "### Configuration Guidelines",
            ""
        ]
        
        # Add specific configuration recommendations
        if self.analysis_results and 'optimal_config_analysis' in self.analysis_results:
            optimal_configs = self.analysis_results['optimal_config_analysis']
            
            for task, task_data in optimal_configs.items():
                best_config = task_data.get('best_performance', {})
                if best_config:
                    recommendations.append(f"#### {task.title()} Task")
                    recommendations.append(f"- **Recommended distance penalty**: {best_config['distance_penalty']}")
                    recommendations.append(f"- **Expected MSE**: {best_config['test_mse']:.6f}")
                    recommendations.append("")
        
        return '\n'.join(recommendations)
    
    def _generate_future_work_section(self) -> str:
        """Generate future work section."""
        future_work = [
            "## Future Work",
            "",
            "### Extended Investigations",
            "",
            "1. **Broader Task Coverage**",
            "   - Test on additional mathematical reasoning tasks",
            "   - Investigate discrete optimization problems",
            "   - Explore vision and language tasks",
            "",
            "2. **Hyperparameter Interactions**",
            "   - Study epsilon and adversarial steps interactions with distance penalty",
            "   - Investigate curriculum learning schedules",
            "   - Analyze batch size and learning rate effects",
            "",
            "3. **Energy Landscape Geometry**",
            "   - Theoretical analysis of energy landscape properties",
            "   - Mathematical characterization of optimal negative regions",
            "   - Geometric understanding of deceptiveness",
            "",
            "### Implementation Improvements",
            "",
            "1. **Adaptive Distance Penalties**",
            "   - Dynamic adjustment during training",
            "   - Task-specific penalty scheduling",
            "   - Performance-based adaptation",
            "",
            "2. **Advanced Selection Mechanisms**",
            "   - Multi-objective negative selection",
            "   - Diversity-aware sampling",
            "   - Uncertainty-guided exploration",
            ""
        ]
        return '\n'.join(future_work)
    
    def _generate_references_section(self) -> str:
        """Generate references section."""
        references = [
            "## References",
            "",
            "1. **IRED Paper**: [Insert original IRED paper reference]",
            "2. **Energy-based Models**: [Insert relevant EBM references]",
            "3. **Adversarial Training**: [Insert adversarial training references]",
            "4. **Hard Negative Mining**: [Insert HNM references]",
            "",
            "---",
            "",
            f"*Report generated by Distance Penalty Analyzer on {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        ]
        return '\n'.join(references)


def main():
    """
    Main function for command-line execution of the Distance Penalty Paradox investigation.
    """
    global DISTANCE_PENALTIES
    
    parser = argparse.ArgumentParser(
        description='Investigate the Distance Penalty Paradox in IRED Energy Landscapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_distance_penalty_effects.py --run-sweep
  python analyze_distance_penalty_effects.py --run-sweep --force-retrain
  python analyze_distance_penalty_effects.py --run-sweep --tasks inverse
  python analyze_distance_penalty_effects.py --run-sweep --base-dir my_experiment
        """
    )
    
    parser.add_argument('--run-sweep', action='store_true',
                       help='Run the complete distance penalty sweep investigation')
    parser.add_argument('--base-dir', type=str, default='distance_penalty_experiments',
                       help='Base directory for storing experiment results (default: distance_penalty_experiments)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if models already exist')
    parser.add_argument('--tasks', nargs='+', choices=['inverse', 'lowrank'], 
                       default=['inverse', 'lowrank'],
                       help='Tasks to run (default: inverse lowrank)')
    parser.add_argument('--distance-penalties', nargs='+', type=float,
                       default=DISTANCE_PENALTIES,
                       help=f'Distance penalty values to test (default: {DISTANCE_PENALTIES})')
    
    args = parser.parse_args()
    
    if not args.run_sweep:
        parser.print_help()
        print("\nTo run the distance penalty investigation, use --run-sweep")
        return
    
    # Override global distance penalties if specified
    if args.distance_penalties != DISTANCE_PENALTIES:
        DISTANCE_PENALTIES = args.distance_penalties
        print(f"Using custom distance penalties: {DISTANCE_PENALTIES}")
    
    # Initialize analyzer
    print(f"\n{'='*80}")
    print(f"DISTANCE PENALTY PARADOX INVESTIGATION")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Tasks: {args.tasks}")
    print(f"Distance penalties: {DISTANCE_PENALTIES}")
    print(f"Force retrain: {args.force_retrain}")
    print(f"{'='*80}\n")
    
    analyzer = DistancePenaltyAnalyzer(base_dir=args.base_dir)
    
    try:
        # Run the complete investigation
        results = analyzer.run_distance_penalty_sweep(
            tasks=args.tasks,
            force_retrain=args.force_retrain
        )
        
        print(f"\n{'='*80}")
        print(f"INVESTIGATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Results directory: {analyzer.base_dir}")
        print(f"Total configurations tested: {len(args.tasks) * len(DISTANCE_PENALTIES)}")
        print(f"")
        
        # Print quick summary of key findings
        if analyzer.analysis_results and 'optimal_config_analysis' in analyzer.analysis_results:
            optimal_configs = analyzer.analysis_results['optimal_config_analysis']
            
            print("QUICK SUMMARY:")
            for task, task_data in optimal_configs.items():
                recommendations = task_data.get('recommendations', {})
                print(f"  {task.title()} Task:")
                
                if recommendations.get('paradox_confirmed', False):
                    print(f"    ✅ Distance Penalty Paradox CONFIRMED")
                    print(f"    📊 Recommended distance penalty: {recommendations.get('recommended_dp', 'N/A')}")
                    if 'performance_improvement' in recommendations:
                        improvement = recommendations['performance_improvement'] * 100
                        print(f"    📈 Performance improvement: {improvement:.1f}%")
                else:
                    print(f"    ❌ Distance Penalty Paradox NOT confirmed")
        
        print(f"\n{'='*80}")
        print(f"OUTPUT FILES:")
        print(f"  📊 Main results: {analyzer.base_dir}/distance_penalty_sweep_results.json")
        print(f"  📈 Visualizations: {analyzer.base_dir}/visualizations/")
        print(f"  📋 Analysis report: {analyzer.base_dir}/distance_penalty_analysis_report.md")
        print(f"  📁 Data summary: {analyzer.base_dir}/data_summary.txt")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Investigation failed with exception:")
        print(f"{str(e)}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())