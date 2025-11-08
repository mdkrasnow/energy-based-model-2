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
import errno
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
from scipy import stats

# Import model components for diagnostics
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Addition, Inverse, LowRankDataset
from diffusion_lib.adversarial_corruption import _adversarial_corruption


# Hyperparameters
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
TRAIN_ITERATIONS = 50000
DIFFUSION_STEPS = 10
RANK = 20

# Random seeds for multiple trials
RANDOM_SEEDS = [42, 43, 44, 45, 46]

# Tasks to run (excluding addition per requirements)
TASKS = ['inverse', 'lowrank']

# ANM configurations to test
ANM_CONFIGS = [
    {
        'name': 'anm_eps0.9_steps135_dp0.04',
        'epsilon': 0.9,
        'adv_steps': 135,
        'distance_penalty': 0.04
    },
    {
        'name': 'anm_eps0.1_steps125_dp0.01',
        'epsilon': 0.1,
        'adv_steps': 125,
        'distance_penalty': 0.01
    },
    {
        'name': 'anm_eps1.0_steps25_dp0.002',
        'epsilon': 1.0,
        'adv_steps': 25,
        'distance_penalty': 0.002
    },
    {
        'name': 'anm_eps1.0_steps5_dp0.001',
        'epsilon': 1.0,
        'adv_steps': 5,
        'distance_penalty': 0.001
    }
]


class StatisticalValidator:
    def __init__(self, base_dir=None):
        """
        base_dir:
          - If provided, use it as-is.
          - If None, default to:
              $STATISTICAL_EXPERIMENT_DIR or 'experiments_statistical'
        """
        if base_dir is None:
            base_dir = os.environ.get("STATISTICAL_EXPERIMENT_DIR", "experiments_statistical")
            if not base_dir:  # Handle empty string case
                base_dir = "experiments_statistical"

        self.base_dir = Path(base_dir)
        if not self._robust_mkdir(self.base_dir):
            raise OSError(f"Failed to create base directory: {self.base_dir}")

        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.statistical_results = {}

    def _robust_mkdir(self, dir_path, max_retries=3):
        """Robustly create directory, handling stale file handles on network filesystems."""
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
        
    def get_result_dir(self, dataset, model_type='baseline', epsilon=None, adv_steps=None, 
                      distance_penalty=None, seed=None):
        """Get the results directory for a given dataset, model type, and seed"""
        base = f'results/ds_{dataset}/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        if model_type == 'anm':
            if epsilon is not None and adv_steps is not None:
                base += f'_anm_eps{epsilon}_steps{adv_steps}'
                if distance_penalty is not None:
                    base += f'_dp{distance_penalty}'
        if seed is not None:
            base += f'_seed{seed}'
        return base
    
    def train_model(self, dataset, model_type='baseline', seed=42, force_retrain=False,
                   epsilon=None, adv_steps=None, distance_penalty=None):
        """Train a model for a specific dataset, model type, and seed"""
        result_dir = self.get_result_dir(dataset, model_type, epsilon, adv_steps, distance_penalty, seed)
        
        # Check if model already exists
        if not force_retrain and self._robust_file_exists(f'{result_dir}/model-50.pt'):
            print(f"\n{'='*80}")
            print(f"Model for {dataset} ({model_type}, seed={seed}) already exists. Skipping training.")
            print(f"Use --force to retrain.")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return True
            
        config_desc = model_type
        if epsilon is not None:
            config_desc += f" (eps={epsilon}, steps={adv_steps}, dp={distance_penalty})"
        
        print(f"\n{'='*80}")
        print(f"Training IRED ({config_desc.upper()}) on {dataset.upper()} - Seed {seed}")
        print(f"{'='*80}")
        print(f"Model Type: {model_type}")
        if epsilon is not None and adv_steps is not None:
            print(f"ANM Hyperparameters: epsilon={epsilon}, adversarial_steps={adv_steps}, distance_penalty={distance_penalty}")
        print(f"Random Seed: {seed}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Training iterations: {TRAIN_ITERATIONS}")
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
            '--seed', str(seed),
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
                print(f"Training completed for {dataset} ({config_desc}, seed={seed}) in {elapsed/60:.2f} minutes")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return True
            else:
                print(f"\n{'='*80}")
                print(f"ERROR: Training failed for {dataset} ({config_desc}, seed={seed}) with exit code {result}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return False
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Training failed for {dataset} ({config_desc}, seed={seed}): {e}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            return False
    
    def evaluate_model(self, dataset, model_type='baseline', seed=42, epsilon=None, 
                      adv_steps=None, distance_penalty=None, ood=False):
        """Evaluate a trained model on same or harder difficulty"""
        result_dir = self.get_result_dir(dataset, model_type, epsilon, adv_steps, distance_penalty, seed)
        
        # Check if model exists
        if not self._robust_file_exists(f'{result_dir}/model-50.pt'):
            print(f"\n{'='*80}")
            print(f"ERROR: No trained model found for {dataset} ({model_type}, seed={seed})")
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
        print(f"Evaluating IRED ({config_desc.upper()}) on {dataset.upper()} - {difficulty} - Seed {seed}")
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
            '--seed', str(seed),
            '--load-milestone', '1',
            '--evaluate',
        ]
        
        # Add model-specific parameters
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
                print(f"Evaluation completed for {dataset} ({config_desc}, seed={seed}) - {difficulty}")
                if mse is not None:
                    print(f"MSE: {mse:.4f}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                return mse
            else:
                print(f"\n{'='*80}")
                print(f"ERROR: Evaluation failed for {dataset} ({config_desc}, seed={seed}) - {difficulty} with exit code {result}")
                print(f"{'='*80}\n")
                sys.stdout.flush()
                return None
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Evaluation failed for {dataset} ({config_desc}, seed={seed}) - {difficulty}: {e}")
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
    
    def run_all_experiments(self, force_retrain=False):
        """Run all experiments across seeds, tasks, and configurations"""
        print(f"\n{'#'*80}")
        print(f"# STATISTICAL VALIDATION OF ANM CONFIGURATIONS")
        print(f"# Tasks: {', '.join(TASKS)}")
        print(f"# Seeds: {', '.join(map(str, RANDOM_SEEDS))}")
        print(f"# Configurations: Baseline + {len(ANM_CONFIGS)} ANM variants")
        print(f"# Training Steps: {TRAIN_ITERATIONS}")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        for seed in RANDOM_SEEDS:
            print(f"\n{'#'*80}")
            print(f"# PROCESSING SEED {seed}")
            print(f"{'#'*80}\n")
            sys.stdout.flush()
            
            for dataset in TASKS:
                print(f"\n--- Dataset: {dataset.upper()} ---\n")
                sys.stdout.flush()
                
                # Train and evaluate baseline
                if self.train_model(dataset, 'baseline', seed, force_retrain):
                    same_mse = self.evaluate_model(dataset, 'baseline', seed, ood=False)
                    harder_mse = self.evaluate_model(dataset, 'baseline', seed, ood=True)
                    
                    if same_mse is not None:
                        self.results[dataset]['baseline']['same_difficulty'].append(same_mse)
                    if harder_mse is not None:
                        self.results[dataset]['baseline']['harder_difficulty'].append(harder_mse)
                
                # Train and evaluate each ANM configuration
                for config in ANM_CONFIGS:
                    if self.train_model(
                        dataset, 'anm', seed, force_retrain,
                        epsilon=config['epsilon'],
                        adv_steps=config['adv_steps'],
                        distance_penalty=config['distance_penalty']
                    ):
                        same_mse = self.evaluate_model(
                            dataset, 'anm', seed,
                            epsilon=config['epsilon'],
                            adv_steps=config['adv_steps'],
                            distance_penalty=config['distance_penalty'],
                            ood=False
                        )
                        harder_mse = self.evaluate_model(
                            dataset, 'anm', seed,
                            epsilon=config['epsilon'],
                            adv_steps=config['adv_steps'],
                            distance_penalty=config['distance_penalty'],
                            ood=True
                        )
                        
                        if same_mse is not None:
                            self.results[dataset][config['name']]['same_difficulty'].append(same_mse)
                        if harder_mse is not None:
                            self.results[dataset][config['name']]['harder_difficulty'].append(harder_mse)
        
        return self.results
    
    def compute_cohens_d(self, sample1, sample2):
        """Compute Cohen's d effect size"""
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(sample1) - np.mean(sample2)) / pooled_std
    
    def perform_statistical_analysis(self):
        """Perform statistical analysis on collected results"""
        print(f"\n{'#'*80}")
        print(f"# STATISTICAL ANALYSIS")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        self.statistical_results = defaultdict(lambda: defaultdict(dict))
        
        for dataset in TASKS:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset.upper()}")
            print(f"{'='*80}\n")
            
            baseline_same = self.results[dataset]['baseline']['same_difficulty']
            baseline_harder = self.results[dataset]['baseline']['harder_difficulty']
            
            if not baseline_same or not baseline_harder:
                print(f"Skipping {dataset} - missing baseline results\n")
                continue
            
            print(f"Baseline Statistics:")
            print(f"  Same Difficulty:   {np.mean(baseline_same):.6f} ± {np.std(baseline_same):.6f}")
            print(f"  Harder Difficulty: {np.mean(baseline_harder):.6f} ± {np.std(baseline_harder):.6f}")
            print()
            
            # Store baseline stats
            self.statistical_results[dataset]['baseline'] = {
                'same_difficulty': {
                    'mean': float(np.mean(baseline_same)),
                    'std': float(np.std(baseline_same)),
                    'values': [float(x) for x in baseline_same]
                },
                'harder_difficulty': {
                    'mean': float(np.mean(baseline_harder)),
                    'std': float(np.std(baseline_harder)),
                    'values': [float(x) for x in baseline_harder]
                }
            }
            
            for config in ANM_CONFIGS:
                anm_same = self.results[dataset][config['name']]['same_difficulty']
                anm_harder = self.results[dataset][config['name']]['harder_difficulty']
                
                if not anm_same or not anm_harder:
                    print(f"Skipping {config['name']} - missing results\n")
                    continue
                
                print(f"\n{config['name']}:")
                print(f"  Configuration: eps={config['epsilon']}, steps={config['adv_steps']}, dp={config['distance_penalty']}")
                print(f"  Same Difficulty:   {np.mean(anm_same):.6f} ± {np.std(anm_same):.6f}")
                print(f"  Harder Difficulty: {np.mean(anm_harder):.6f} ± {np.std(anm_harder):.6f}")
                
                # Paired t-test (same difficulty)
                t_stat_same, p_value_same = stats.ttest_rel(baseline_same, anm_same)
                cohens_d_same = self.compute_cohens_d(baseline_same, anm_same)
                
                # Paired t-test (harder difficulty)
                t_stat_harder, p_value_harder = stats.ttest_rel(baseline_harder, anm_harder)
                cohens_d_harder = self.compute_cohens_d(baseline_harder, anm_harder)
                
                # Improvement percentage
                improvement_same = ((np.mean(baseline_same) - np.mean(anm_same)) / np.mean(baseline_same)) * 100
                improvement_harder = ((np.mean(baseline_harder) - np.mean(anm_harder)) / np.mean(baseline_harder)) * 100
                
                print(f"\n  Statistical Tests (Same Difficulty):")
                print(f"    t-statistic: {t_stat_same:.4f}")
                print(f"    p-value: {p_value_same:.6f} {'***' if p_value_same < 0.001 else '**' if p_value_same < 0.01 else '*' if p_value_same < 0.05 else 'ns'}")
                print(f"    Cohen's d: {cohens_d_same:.4f}")
                print(f"    Improvement: {improvement_same:+.2f}%")
                
                print(f"\n  Statistical Tests (Harder Difficulty):")
                print(f"    t-statistic: {t_stat_harder:.4f}")
                print(f"    p-value: {p_value_harder:.6f} {'***' if p_value_harder < 0.001 else '**' if p_value_harder < 0.01 else '*' if p_value_harder < 0.05 else 'ns'}")
                print(f"    Cohen's d: {cohens_d_harder:.4f}")
                print(f"    Improvement: {improvement_harder:+.2f}%")
                
                # Store results
                self.statistical_results[dataset][config['name']] = {
                    'same_difficulty': {
                        'mean': float(np.mean(anm_same)),
                        'std': float(np.std(anm_same)),
                        'values': [float(x) for x in anm_same],
                        't_statistic': float(t_stat_same),
                        'p_value': float(p_value_same),
                        'cohens_d': float(cohens_d_same),
                        'improvement_percent': float(improvement_same),
                        'significant': bool(p_value_same < 0.05)
                    },
                    'harder_difficulty': {
                        'mean': float(np.mean(anm_harder)),
                        'std': float(np.std(anm_harder)),
                        'values': [float(x) for x in anm_harder],
                        't_statistic': float(t_stat_harder),
                        'p_value': float(p_value_harder),
                        'cohens_d': float(cohens_d_harder),
                        'improvement_percent': float(improvement_harder),
                        'significant': bool(p_value_harder < 0.05)
                    },
                    'config': config
                }
        
        sys.stdout.flush()
    
    def create_visualizations(self):
        """Create box plots for each dataset and difficulty level"""
        print(f"\n{'#'*80}")
        print(f"# CREATING VISUALIZATIONS")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        viz_dir = self.base_dir / 'visualizations'
        if not self._robust_mkdir(viz_dir):
            print(f"Error: Failed to create visualization directory: {viz_dir}")
            return
        
        for dataset in TASKS:
            # Check if raw results exist for this dataset
            if dataset not in self.results or 'baseline' not in self.results[dataset]:
                print(f"Skipping {dataset.upper()} visualizations - no raw results available")
                continue
            
            for difficulty in ['same_difficulty', 'harder_difficulty']:
                # Check if baseline data exists for this difficulty
                baseline_results = self.results[dataset].get('baseline', {})
                if difficulty not in baseline_results:
                    print(f"Skipping {dataset.upper()} {difficulty} visualization - no baseline data")
                    continue
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Collect data for plotting
                data_to_plot = []
                labels = []
                colors = []
                
                # Baseline
                baseline_data = baseline_results[difficulty]
                if baseline_data:
                    data_to_plot.append(baseline_data)
                    labels.append('Baseline')
                    colors.append('lightblue')
                
                # ANM configurations
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    # Check if config results exist
                    if config_name in self.results[dataset] and difficulty in self.results[dataset][config_name]:
                        anm_data = self.results[dataset][config_name][difficulty]
                        if anm_data:
                            data_to_plot.append(anm_data)
                            label = f"eps={config['epsilon']}\nsteps={config['adv_steps']}\ndp={config['distance_penalty']}"
                            labels.append(label)
                            
                            # Highlight promising config
                            if config['epsilon'] == 1.0 and config['adv_steps'] == 5:
                                colors.append('lightgreen')
                            else:
                                colors.append('lightcoral')
                
                # Robust data validation and filtering
                print(f"  DEBUG: Before validation - {len(data_to_plot)} data arrays, {len(labels)} labels")
                
                # Filter out empty data arrays and corresponding labels/colors
                valid_indices = []
                for i, data_array in enumerate(data_to_plot):
                    # Check if data_array exists and has valid content
                    if data_array is not None and len(data_array) > 0:
                        # Simple check for valid numeric data
                        try:
                            # Convert to numpy array for validation
                            arr = np.array(data_array)
                            if not np.isnan(arr).all():  # At least some values are not NaN
                                valid_indices.append(i)
                            else:
                                print(f"  WARNING: Filtering out all-NaN data for '{labels[i] if i < len(labels) else 'unknown'}'")
                        except (ValueError, TypeError):
                            print(f"  WARNING: Filtering out invalid data type for '{labels[i] if i < len(labels) else 'unknown'}'")
                    else:
                        print(f"  WARNING: Filtering out empty/null data for '{labels[i] if i < len(labels) else 'unknown'}'")
                
                if not valid_indices:
                    print(f"  ERROR: No valid data arrays found for {dataset} {difficulty}. Skipping visualization.")
                    continue
                
                # Keep only valid data, labels, and colors
                data_to_plot = [data_to_plot[i] for i in valid_indices]
                labels = [labels[i] for i in valid_indices] 
                colors = [colors[i] for i in valid_indices]
                
                print(f"  DEBUG: After validation - {len(data_to_plot)} valid data arrays")
                
                # Final dimension check
                if len(data_to_plot) != len(labels) or len(data_to_plot) != len(colors):
                    print(f"  ERROR: Dimension mismatch after filtering - data:{len(data_to_plot)}, labels:{len(labels)}, colors:{len(colors)}")
                    continue
                
                # Create box plot with comprehensive error handling
                print(f"  DEBUG: Creating boxplot with {len(data_to_plot)} data arrays, {len(labels)} labels")
                for i, (data_array, label) in enumerate(zip(data_to_plot, labels)):
                    try:
                        min_val, max_val = np.min(data_array), np.max(data_array)
                        print(f"    [{i}] {label}: {len(data_array)} values, range [{min_val:.4f}, {max_val:.4f}]")
                    except (ValueError, TypeError):
                        print(f"    [{i}] {label}: {len(data_array)} values, range [invalid data]")
                
                try:
                    # Try new parameter name (matplotlib >= 3.9)
                    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                                   widths=0.6, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
                except TypeError as e:
                    if 'tick_labels' in str(e):
                        # Fall back to old parameter name (matplotlib < 3.9) 
                        print(f"  DEBUG: Falling back to 'labels' parameter for older matplotlib version")
                        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                                       widths=0.6, showmeans=True,
                                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
                    else:
                        print(f"  ERROR: Unexpected TypeError in boxplot creation: {e}")
                        print(f"  DEBUG: Data shapes: {[np.array(d).shape for d in data_to_plot]}")
                        continue
                except ValueError as e:
                    print(f"  ERROR: ValueError in boxplot creation: {e}")
                    print(f"  DEBUG: Data info - arrays: {len(data_to_plot)}, labels: {len(labels)}")
                    print(f"  DEBUG: Data shapes: {[np.array(d).shape for d in data_to_plot]}")
                    print(f"  DEBUG: Data samples: {[np.array(d)[:3] if len(d) >= 3 else np.array(d) for d in data_to_plot]}")
                    continue
                except Exception as e:
                    print(f"  ERROR: Unexpected error in boxplot creation: {type(e).__name__}: {e}")
                    print(f"  DEBUG: data_to_plot type: {type(data_to_plot)}, length: {len(data_to_plot)}")
                    continue
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                # Add significance markers
                if (dataset in self.statistical_results and 
                    ANM_CONFIGS and
                    ANM_CONFIGS[0]['name'] in self.statistical_results[dataset] and
                    difficulty in self.statistical_results[dataset][ANM_CONFIGS[0]['name']]):
                    
                    y_max = max([max(d) for d in data_to_plot]) * 1.1
                    for i, config in enumerate(ANM_CONFIGS, start=1):
                        config_name = config['name']
                        if (config_name in self.statistical_results[dataset] and
                            difficulty in self.statistical_results[dataset][config_name]):
                            
                            difficulty_stats = self.statistical_results[dataset][config_name][difficulty]
                            if 'p_value' in difficulty_stats:
                                p_value = difficulty_stats['p_value']
                                if p_value < 0.001:
                                    ax.text(i + 1, y_max, '***', ha='center', fontsize=14, fontweight='bold')
                                elif p_value < 0.01:
                                    ax.text(i + 1, y_max, '**', ha='center', fontsize=14, fontweight='bold')
                                elif p_value < 0.05:
                                    ax.text(i + 1, y_max, '*', ha='center', fontsize=14, fontweight='bold')
                
                difficulty_title = "Same Difficulty" if difficulty == 'same_difficulty' else "Harder Difficulty (OOD)"
                ax.set_title(f'{dataset.upper()} - {difficulty_title}\nMSE Distribution Across 5 Random Seeds', 
                           fontsize=14, fontweight='bold')
                ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
                ax.set_xlabel('Configuration', fontsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                plt.xticks(rotation=0, ha='center', fontsize=9)
                plt.tight_layout()
                
                # Save figure
                filename = f'{dataset}_{difficulty}_boxplot.png'
                filepath = viz_dir / filename
                if self._robust_savefig(filepath, dpi=300, bbox_inches='tight'):
                    print(f"Saved: {filepath}")
                else:
                    print(f"Failed to save: {filepath}")
                plt.close()
        
        print(f"\n{'#'*80}\n")
        sys.stdout.flush()
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report"""
        print(f"\n{'#'*80}")
        print(f"# GENERATING MARKDOWN REPORT")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        report_path = self.base_dir / 'statistical_validation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Statistical Validation of Adversarial Negative Mining (ANM) Configurations\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Training Iterations:** {TRAIN_ITERATIONS}\n\n")
            f.write(f"**Random Seeds:** {', '.join(map(str, RANDOM_SEEDS))}\n\n")
            f.write(f"**Number of Trials:** {len(RANDOM_SEEDS)}\n\n")
            
            f.write("## Experimental Setup\n\n")
            f.write("This report presents a rigorous statistical validation of four ANM hyperparameter configurations ")
            f.write("compared against the baseline IRED method. Each configuration was trained and evaluated 5 times ")
            f.write("with different random seeds to assess the statistical significance of observed improvements.\n\n")
            
            f.write("### Tested Configurations\n\n")
            f.write("| Configuration | Epsilon | Adversarial Steps | Distance Penalty |\n")
            f.write("|--------------|---------|-------------------|------------------|\n")
            f.write("| Baseline (IRED) | N/A | N/A | N/A |\n")
            for config in ANM_CONFIGS:
                f.write(f"| {config['name']} | {config['epsilon']} | {config['adv_steps']} | {config['distance_penalty']} |\n")
            f.write("\n")
            
            f.write("### Tasks\n\n")
            f.write("- **Matrix Inverse** (inverse): Predicting matrix inverses\n")
            f.write("- **Matrix Completion** (lowrank): Low-rank matrix completion\n\n")
            
            f.write("### Evaluation Settings\n\n")
            f.write("- **Same Difficulty:** Test on same distribution as training\n")
            f.write("- **Harder Difficulty (OOD):** Test on out-of-distribution harder problems\n\n")
            
            f.write("---\n\n")
            
            # Results for each task
            for dataset in TASKS:
                # Check if statistical results exist for this dataset
                if dataset not in self.statistical_results or 'baseline' not in self.statistical_results[dataset]:
                    task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                    f.write(f"## {task_name} Results\n\n")
                    f.write("*No statistical results available for this dataset.*\n\n")
                    f.write("---\n\n")
                    continue
                    
                # Check if baseline statistical results exist
                baseline_stats = self.statistical_results[dataset].get('baseline', {})
                if 'same_difficulty' not in baseline_stats or 'harder_difficulty' not in baseline_stats:
                    task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                    f.write(f"## {task_name} Results\n\n")
                    f.write("*Incomplete baseline statistical results for this dataset.*\n\n")
                    f.write("---\n\n")
                    continue
                
                task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                f.write(f"## {task_name} Results\n\n")
                
                # Summary table
                f.write("### Performance Summary\n\n")
                f.write("| Configuration | Same Difficulty MSE | Harder Difficulty MSE |\n")
                f.write("|--------------|---------------------|----------------------|\n")
                
                # Baseline
                baseline_same = baseline_stats['same_difficulty']
                baseline_harder = baseline_stats['harder_difficulty']
                f.write(f"| Baseline | {baseline_same['mean']:.6f} ± {baseline_same['std']:.6f} | ")
                f.write(f"{baseline_harder['mean']:.6f} ± {baseline_harder['std']:.6f} |\n")
                
                # ANM configs
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    if config_name in self.statistical_results[dataset]:
                        config_stats = self.statistical_results[dataset][config_name]
                        
                        # Check if both difficulty levels exist
                        if 'same_difficulty' not in config_stats or 'harder_difficulty' not in config_stats:
                            config_label = f"{config_name}"
                            f.write(f"| {config_label} | *Incomplete data* | *Incomplete data* |\n")
                            continue
                        
                        same_stats = config_stats['same_difficulty']
                        harder_stats = config_stats['harder_difficulty']
                        
                        config_label = f"{config_name}"
                        f.write(f"| {config_label} | {same_stats['mean']:.6f} ± {same_stats['std']:.6f} | ")
                        f.write(f"{harder_stats['mean']:.6f} ± {harder_stats['std']:.6f} |\n")
                f.write("\n")
                
                # Statistical significance table
                f.write("### Statistical Significance Tests\n\n")
                f.write("Paired t-tests comparing each ANM configuration against baseline (5 paired samples).\n\n")
                
                f.write("#### Same Difficulty\n\n")
                f.write("| Configuration | Improvement | p-value | Cohen's d | Significant? |\n")
                f.write("|--------------|-------------|---------|-----------|-------------|\n")
                
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    if config_name in self.statistical_results[dataset]:
                        config_stats = self.statistical_results[dataset][config_name]
                        
                        # Check if same difficulty stats exist
                        if 'same_difficulty' not in config_stats:
                            f.write(f"| {config_name} | *No data* | *No data* | *No data* | *No data* |\n")
                            continue
                            
                        stats_same = config_stats['same_difficulty']
                        
                        improvement = stats_same['improvement_percent']
                        p_val = stats_same['p_value']
                        cohens_d = stats_same['cohens_d']
                        sig = "✅ Yes" if stats_same['significant'] else "❌ No"
                        
                        sig_marker = ""
                        if p_val < 0.001:
                            sig_marker = "***"
                        elif p_val < 0.01:
                            sig_marker = "**"
                        elif p_val < 0.05:
                            sig_marker = "*"
                        
                        f.write(f"| {config_name} | {improvement:+.2f}% | {p_val:.6f}{sig_marker} | {cohens_d:.4f} | {sig} |\n")
                f.write("\n")
                f.write("*Significance levels: *** p<0.001, ** p<0.01, * p<0.05*\n\n")
                
                f.write("#### Harder Difficulty (OOD)\n\n")
                f.write("| Configuration | Improvement | p-value | Cohen's d | Significant? |\n")
                f.write("|--------------|-------------|---------|-----------|-------------|\n")
                
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    if config_name in self.statistical_results[dataset]:
                        config_stats = self.statistical_results[dataset][config_name]
                        
                        # Check if harder difficulty stats exist
                        if 'harder_difficulty' not in config_stats:
                            f.write(f"| {config_name} | *No data* | *No data* | *No data* | *No data* |\n")
                            continue
                            
                        stats_harder = config_stats['harder_difficulty']
                        
                        improvement = stats_harder['improvement_percent']
                        p_val = stats_harder['p_value']
                        cohens_d = stats_harder['cohens_d']
                        sig = "✅ Yes" if stats_harder['significant'] else "❌ No"
                        
                        sig_marker = ""
                        if p_val < 0.001:
                            sig_marker = "***"
                        elif p_val < 0.01:
                            sig_marker = "**"
                        elif p_val < 0.05:
                            sig_marker = "*"
                        
                        f.write(f"| {config_name} | {improvement:+.2f}% | {p_val:.6f}{sig_marker} | {cohens_d:.4f} | {sig} |\n")
                f.write("\n")
                f.write("*Significance levels: *** p<0.001, ** p<0.01, * p<0.05*\n\n")
                
                # Visualizations
                f.write("### Visualizations\n\n")
                f.write(f"![{task_name} Same Difficulty](visualizations/{dataset}_same_difficulty_boxplot.png)\n\n")
                f.write(f"![{task_name} Harder Difficulty](visualizations/{dataset}_harder_difficulty_boxplot.png)\n\n")
                
                f.write("---\n\n")
            
            # Overall conclusions
            f.write("## Overall Conclusions\n\n")
            
            f.write("### Summary of Statistical Significance\n\n")
            
            for dataset in TASKS:
                # Skip datasets with no statistical results
                if dataset not in self.statistical_results or 'baseline' not in self.statistical_results[dataset]:
                    continue
                    
                task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                f.write(f"**{task_name}:**\n\n")
                
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    if config_name in self.statistical_results[dataset]:
                        config_stats = self.statistical_results[dataset][config_name]
                        
                        # Check if both difficulty stats exist
                        if 'same_difficulty' not in config_stats or 'harder_difficulty' not in config_stats:
                            f.write(f"- **{config_name}**: *Incomplete data*\n")
                            continue
                        
                        same_stats = config_stats['same_difficulty']
                        harder_stats = config_stats['harder_difficulty']
                        
                        same_sig = same_stats['significant']
                        harder_sig = harder_stats['significant']
                        same_imp = same_stats['improvement_percent']
                        harder_imp = harder_stats['improvement_percent']
                        
                        f.write(f"- **{config_name}**:\n")
                        f.write(f"  - Same Difficulty: {same_imp:+.2f}% ({'✅ Significant' if same_sig else '❌ Not Significant'})\n")
                        f.write(f"  - Harder Difficulty: {harder_imp:+.2f}% ({'✅ Significant' if harder_sig else '❌ Not Significant'})\n")
                f.write("\n")
            
            f.write("### Key Findings\n\n")
            
            # Find best performing configuration
            best_configs = {}
            for dataset in TASKS:
                # Skip datasets with no statistical results
                if dataset not in self.statistical_results or 'baseline' not in self.statistical_results[dataset]:
                    continue
                    
                best_same_imp = -float('inf')
                best_same_config = None
                best_harder_imp = -float('inf')
                best_harder_config = None
                
                for config in ANM_CONFIGS:
                    config_name = config['name']
                    if config_name in self.statistical_results[dataset]:
                        config_stats = self.statistical_results[dataset][config_name]
                        
                        # Check if both difficulty stats exist
                        if 'same_difficulty' not in config_stats or 'harder_difficulty' not in config_stats:
                            continue
                            
                        same_imp = config_stats['same_difficulty']['improvement_percent']
                        harder_imp = config_stats['harder_difficulty']['improvement_percent']
                        
                        if same_imp > best_same_imp:
                            best_same_imp = same_imp
                            best_same_config = config
                        
                        if harder_imp > best_harder_imp:
                            best_harder_imp = harder_imp
                            best_harder_config = config
                
                if best_same_config is not None and best_harder_config is not None:
                    best_configs[dataset] = {
                        'same': (best_same_config, best_same_imp),
                        'harder': (best_harder_config, best_harder_imp)
                    }
            
            f.write("1. **Best Performing Configurations:**\n")
            for dataset in TASKS:
                # Skip datasets with no best configs
                if dataset not in best_configs:
                    task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                    f.write(f"   - **{task_name}:** *No data available*\n")
                    continue
                    
                task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                f.write(f"   - **{task_name}:**\n")
                
                same_config, same_imp = best_configs[dataset]['same']
                harder_config, harder_imp = best_configs[dataset]['harder']
                
                f.write(f"     - Same Difficulty: {same_config['name']} ({same_imp:+.2f}%)\n")
                f.write(f"     - Harder Difficulty: {harder_config['name']} ({harder_imp:+.2f}%)\n")
            f.write("\n")
            
            f.write("2. **Eps=1.0, Steps=5, DP=0.001 Configuration:**\n")
            f.write("   As noted in preliminary results, this configuration deserves special attention:\n")
            for dataset in TASKS:
                # Skip datasets with no statistical results
                if dataset not in self.statistical_results or 'baseline' not in self.statistical_results[dataset]:
                    continue
                    
                task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
                config_name = 'anm_eps1.0_steps5_dp0.001'
                
                if config_name in self.statistical_results[dataset]:
                    config_stats = self.statistical_results[dataset][config_name]
                    
                    # Check if both difficulty stats exist
                    if 'same_difficulty' not in config_stats or 'harder_difficulty' not in config_stats:
                        f.write(f"   - **{task_name}:** *Incomplete data*\n")
                        continue
                        
                    same_stats = config_stats['same_difficulty']
                    harder_stats = config_stats['harder_difficulty']
                    
                    f.write(f"   - **{task_name}:**\n")
                    f.write(f"     - Same Difficulty: {same_stats['improvement_percent']:+.2f}% ")
                    f.write(f"(p={same_stats['p_value']:.6f}, {'significant' if same_stats['significant'] else 'not significant'})\n")
                    f.write(f"     - Harder Difficulty: {harder_stats['improvement_percent']:+.2f}% ")
                    f.write(f"(p={harder_stats['p_value']:.6f}, {'significant' if harder_stats['significant'] else 'not significant'})\n")
            f.write("\n")
            
            f.write("3. **Statistical Robustness:**\n")
            f.write("   Configurations showing consistent improvements across all 5 random seeds demonstrate ")
            f.write("robust performance rather than sensitivity to initialization.\n\n")
            
            f.write("4. **Effect Sizes:**\n")
            f.write("   Cohen's d values provide a measure of practical significance beyond statistical significance. ")
            f.write("Generally:\n")
            f.write("   - |d| < 0.2: Small effect\n")
            f.write("   - |d| ≈ 0.5: Medium effect\n")
            f.write("   - |d| > 0.8: Large effect\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("Based on the statistical analysis:\n\n")
            f.write("1. Configurations showing both statistical significance (p < 0.05) and meaningful effect sizes ")
            f.write("(|Cohen's d| > 0.5) should be prioritized for further investigation.\n\n")
            f.write("2. The variability across random seeds (indicated by standard deviations) should be considered ")
            f.write("when interpreting results - lower variance indicates more stable performance.\n\n")
            f.write("3. Configurations that perform well on harder difficulty (OOD) tests demonstrate better ")
            f.write("generalization capabilities.\n\n")
            
            f.write("---\n\n")
            f.write(f"*Report generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}*\n")
        
        print(f"Markdown report saved to: {report_path}\n")
        sys.stdout.flush()
        
        return report_path
    
    def save_results_json(self):
        """Save all results to JSON file"""
        print(f"\n{'#'*80}")
        print(f"# SAVING RESULTS TO JSON")
        print(f"{'#'*80}\n")
        sys.stdout.flush()
        
        json_path = self.base_dir / 'statistical_validation_results.json'
        
        results_dict = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'train_iterations': TRAIN_ITERATIONS,
                'diffusion_steps': DIFFUSION_STEPS,
                'rank': RANK,
                'random_seeds': RANDOM_SEEDS,
                'num_trials': len(RANDOM_SEEDS),
                'tasks': TASKS,
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
            'raw_results': dict(self.results),
            'statistical_analysis': dict(self.statistical_results)
        }
        
        if self._robust_write_file(json_path, results_dict):
            print(f"Results saved to: {json_path}\n")
        else:
            print(f"Failed to save results to: {json_path}\n")
        sys.stdout.flush()
        
        return json_path
    
    def print_summary_table(self):
        """Print a concise summary table to console"""
        print(f"\n{'#'*80}")
        print(f"# SUMMARY TABLE")
        print(f"{'#'*80}\n")
        
        for dataset in TASKS:
            # Check if statistical results exist for this dataset
            if dataset not in self.statistical_results or 'baseline' not in self.statistical_results[dataset]:
                print(f"\nSkipping {dataset.upper()} - no statistical results available")
                continue
                
            # Check if baseline statistical results exist
            baseline_stats = self.statistical_results[dataset].get('baseline', {})
            if 'same_difficulty' not in baseline_stats or 'harder_difficulty' not in baseline_stats:
                print(f"\nSkipping {dataset.upper()} - incomplete baseline statistical results")
                continue
            
            task_name = "Matrix Inverse" if dataset == 'inverse' else "Matrix Completion"
            print(f"\n{task_name}:")
            print(f"{'='*80}")
            print(f"{'Configuration':<40s} {'Same Diff':>15s} {'Harder Diff':>15s} {'Sig?':>10s}")
            print(f"{'-'*40} {'-'*15} {'-'*15} {'-'*10}")
            
            # Baseline
            baseline_same = baseline_stats['same_difficulty']
            baseline_harder = baseline_stats['harder_difficulty']
            print(f"{'Baseline':<40s} {baseline_same['mean']:>15.6f} {baseline_harder['mean']:>15.6f} {'N/A':>10s}")
            
            # ANM configs
            for config in ANM_CONFIGS:
                config_name = config['name']
                if config_name in self.statistical_results[dataset]:
                    config_stats = self.statistical_results[dataset][config_name]
                    
                    # Check if both difficulty levels exist
                    if 'same_difficulty' not in config_stats or 'harder_difficulty' not in config_stats:
                        print(f"{'  └─ ' + config_name:<40s} {'Incomplete data':>41s}")
                        continue
                    
                    same_stats = config_stats['same_difficulty']
                    harder_stats = config_stats['harder_difficulty']
                    
                    config_label = f"{config_name}"
                    same_sig = "✅ Yes" if same_stats['significant'] else "❌ No"
                    harder_sig = "✅ Yes" if harder_stats['significant'] else "❌ No"
                    
                    print(f"{config_label:<40s} {same_stats['mean']:>15.6f} {harder_stats['mean']:>15.6f} {same_sig:>10s}")
                    print(f"{'  └─ Improvement':<40s} {same_stats['improvement_percent']:>14.2f}% {harder_stats['improvement_percent']:>14.2f}% {harder_sig:>10s}")
                    print(f"{'  └─ p-value':<40s} {same_stats['p_value']:>15.6f} {harder_stats['p_value']:>15.6f}")
        
        print(f"\n{'#'*80}\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Statistical Validation of ANM Configurations')
    parser.add_argument('--base-dir', default='experiments_statistical', 
                       help='Base directory for experiments')
    parser.add_argument('--force', action='store_true', 
                       help='Force retrain even if models exist')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only perform analysis (requires existing models)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    args = parser.parse_args()
    
    validator = StatisticalValidator(base_dir=args.base_dir)
    
    # Run experiments
    if not args.skip_training:
        results = validator.run_all_experiments(force_retrain=args.force)
    else:
        print("\nSkipping training - loading existing results...\n")
        # Would need to implement loading from saved checkpoints
        print("ERROR: --skip-training not yet implemented. Please run full pipeline.")
        return
    
    # Perform statistical analysis
    validator.perform_statistical_analysis()
    
    # Create visualizations
    if not args.skip_viz:
        validator.create_visualizations()
    
    # Generate reports
    validator.print_summary_table()
    json_path = validator.save_results_json()
    md_path = validator.generate_markdown_report()
    
    print(f"\n{'#'*80}")
    print(f"# VALIDATION COMPLETE")
    print(f"{'#'*80}")
    print(f"\nGenerated outputs:")
    print(f"  - JSON results: {json_path}")
    print(f"  - Markdown report: {md_path}")
    if not args.skip_viz:
        print(f"  - Visualizations: {validator.base_dir / 'visualizations'}")
    print(f"\n{'#'*80}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()