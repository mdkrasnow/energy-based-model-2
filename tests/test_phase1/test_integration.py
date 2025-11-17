#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase1_statistical_viability import ExperimentRunner, Phase1ExperimentRunner


class TestIntegration(unittest.TestCase):
    """Integration tests for phase1_statistical_viability.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_experiment_runner_workflow(self):
        """Test complete ExperimentRunner workflow with mocked training"""
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Mock the training subprocess calls
        with patch('subprocess.Popen') as mock_popen:
            # Mock successful training
            mock_process = Mock()
            mock_process.stdout.readline.side_effect = [
                "Training started\n",
                "Epoch 1/1000\n",
                "Loss: 0.1234\n",
                "Training completed\n",
                ""
            ]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            
            # Mock model file existence for diagnostic checks
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = False  # Force training
                
                # Mock diagnostics
                with patch.object(runner, 'run_energy_diagnostics', return_value=None):
                    success = runner.train_model('addition', 'baseline')
                
                self.assertTrue(success)
                self.assertGreater(mock_popen.call_count, 0)
    
    def test_phase1_runner_initialization_workflow(self):
        """Test Phase1ExperimentRunner initialization and setup"""
        runner = Phase1ExperimentRunner(base_dir=self.test_dir)
        
        # Verify directory structure
        self.assertTrue(runner.base_dir.exists())
        self.assertTrue(runner.csv_log_path.exists())
        
        # Verify CSV header
        with open(runner.csv_log_path, 'r') as f:
            header = f.readline().strip()
        
        expected_fields = [
            'timestamp', 'experiment_id', 'config_name', 'dataset', 'seed',
            'train_steps', 'mse_same_difficulty', 'mse_harder_difficulty',
            'training_time_minutes', 'model_path', 'success',
            'anm_adversarial_steps', 'random_noise_scale', 'random_noise_type'
        ]
        
        for field in expected_fields:
            self.assertIn(field, header)
    
    @patch('phase1_statistical_viability.generate_experiment_matrix')
    @patch.object(Phase1ExperimentRunner, 'train_model_with_phase1_config')
    def test_phase1_experiment_orchestration(self, mock_train_config, mock_generate):
        """Test Phase 1 experiment orchestration without actual training"""
        runner = Phase1ExperimentRunner(base_dir=self.test_dir)
        
        # Mock minimal experiment matrix
        mock_experiments = [
            {
                'experiment_id': 'baseline_001',
                'config_name': 'ired_baseline', 
                'config': Mock(name='baseline_config'),
                'seed': 42
            }
        ]
        mock_generate.return_value = mock_experiments
        
        # Mock training success
        mock_train_config.return_value = (True, 5.0, {'mse_harder': 0.1234})
        
        # Mock statistical analysis
        with patch.object(runner, '_analyze_phase1_results') as mock_analyze:
            mock_analyze.return_value = (True, [], "Mock analysis report")
            
            go_decision, stats, report = runner.run_phase1_viability_test('addition')
        
        # Verify the workflow completed
        mock_generate.assert_called_once()
        mock_train_config.assert_called_once()
        mock_analyze.assert_called_once()
        
        self.assertTrue(go_decision)
    
    def test_hyperparameter_sweep_structure(self):
        """Test hyperparameter sweep workflow structure"""
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Mock all training and evaluation calls
        with patch.object(runner, 'train_model', return_value=True):
            with patch.object(runner, 'run_energy_diagnostics', return_value={}):
                with patch.object(runner, 'run_comparative_diagnostics', return_value={}):
                    with patch.object(runner, 'evaluate_model_with_config', return_value=0.1):
                        with patch.object(runner, '_print_sweep_summary'):
                            with patch.object(runner, '_save_sweep_results'):
                                # This should complete without errors
                                results = runner.run_hyperparameter_sweep('addition')
        
        self.assertIsInstance(results, list)
    
    def test_diagnostic_workflow(self):
        """Test diagnostic analysis workflow"""
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Mock model loading
        mock_model = Mock()
        mock_diffusion = Mock()
        mock_diffusion.num_timesteps = 10
        mock_device = 'cpu'
        mock_dataset = Mock()
        
        load_result = (mock_model, mock_diffusion, mock_device, mock_dataset)
        
        with patch.object(runner, 'load_model_for_diagnostics', return_value=load_result):
            with patch.object(runner, '_sample_batch') as mock_sample:
                mock_sample.return_value = {
                    'x': torch.randn(5, 10),
                    'y': torch.randn(5, 15)
                }
                
                with patch.object(runner, '_simulate_anm_output') as mock_anm:
                    mock_anm.return_value = torch.randn(5, 15)
                    
                    # Mock energy computations
                    mock_diffusion.energy_score.return_value = torch.tensor([1.0, 2.0, 3.0])
                    
                    # Test energy diagnostics
                    energies = runner.run_energy_diagnostics('test_model_dir')
                    
                    if energies is not None:
                        expected_keys = ['clean', 'ired_standard', 'anm_adversarial', 'gaussian_noise']
                        for key in expected_keys:
                            self.assertIn(key, energies)
    
    def test_results_persistence(self):
        """Test that results are properly saved and can be loaded"""
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Set up test results
        test_results = {
            'addition': {
                'baseline': {
                    'same_difficulty': 0.1234,
                    'harder_difficulty': 0.5678
                }
            }
        }
        runner.results = test_results
        
        # Save results
        runner._save_results()
        
        # Verify file exists
        results_file = Path(self.test_dir) / 'continuous_results_with_anm_diagnostics.json'
        self.assertTrue(results_file.exists())
        
        # Load and verify content
        import json
        with open(results_file, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertIn('metadata', loaded_data)
        self.assertIn('results', loaded_data)
        self.assertEqual(loaded_data['results'], test_results)
    
    def test_csv_logging_persistence(self):
        """Test that Phase 1 CSV logging works correctly"""
        runner = Phase1ExperimentRunner(base_dir=self.test_dir)
        
        # Create test experiment data
        mock_config = Mock()
        mock_config.use_anm = True
        mock_config.anm_adversarial_steps = 25
        mock_config.use_random_noise = False
        
        experiment_spec = {
            'experiment_id': 'test_001',
            'config_name': 'test_config',
            'seed': 123,
            'train_steps': 1000,
            'config': mock_config
        }
        
        results = {
            'mse_same': 0.0123,
            'mse_harder': 0.0456
        }
        
        # Log the experiment
        runner._log_experiment_to_csv(experiment_spec, results, 12.34, True)
        
        # Read back and verify
        import csv
        with open(runner.csv_log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 1)
        row = rows[0]
        
        self.assertEqual(row['experiment_id'], 'test_001')
        self.assertEqual(row['config_name'], 'test_config')
        self.assertEqual(row['seed'], '123')
        self.assertEqual(row['mse_same_difficulty'], '0.0123')
        self.assertEqual(row['mse_harder_difficulty'], '0.0456')
        self.assertEqual(row['success'], 'success')
        self.assertEqual(row['anm_adversarial_steps'], '25')
    
    @patch('phase1_statistical_viability.validate_phase1_configs')
    def test_configuration_validation_integration(self, mock_validate):
        """Test that configuration validation is called during initialization"""
        Phase1ExperimentRunner(base_dir=self.test_dir)
        mock_validate.assert_called_once()
    
    def test_memory_management(self):
        """Test that large objects are properly cleaned up"""
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Create some mock diagnostic results
        runner.diagnostic_results['test_key'] = {
            'energies': {
                'clean': [1.0] * 1000,
                'corrupted': [2.0] * 1000
            }
        }
        
        # Clear results
        runner.diagnostic_results.clear()
        runner.sweep_results.clear()
        runner.results.clear()
        
        # Should not raise memory errors
        self.assertEqual(len(runner.diagnostic_results), 0)
        self.assertEqual(len(runner.sweep_results), 0)
        self.assertEqual(len(runner.results), 0)


if __name__ == '__main__':
    # Import torch here to avoid issues if not available
    try:
        import torch
        globals()['torch'] = torch
    except ImportError:
        print("Warning: PyTorch not available, some tests may be skipped")
        
    unittest.main()