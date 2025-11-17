#!/usr/bin/env python3
"""
Comprehensive test suite for phase1_statistical_viability.py
This file contains simplified, reliable tests that verify core functionality.
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
import time
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase1_statistical_viability import (
    ExperimentRunner, 
    Phase1ExperimentRunner, 
    BATCH_SIZE, 
    LEARNING_RATE, 
    TRAIN_ITERATIONS, 
    DIFFUSION_STEPS, 
    RANK,
    TASKS
)


class TestConstants(unittest.TestCase):
    """Test that all constants are properly defined"""
    
    def test_constants_exist(self):
        """Test that all required constants exist and have reasonable values"""
        self.assertIsInstance(BATCH_SIZE, int)
        self.assertGreater(BATCH_SIZE, 0)
        
        self.assertIsInstance(LEARNING_RATE, float)
        self.assertGreater(LEARNING_RATE, 0)
        self.assertLess(LEARNING_RATE, 1)
        
        self.assertIsInstance(TRAIN_ITERATIONS, int)
        self.assertGreater(TRAIN_ITERATIONS, 0)
        
        self.assertIsInstance(DIFFUSION_STEPS, int)
        self.assertGreater(DIFFUSION_STEPS, 0)
        
        self.assertIsInstance(RANK, int)
        self.assertGreater(RANK, 0)
        
        self.assertIsInstance(TASKS, list)
        self.assertIn('addition', TASKS)


class TestExperimentRunnerBasic(unittest.TestCase):
    """Basic tests for ExperimentRunner that don't require complex mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.runner = ExperimentRunner(base_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test ExperimentRunner initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.runner.base_dir, Path(self.test_dir))
        self.assertEqual(self.runner.results, {})
        self.assertIsInstance(self.runner.diagnostic_results, dict)
        self.assertEqual(self.runner.sweep_results, [])
    
    def test_get_result_dir(self):
        """Test get_result_dir method"""
        # Test baseline model
        result_dir = self.runner.get_result_dir('addition', 'baseline')
        expected = f'results/ds_addition/model_mlp_diffsteps_{DIFFUSION_STEPS}'
        self.assertEqual(result_dir, expected)
        
        # Test ANM model with specific steps
        result_dir = self.runner.get_result_dir('addition', 'anm', adv_steps=10)
        expected = f'results/ds_addition/model_mlp_diffsteps_{DIFFUSION_STEPS}_anm_steps10'
        self.assertEqual(result_dir, expected)
        
        # Test with seed
        result_dir = self.runner.get_result_dir('addition', 'baseline', seed=42)
        expected = f'results/ds_addition/model_mlp_diffsteps_{DIFFUSION_STEPS}_seed42'
        self.assertEqual(result_dir, expected)
    
    def test_parse_mse_simple(self):
        """Test MSE parsing with correct format"""
        # Test with exact format expected by the parser
        output = "mse  0.1234"
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertEqual(mse, 0.1234)
        
        # Test with mse_error format
        output = "mse_error  0.5678"
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertEqual(mse, 0.5678)
        
        # Test with no MSE
        output = "No MSE in this output"
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertIsNone(mse)
    
    def test_save_results(self):
        """Test results saving functionality"""
        # Set up test results
        test_results = {
            'addition': {
                'baseline': {
                    'same_difficulty': 0.1234,
                    'harder_difficulty': 0.5678
                }
            }
        }
        self.runner.results = test_results
        
        # Save results
        self.runner._save_results()
        
        # Check file exists
        results_file = Path(self.test_dir) / 'continuous_results_with_anm_diagnostics.json'
        self.assertTrue(results_file.exists())
        
        # Verify content
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('metadata', data)
        self.assertIn('results', data)
        self.assertEqual(data['results'], test_results)
    
    def test_energy_summary(self):
        """Test energy summary printing"""
        energies = {
            'clean': [1.0, 1.1, 0.9],
            'ired_standard': [2.0, 2.1, 1.9],
            'anm_adversarial': [2.5, 2.6, 2.4],
            'gaussian_noise': [1.5, 1.6, 1.4]
        }
        
        # Should not raise exceptions
        with patch('builtins.print'):
            self.runner._print_energy_summary(energies, 'anm')


class TestPhase1RunnerBasic(unittest.TestCase):
    """Basic tests for Phase1ExperimentRunner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.runner = Phase1ExperimentRunner(base_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test Phase1ExperimentRunner initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.runner.base_dir, Path(self.test_dir))
        self.assertIsNotNone(self.runner.statistical_analyzer)
        self.assertEqual(self.runner.phase1_results, [])
        self.assertTrue(self.runner.csv_log_path.exists())
    
    def test_csv_creation(self):
        """Test that CSV file is created with proper headers"""
        self.assertTrue(self.runner.csv_log_path.exists())
        
        with open(self.runner.csv_log_path, 'r') as f:
            header = f.readline().strip()
        
        # Check for key headers
        self.assertIn('experiment_id', header)
        self.assertIn('config_name', header)
        self.assertIn('mse_same_difficulty', header)
        self.assertIn('mse_harder_difficulty', header)
        self.assertIn('success', header)
    
    def test_csv_logging(self):
        """Test CSV logging functionality"""
        # Create test data
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
        
        # Log experiment
        self.runner._log_experiment_to_csv(experiment_spec, results, 12.34, True)
        
        # Verify logging
        with open(self.runner.csv_log_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)  # Header + data
        
        data_line = lines[1].strip().split(',')
        self.assertEqual(data_line[1], 'test_001')  # experiment_id
        self.assertEqual(data_line[2], 'test_config')  # config_name
    
    def test_inheritance(self):
        """Test that Phase1ExperimentRunner inherits properly"""
        # Should have parent methods
        self.assertTrue(hasattr(self.runner, 'get_result_dir'))
        self.assertTrue(hasattr(self.runner, '_save_results'))
        
        # Should have Phase 1 specific methods
        self.assertTrue(hasattr(self.runner, 'run_phase1_viability_test'))
        self.assertTrue(hasattr(self.runner, 'train_model_with_phase1_config'))


class TestMockedTraining(unittest.TestCase):
    """Test training workflows with mocked subprocess calls"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.runner = ExperimentRunner(base_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('subprocess.Popen')
    @patch('os.path.exists')
    def test_train_model_success(self, mock_exists, mock_popen):
        """Test successful model training"""
        mock_exists.return_value = False  # Force training
        
        # Mock successful training process
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Training started\n",
            "Epoch 1/1000\n",
            "Training completed\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        with patch.object(self.runner, 'run_energy_diagnostics', return_value=None):
            success = self.runner.train_model('addition', 'baseline')
        
        self.assertTrue(success)
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('os.path.exists')
    def test_evaluate_model_success(self, mock_exists, mock_popen):
        """Test successful model evaluation"""
        mock_exists.return_value = True  # Model exists
        
        # Mock successful evaluation
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Loading model\n",
            "mse  0.0123\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        mse = self.runner.evaluate_model('addition', 'baseline')
        
        self.assertEqual(mse, 0.0123)


class TestDataStructures(unittest.TestCase):
    """Test data structure handling"""
    
    def test_diagnostic_results(self):
        """Test diagnostic results structure"""
        runner = ExperimentRunner()
        
        # Should be able to store diagnostic data
        runner.diagnostic_results['test_key'] = {
            'energies': {'clean': [1.0, 1.1], 'corrupted': [2.0, 2.1]}
        }
        
        self.assertIn('test_key', runner.diagnostic_results)
        self.assertEqual(len(runner.diagnostic_results['test_key']['energies']['clean']), 2)
    
    def test_sweep_results(self):
        """Test sweep results structure"""
        runner = ExperimentRunner()
        
        test_result = {
            'config': 'test_config',
            'adv_steps': 10,
            'energy_gap_percent': 5.0
        }
        
        runner.sweep_results.append(test_result)
        
        self.assertEqual(len(runner.sweep_results), 1)
        self.assertEqual(runner.sweep_results[0]['config'], 'test_config')


class TestErrorHandling(unittest.TestCase):
    """Test error handling"""
    
    def test_model_loading_missing_file(self):
        """Test model loading with missing checkpoint"""
        runner = ExperimentRunner()
        
        result = runner.load_model_for_diagnostics('nonexistent_directory')
        self.assertIsNone(result)
    
    @patch('subprocess.Popen')
    def test_training_failure(self, mock_popen):
        """Test handling of training failure"""
        runner = ExperimentRunner()
        
        # Mock failed training
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Training failed\n",
            ""
        ]
        mock_process.wait.return_value = 1  # Failed
        mock_popen.return_value = mock_process
        
        with patch('os.path.exists', return_value=False):
            success = runner.train_model('addition', 'baseline')
        
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()