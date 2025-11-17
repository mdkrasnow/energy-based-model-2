#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase1_statistical_viability import (
    BATCH_SIZE, LEARNING_RATE, TRAIN_ITERATIONS, DIFFUSION_STEPS, RANK, TASKS
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and constants in phase1_statistical_viability.py"""
    
    def test_constants_are_defined(self):
        """Test that all required constants are properly defined"""
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
    
    def test_constants_values(self):
        """Test that constants have expected values from the paper"""
        # Values from Appendix A of the paper
        self.assertEqual(BATCH_SIZE, 2048)
        self.assertEqual(LEARNING_RATE, 1e-4)
        self.assertEqual(TRAIN_ITERATIONS, 1000)
        self.assertEqual(DIFFUSION_STEPS, 10)
        self.assertEqual(RANK, 20)


class TestHelperMethods(unittest.TestCase):
    """Test helper methods in the classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_sample_batch_error_handling(self):
        """Test _sample_batch handles errors gracefully"""
        from phase1_statistical_viability import ExperimentRunner
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Mock dataset that raises an error
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(side_effect=RuntimeError("Dataset error"))
        
        with self.assertRaises(RuntimeError):
            runner._sample_batch(mock_dataset, 10)
    
    def test_get_result_dir_edge_cases(self):
        """Test get_result_dir with edge cases"""
        from phase1_statistical_viability import ExperimentRunner
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Test with empty string dataset
        result_dir = runner.get_result_dir('', 'baseline')
        self.assertIn('ds_', result_dir)
        
        # Test with special characters (though not recommended)
        result_dir = runner.get_result_dir('test-dataset_123', 'anm', adv_steps=100)
        self.assertIn('test-dataset_123', result_dir)
        self.assertIn('anm_steps100', result_dir)
    
    def test_parse_mse_output_edge_cases(self):
        """Test MSE parsing with various output formats"""
        from phase1_statistical_viability import ExperimentRunner
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Test with multiple MSE values (should get the last one)
        output = """
        Initial mse  0.9999
        Intermediate results
        mse  0.5555  
        Final evaluation
        mse  0.1234
        """
        mse = runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.1234, places=4)
        
        # Test with scientific notation
        output = "mse  1.234e-4"
        mse = runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 1.234e-4, places=6)
        
        # Test with malformed lines
        output = """
        mse
        mse  not_a_number
        mse  0.1234  extra_text
        """
        mse = runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.1234, places=4)
    
    def test_energy_summary_with_edge_cases(self):
        """Test energy summary printing with edge cases"""
        from phase1_statistical_viability import ExperimentRunner
        runner = ExperimentRunner(base_dir=self.test_dir)
        
        # Test with empty energy lists
        energies = {
            'clean': [],
            'ired_standard': [],
            'anm_adversarial': [],
            'gaussian_noise': []
        }
        
        # Should handle empty lists gracefully
        with patch('builtins.print'):
            runner._print_energy_summary(energies, 'anm')
        
        # Test with NaN values
        energies = {
            'clean': [1.0, float('nan'), 1.1],
            'ired_standard': [2.0, 2.1, float('inf')],
            'anm_adversarial': [2.5, 2.6, 2.4],
            'gaussian_noise': [1.5, 1.6, 1.4]
        }
        
        # Should handle special float values
        with patch('builtins.print'):
            runner._print_energy_summary(energies, 'anm')


class TestDataStructures(unittest.TestCase):
    """Test data structure handling"""
    
    def test_diagnostic_results_structure(self):
        """Test that diagnostic results follow expected structure"""
        from phase1_statistical_viability import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Test that diagnostic_results is a defaultdict
        self.assertIsInstance(runner.diagnostic_results, dict)
        
        # Test that we can add nested structures
        runner.diagnostic_results['test_key']['energies'] = {
            'clean': [1.0, 1.1],
            'corrupted': [2.0, 2.1]
        }
        
        self.assertEqual(len(runner.diagnostic_results['test_key']['energies']['clean']), 2)
    
    def test_sweep_results_structure(self):
        """Test sweep results structure"""
        from phase1_statistical_viability import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Test initial state
        self.assertEqual(runner.sweep_results, [])
        
        # Test adding sweep result
        test_result = {
            'config': 'test_config',
            'adv_steps': 10,
            'energy_gap_percent': 5.0,
            'mse_same': 0.1,
            'mse_harder': 0.2
        }
        
        runner.sweep_results.append(test_result)
        self.assertEqual(len(runner.sweep_results), 1)
        self.assertEqual(runner.sweep_results[0]['config'], 'test_config')


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout the module"""
    
    def test_model_loading_with_missing_files(self):
        """Test model loading when checkpoint files are missing"""
        from phase1_statistical_viability import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Test with non-existent model directory
        result = runner.load_model_for_diagnostics('nonexistent_model_dir')
        self.assertIsNone(result)
    
    @patch('subprocess.Popen')
    def test_training_failure_handling(self, mock_popen):
        """Test handling of training failures"""
        from phase1_statistical_viability import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Mock failed subprocess
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Training started\n",
            "ERROR: Training failed\n",
            ""
        ]
        mock_process.wait.return_value = 1  # Non-zero exit code
        mock_popen.return_value = mock_process
        
        with patch('os.path.exists', return_value=False):
            success = runner.train_model('addition', 'baseline')
        
        self.assertFalse(success)
    
    @patch('subprocess.Popen')
    def test_evaluation_failure_handling(self, mock_popen):
        """Test handling of evaluation failures"""
        from phase1_statistical_viability import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Mock failed evaluation
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = ["Evaluation failed\n", ""]
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process
        
        with patch('os.path.exists', return_value=True):
            mse = runner.evaluate_model('addition', 'baseline')
        
        self.assertIsNone(mse)


if __name__ == '__main__':
    unittest.main()