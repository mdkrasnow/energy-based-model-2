#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch for tests that don't require actual tensors
    torch = Mock()
    torch.randn = Mock(return_value="mock_tensor")
    torch.randint = Mock(return_value="mock_tensor") 
    torch.tensor = Mock(return_value="mock_tensor")

from phase1_statistical_viability import ExperimentRunner, BATCH_SIZE, LEARNING_RATE, TRAIN_ITERATIONS, DIFFUSION_STEPS, RANK


class TestExperimentRunner(unittest.TestCase):
    """Test cases for the ExperimentRunner class"""
    
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
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_sample_batch(self):
        """Test _sample_batch method"""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=[
            (np.array([1, 2, 3]), np.array([4, 5, 6])),
            (np.array([7, 8, 9]), np.array([10, 11, 12]))
        ])
        
        batch = self.runner._sample_batch(mock_dataset, 2)
        
        self.assertIn('x', batch)
        self.assertIn('y', batch)
        self.assertEqual(batch['x'].shape[0], 2)
        self.assertEqual(batch['y'].shape[0], 2)
        self.assertTrue(torch.is_tensor(batch['x']))
        self.assertTrue(torch.is_tensor(batch['y']))
    
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
    
    def test_parse_mse_from_output(self):
        """Test _parse_mse_from_output method"""
        # Test with table format
        output = """
        Training completed
        mse  0.1234
        Other output
        """
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.1234, places=4)
        
        # Test with mse_error format
        output = """
        Training results:
        mse_error  0.5678
        """
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.5678, places=4)
        
        # Test with no MSE found
        output = "No MSE in this output"
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertIsNone(mse)
    
    def test_save_results(self):
        """Test results saving"""
        # Set up some test results
        self.runner.results = {
            'addition': {
                'baseline': {
                    'same_difficulty': 0.1,
                    'harder_difficulty': 0.2
                }
            }
        }
        
        self.runner._save_results()
        
        # Check that results file was created
        results_file = Path(self.test_dir) / 'continuous_results_with_anm_diagnostics.json'
        self.assertTrue(results_file.exists())
        
        # Verify content
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('metadata', data)
        self.assertIn('results', data)
        self.assertEqual(data['results'], self.runner.results)
        self.assertEqual(data['metadata']['batch_size'], BATCH_SIZE)
    
    def test_print_energy_summary(self):
        """Test energy summary printing"""
        energies = {
            'clean': [1.0, 1.1, 0.9],
            'ired_standard': [2.0, 2.1, 1.9],
            'anm_adversarial': [2.5, 2.6, 2.4],
            'gaussian_noise': [1.5, 1.6, 1.4]
        }
        
        # Should not raise any exceptions
        with patch('builtins.print'):
            self.runner._print_energy_summary(energies, 'anm')
    
    def test_get_result_dir_edge_cases(self):
        """Test get_result_dir with edge cases"""
        # Test with empty string dataset
        result_dir = self.runner.get_result_dir('', 'baseline')
        self.assertIn('ds_', result_dir)
        
        # Test with special characters
        result_dir = self.runner.get_result_dir('test-dataset_123', 'anm', adv_steps=100)
        self.assertIn('test-dataset_123', result_dir)
        self.assertIn('anm_steps100', result_dir)
    
    def test_parse_mse_edge_cases(self):
        """Test MSE parsing with various output formats"""
        # Test with multiple MSE values (should get the last one)
        output = """
        Initial mse  0.9999
        Intermediate results
        mse  0.5555  
        Final evaluation
        mse  0.1234
        """
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.1234, places=4)
        
        # Test with scientific notation
        output = "mse  1.234e-4"
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 1.234e-4, places=6)
        
        # Test with malformed lines
        output = """
        mse
        mse  not_a_number
        mse  0.1234  extra_text
        """
        mse = self.runner._parse_mse_from_output(output, "")
        self.assertAlmostEqual(mse, 0.1234, places=4)


if __name__ == '__main__':
    unittest.main()