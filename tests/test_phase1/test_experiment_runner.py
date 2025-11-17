#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
        self.assertIsInstance(batch['x'], torch.Tensor)
        self.assertIsInstance(batch['y'], torch.Tensor)
    
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
    
    @patch('subprocess.Popen')
    def test_train_model_baseline(self, mock_popen):
        """Test training baseline model"""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Training started\n",
            "Epoch 1\n", 
            "Training completed\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Mock model existence check
        with patch('os.path.exists', return_value=False):
            with patch.object(self.runner, 'run_energy_diagnostics', return_value=None):
                success = self.runner.train_model('addition', 'baseline')
        
        self.assertTrue(success)
        mock_popen.assert_called_once()
        
        # Verify command structure
        call_args = mock_popen.call_args[0][0]
        self.assertIn('python', call_args)
        self.assertIn('train.py', call_args)
        self.assertIn('--dataset', call_args)
        self.assertIn('addition', call_args)
    
    @patch('subprocess.Popen')
    def test_train_model_anm(self, mock_popen):
        """Test training ANM model"""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Training ANM model\n",
            "Using adversarial corruption\n",
            "Training completed\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Mock model existence check
        with patch('os.path.exists', return_value=False):
            with patch.object(self.runner, 'run_energy_diagnostics', return_value=None):
                success = self.runner.train_model('addition', 'anm', adv_steps=10)
        
        self.assertTrue(success)
        
        # Verify ANM-specific parameters
        call_args = mock_popen.call_args[0][0]
        self.assertIn('--use-anm', call_args)
        self.assertIn('--anm-adversarial-steps', call_args)
        self.assertIn('10', call_args)
    
    @patch('subprocess.Popen')
    def test_evaluate_model(self, mock_popen):
        """Test model evaluation"""
        # Mock subprocess for evaluation
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Loading model\n",
            "Evaluating on test set\n",
            "mse  0.0123\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Mock model existence
        with patch('os.path.exists', return_value=True):
            mse = self.runner.evaluate_model('addition', 'baseline', ood=False)
        
        self.assertAlmostEqual(mse, 0.0123, places=4)
        
        # Verify command structure
        call_args = mock_popen.call_args[0][0]
        self.assertIn('--evaluate', call_args)
        self.assertIn('--load-milestone', call_args)
    
    @patch('subprocess.Popen')
    def test_evaluate_model_ood(self, mock_popen):
        """Test OOD model evaluation"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Evaluating OOD\n",
            "mse  0.0456\n",
            ""
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        with patch('os.path.exists', return_value=True):
            mse = self.runner.evaluate_model('addition', 'baseline', ood=True)
        
        self.assertAlmostEqual(mse, 0.0456, places=4)
        
        # Verify OOD flag
        call_args = mock_popen.call_args[0][0]
        self.assertIn('--ood', call_args)
    
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
    
    @patch('torch.load')
    @patch('os.path.exists')
    def test_load_model_for_diagnostics(self, mock_exists, mock_torch_load):
        """Test model loading for diagnostics"""
        mock_exists.return_value = True
        
        # Mock checkpoint
        mock_checkpoint = {
            'model': {
                'layer.weight': torch.randn(10, 5),
                'layer.bias': torch.randn(10)
            }
        }
        mock_torch_load.return_value = mock_checkpoint
        
        with patch('phase1_statistical_viability.Addition') as mock_dataset:
            mock_dataset.return_value.inp_dim = 5
            mock_dataset.return_value.out_dim = 10
            
            with patch('phase1_statistical_viability.EBM') as mock_ebm:
                with patch('phase1_statistical_viability.DiffusionWrapper') as mock_wrapper:
                    with patch('phase1_statistical_viability.GaussianDiffusion1D') as mock_diffusion:
                        mock_model = Mock()
                        mock_model.state_dict.return_value = {
                            'layer.weight': torch.randn(10, 5),
                            'layer.bias': torch.randn(10)
                        }
                        mock_wrapper.return_value = mock_model
                        
                        result = self.runner.load_model_for_diagnostics('test_model_dir')
                        
                        self.assertIsNotNone(result)
                        self.assertEqual(len(result), 4)  # model, diffusion, device, dataset
    
    def test_simulate_anm_output(self):
        """Test ANM output simulation"""
        # Create mock inputs
        x_clean = torch.randn(4, 10)
        y_clean = torch.randn(4, 15)
        t = torch.randint(0, 10, (4,))
        
        # Mock diffusion object
        mock_diffusion = Mock()
        mock_diffusion.anm_adversarial_steps = 5
        
        with patch('phase1_statistical_viability._adversarial_corruption') as mock_corruption:
            mock_corruption.return_value = torch.randn(4, 15)
            
            result = self.runner._simulate_anm_output(x_clean, y_clean, t, mock_diffusion)
            
            self.assertEqual(result.shape, y_clean.shape)
            mock_corruption.assert_called_once()
    
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
    
    def test_comparative_diagnostics_structure(self):
        """Test structure of comparative diagnostics"""
        # Mock the load_model_for_diagnostics method
        mock_model = Mock()
        mock_diffusion = Mock()
        mock_diffusion.num_timesteps = 10
        mock_device = 'cpu'
        mock_dataset = Mock()
        
        load_result = (mock_model, mock_diffusion, mock_device, mock_dataset)
        
        with patch.object(self.runner, 'load_model_for_diagnostics', return_value=load_result):
            with patch.object(self.runner, '_sample_batch') as mock_sample:
                mock_sample.return_value = {
                    'x': torch.randn(10, 5),
                    'y': torch.randn(10, 3)
                }
                
                with patch.object(self.runner, '_simulate_anm_output') as mock_anm:
                    mock_anm.return_value = torch.randn(10, 3)
                    
                    # Mock energy score calls
                    mock_diffusion.energy_score.return_value = torch.tensor([1.0, 2.0, 3.0])
                    
                    result = self.runner.run_comparative_diagnostics('baseline_dir', 'anm_dir')
                    
                    if result is not None:
                        self.assertIn('energy_ired', result)
                        self.assertIn('energy_anm', result)
                        self.assertIn('distance_ired', result)
                        self.assertIn('distance_anm', result)
                        self.assertIn('energy_ratio', result)


if __name__ == '__main__':
    unittest.main()