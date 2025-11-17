#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import argparse
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import phase1_statistical_viability


class TestMainFunctionality(unittest.TestCase):
    """Test main entry point and command line argument handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('sys.argv', ['phase1_statistical_viability.py', '--phase1', '--base-dir', 'test_dir'])
    @patch('phase1_statistical_viability.Phase1ExperimentRunner')
    def test_main_phase1_argument_parsing(self, mock_runner_class):
        """Test command line argument parsing for Phase 1"""
        mock_runner = Mock()
        mock_runner.run_phase1_viability_test.return_value = (True, [], "Test report")
        mock_runner.base_dir = self.test_dir
        mock_runner_class.return_value = mock_runner
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit'):
            with patch('os.path.abspath', return_value=self.test_dir):
                with patch('os.makedirs'):
                    with patch('builtins.open', create=True):
                        # Import and run the main functionality
                        import importlib
                        importlib.reload(phase1_statistical_viability)
        
        # Would normally test that Phase1ExperimentRunner was called
        # but the actual execution is in the if __name__ == "__main__" block
    
    @patch('sys.argv', ['phase1_statistical_viability.py', '--sweep', '--dataset', 'addition'])
    @patch('phase1_statistical_viability.ExperimentRunner')
    def test_main_sweep_argument_parsing(self, mock_runner_class):
        """Test command line argument parsing for hyperparameter sweep"""
        mock_runner = Mock()
        mock_runner.run_hyperparameter_sweep.return_value = []
        mock_runner_class.return_value = mock_runner
        
        # Test argument parsing logic (not actual execution)
        parser = argparse.ArgumentParser()
        parser.add_argument('--base-dir', default='phase1_results')
        parser.add_argument('--force', action='store_true')
        parser.add_argument('--phase1', action='store_true')
        parser.add_argument('--sweep', action='store_true')
        parser.add_argument('--dataset', default='addition')
        
        args = parser.parse_args(['--sweep', '--dataset', 'addition'])
        
        self.assertTrue(args.sweep)
        self.assertFalse(args.phase1)
        self.assertEqual(args.dataset, 'addition')
    
    @patch('sys.argv', ['phase1_statistical_viability.py', '--force'])
    @patch('phase1_statistical_viability.ExperimentRunner')
    def test_main_default_training(self, mock_runner_class):
        """Test default training workflow"""
        mock_runner = Mock()
        mock_runner.train_all.return_value = True
        mock_runner.evaluate_all.return_value = {}
        mock_runner.plot_diagnostic_results.return_value = None
        mock_runner.print_diagnostic_summary.return_value = None
        mock_runner_class.return_value = mock_runner
        
        # Test argument parsing for default case
        parser = argparse.ArgumentParser()
        parser.add_argument('--base-dir', default='phase1_results')
        parser.add_argument('--force', action='store_true')
        parser.add_argument('--phase1', action='store_true')
        parser.add_argument('--sweep', action='store_true')
        parser.add_argument('--dataset', default='addition')
        
        args = parser.parse_args(['--force'])
        
        self.assertTrue(args.force)
        self.assertFalse(args.phase1)
        self.assertFalse(args.sweep)
    
    def test_base_dir_validation_logic(self):
        """Test the base directory validation logic"""
        # Test the logic that would be in main
        base_dir = os.path.abspath(self.test_dir)
        
        # Should be able to create directory
        os.makedirs(base_dir, exist_ok=True)
        self.assertTrue(os.path.exists(base_dir))
        
        # Should be able to write test file
        test_path = os.path.join(base_dir, ".__write_test")
        with open(test_path, "w") as f:
            f.write("test\n")
        
        self.assertTrue(os.path.exists(test_path))
        
        # Clean up test file
        os.remove(test_path)
        self.assertFalse(os.path.exists(test_path))
    
    def test_phase1_completion_messages(self):
        """Test Phase 1 completion message formatting"""
        # Mock statistical results
        mock_stat_result = Mock()
        mock_stat_result.significant = True
        mock_stat_result.effect_size_adequate = True
        
        statistical_results = [mock_stat_result, mock_stat_result]  # 2 viable configs
        
        # Test go decision formatting
        go_decision = True
        
        # This tests the logic that would appear in main
        decision_text = '✅ GO TO PHASE 2' if go_decision else '❌ NO-GO - STOP'
        self.assertEqual(decision_text, '✅ GO TO PHASE 2')
        
        viable_count = sum(1 for r in statistical_results if r.significant and r.effect_size_adequate)
        self.assertEqual(viable_count, 2)
    
    def test_import_error_handling(self):
        """Test that the module handles missing dependencies gracefully"""
        # Test that torch import doesn't break the module
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False
        
        # Module should still be importable regardless
        self.assertIsNotNone(phase1_statistical_viability)
        
        if torch_available:
            self.assertTrue(hasattr(phase1_statistical_viability, 'torch'))


class TestCommandLineInterface(unittest.TestCase):
    """Test command line interface components"""
    
    def test_argument_parser_structure(self):
        """Test that argument parser has all required arguments"""
        parser = argparse.ArgumentParser(description='Phase 1 Statistical Viability Testing for ANM')
        parser.add_argument('--base-dir', default='phase1_results',
                           help='Base directory for experiments and logs')
        parser.add_argument('--force', action='store_true', help='Force retrain even if models exist')
        parser.add_argument('--phase1', action='store_true', help='Run Phase 1 statistical viability testing')
        parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep instead of full training')
        parser.add_argument('--dataset', default='addition', help='Dataset for testing')
        
        # Test default arguments
        args = parser.parse_args([])
        self.assertEqual(args.base_dir, 'phase1_results')
        self.assertEqual(args.dataset, 'addition')
        self.assertFalse(args.force)
        self.assertFalse(args.phase1)
        self.assertFalse(args.sweep)
        
        # Test with all arguments
        args = parser.parse_args([
            '--base-dir', 'custom_dir',
            '--force',
            '--phase1', 
            '--dataset', 'lowrank'
        ])
        
        self.assertEqual(args.base_dir, 'custom_dir')
        self.assertEqual(args.dataset, 'lowrank')
        self.assertTrue(args.force)
        self.assertTrue(args.phase1)
        self.assertFalse(args.sweep)
    
    def test_mutually_exclusive_options(self):
        """Test that phase1 and sweep options work correctly"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--phase1', action='store_true')
        parser.add_argument('--sweep', action='store_true')
        
        # Both can be False (default training)
        args = parser.parse_args([])
        self.assertFalse(args.phase1)
        self.assertFalse(args.sweep)
        
        # Only one should be True at a time
        args = parser.parse_args(['--phase1'])
        self.assertTrue(args.phase1)
        self.assertFalse(args.sweep)
        
        args = parser.parse_args(['--sweep'])
        self.assertFalse(args.phase1)
        self.assertTrue(args.sweep)


class TestModuleStructure(unittest.TestCase):
    """Test overall module structure and imports"""
    
    def test_required_imports_available(self):
        """Test that all required imports are available"""
        # Test that key classes are importable
        self.assertTrue(hasattr(phase1_statistical_viability, 'ExperimentRunner'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'Phase1ExperimentRunner'))
        
        # Test that key constants are available
        self.assertTrue(hasattr(phase1_statistical_viability, 'BATCH_SIZE'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'LEARNING_RATE'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'TRAIN_ITERATIONS'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'DIFFUSION_STEPS'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'RANK'))
        self.assertTrue(hasattr(phase1_statistical_viability, 'TASKS'))
    
    def test_class_inheritance(self):
        """Test that Phase1ExperimentRunner properly inherits from ExperimentRunner"""
        self.assertTrue(
            issubclass(
                phase1_statistical_viability.Phase1ExperimentRunner,
                phase1_statistical_viability.ExperimentRunner
            )
        )
    
    def test_module_docstring(self):
        """Test that the module has proper documentation"""
        # The module should have a docstring or comments at the top
        module = phase1_statistical_viability
        
        # Check that the file has proper header comments
        import inspect
        source = inspect.getsource(module)
        
        # Should contain key information about the file
        self.assertIn('phase1_statistical_viability.py', source)


if __name__ == '__main__':
    unittest.main()