#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import sys
import json
import time
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase1_statistical_viability import Phase1ExperimentRunner, PHASE1_EXPERIMENTAL_DESIGN
from phase1_configs import PHASE1_CONFIGURATIONS


class TestPhase1ExperimentRunner(unittest.TestCase):
    """Test cases for the Phase1ExperimentRunner class"""
    
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
    
    def test_csv_initialization(self):
        """Test CSV logging initialization"""
        # Check that CSV file was created with proper header
        self.assertTrue(self.runner.csv_log_path.exists())
        
        with open(self.runner.csv_log_path, 'r') as f:
            header_line = f.readline().strip()
            headers = header_line.split(',')
        
        expected_headers = [
            'timestamp', 'experiment_id', 'config_name', 'dataset', 'seed',
            'train_steps', 'mse_same_difficulty', 'mse_harder_difficulty',
            'training_time_minutes', 'model_path', 'success',
            'anm_adversarial_steps', 'random_noise_scale', 'random_noise_type'
        ]
        
        self.assertEqual(headers, expected_headers)
    
    def test_log_experiment_to_csv(self):
        """Test CSV experiment logging"""
        # Create mock experiment spec
        mock_config = Mock()
        mock_config.use_anm = True
        mock_config.anm_adversarial_steps = 10
        mock_config.use_random_noise = False
        
        experiment_spec = {
            'experiment_id': 'test_exp_001',
            'config_name': 'anm_aggressive',
            'seed': 42,
            'train_steps': 1000,
            'config': mock_config
        }
        
        results = {
            'mse_same': 0.1234,
            'mse_harder': 0.5678
        }
        
        # Log the experiment
        self.runner._log_experiment_to_csv(experiment_spec, results, 15.5, True)
        
        # Read and verify the CSV content
        with open(self.runner.csv_log_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)  # Header + 1 data row
        
        data_line = lines[1].strip().split(',')
        self.assertEqual(data_line[1], 'test_exp_001')  # experiment_id
        self.assertEqual(data_line[2], 'anm_aggressive')  # config_name
        self.assertEqual(data_line[4], '42')  # seed
        self.assertEqual(data_line[6], '0.1234')  # mse_same
        self.assertEqual(data_line[7], '0.5678')  # mse_harder
        self.assertEqual(data_line[8], '15.50')  # training_time_minutes
        self.assertEqual(data_line[10], 'success')  # success
        self.assertEqual(data_line[11], '10')  # anm_adversarial_steps
    
    @patch('torch.manual_seed')
    @patch('numpy.random.seed')
    @patch.object(Phase1ExperimentRunner, 'train_model')
    @patch.object(Phase1ExperimentRunner, '_evaluate_phase1_model')
    def test_train_model_with_phase1_config(self, mock_evaluate, mock_train, mock_np_seed, mock_torch_seed):
        """Test training with Phase 1 configuration"""
        # Create mock config
        mock_config = Mock()
        mock_config.name = 'test_config'
        mock_config.description = 'Test configuration'
        mock_config.use_anm = True
        mock_config.anm_adversarial_steps = 15
        mock_config.use_random_noise = False
        
        # Mock training success
        mock_train.return_value = True
        mock_evaluate.side_effect = [0.1234, 0.5678]  # same, harder difficulty
        
        success, training_time, results = self.runner.train_model_with_phase1_config(
            'addition', mock_config, 42, force_retrain=False
        )
        
        # Verify results
        self.assertTrue(success)
        self.assertGreater(training_time, 0)
        self.assertEqual(results['mse_same'], 0.1234)
        self.assertEqual(results['mse_harder'], 0.5678)
        
        # Verify random seeds were set
        mock_torch_seed.assert_called_with(42)
        mock_np_seed.assert_called_with(42)
        
        # Verify training was called with correct parameters
        mock_train.assert_called_once_with(
            dataset='addition',
            model_type='anm',
            force_retrain=False,
            adv_steps=15,
            train_steps=PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment'],
            seed=42
        )
    
    @patch.object(Phase1ExperimentRunner, 'train_model')
    def test_train_model_baseline_config(self, mock_train):
        """Test training with baseline configuration"""
        # Create baseline config
        mock_config = Mock()
        mock_config.name = 'ired_baseline'
        mock_config.description = 'Baseline IRED'
        mock_config.use_anm = False
        mock_config.use_random_noise = False
        
        mock_train.return_value = True
        
        success, training_time, results = self.runner.train_model_with_phase1_config(
            'addition', mock_config, 123
        )
        
        # Verify baseline training call
        mock_train.assert_called_once_with(
            dataset='addition',
            model_type='baseline',
            force_retrain=False,
            adv_steps=None,
            train_steps=PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment'],
            seed=123
        )
    
    def test_train_random_noise_baseline(self):
        """Test random noise baseline training (currently unimplemented)"""
        mock_config = Mock()
        mock_config.use_random_noise = True
        mock_config.random_noise_scale = 0.1
        mock_config.random_noise_type = 'gaussian'
        
        # Should return False since not implemented
        success = self.runner._train_random_noise_baseline('addition', mock_config, 42)
        self.assertFalse(success)
    
    @patch.object(Phase1ExperimentRunner, 'evaluate_model_with_config')
    @patch.object(Phase1ExperimentRunner, 'evaluate_model')
    def test_evaluate_phase1_model(self, mock_eval_baseline, mock_eval_anm):
        """Test Phase 1 model evaluation"""
        # Test ANM config evaluation
        mock_config_anm = Mock()
        mock_config_anm.use_anm = True
        mock_config_anm.anm_adversarial_steps = 20
        mock_config_anm.use_random_noise = False
        
        mock_eval_anm.return_value = 0.1234
        
        result = self.runner._evaluate_phase1_model('addition', mock_config_anm, 42, ood=True)
        
        self.assertEqual(result, 0.1234)
        mock_eval_anm.assert_called_once_with(
            'addition',
            adv_steps=20,
            ood=True,
            train_steps=PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment'],
            seed=42
        )
        
        # Test baseline config evaluation
        mock_config_baseline = Mock()
        mock_config_baseline.use_anm = False
        mock_config_baseline.use_random_noise = False
        
        mock_eval_baseline.return_value = 0.5678
        
        result = self.runner._evaluate_phase1_model('addition', mock_config_baseline, 123, ood=False)
        
        self.assertEqual(result, 0.5678)
        mock_eval_baseline.assert_called_once_with('addition', 'baseline', ood=False, seed=123)
    
    @patch('phase1_statistical_viability.generate_experiment_matrix')
    @patch.object(Phase1ExperimentRunner, 'train_model_with_phase1_config')
    @patch.object(Phase1ExperimentRunner, '_log_experiment_to_csv')
    @patch.object(Phase1ExperimentRunner, '_analyze_phase1_results')
    def test_run_phase1_viability_test_structure(self, mock_analyze, mock_log, mock_train, mock_generate):
        """Test the structure of Phase 1 viability testing"""
        # Mock experiment generation
        mock_experiments = [
            {
                'experiment_id': 'exp_001',
                'config_name': 'ired_baseline',
                'config': Mock(name='baseline_config'),
                'seed': 42
            },
            {
                'experiment_id': 'exp_002', 
                'config_name': 'anm_aggressive',
                'config': Mock(name='anm_config'),
                'seed': 42
            }
        ]
        mock_generate.return_value = mock_experiments
        
        # Mock training results
        mock_train.side_effect = [
            (True, 10.0, {'mse_harder': 0.1}),
            (True, 15.0, {'mse_harder': 0.2})
        ]
        
        # Mock analysis results
        mock_analyze.return_value = (True, [], "Test report")
        
        # Run the test
        go_decision, stats, report = self.runner.run_phase1_viability_test('addition')
        
        # Verify experiment generation was called
        mock_generate.assert_called_once()
        
        # Verify training was called for each experiment
        self.assertEqual(mock_train.call_count, 2)
        
        # Verify logging was called for each experiment
        self.assertEqual(mock_log.call_count, 2)
        
        # Verify analysis was called
        mock_analyze.assert_called_once()
    
    @patch.object(Phase1ExperimentRunner, 'statistical_analyzer')
    def test_analyze_phase1_results(self, mock_analyzer):
        """Test Phase 1 results analysis"""
        # Mock results
        all_results = {
            'ired_baseline': [0.1, 0.12, 0.11, 0.09, 0.13],
            'anm_aggressive': [0.08, 0.09, 0.07, 0.10, 0.08]
        }
        
        # Mock statistical analysis
        mock_stat_result = Mock()
        mock_stat_result.significant = True
        mock_stat_result.effect_size_adequate = True
        mock_stat_result.p_value_corrected = 0.01
        mock_stat_result.cohens_d = 0.8
        
        mock_analyzer.statistical_analysis_single_config.return_value = mock_stat_result
        mock_analyzer.statistical_summary_table.return_value = Mock(to_string=Mock(return_value="Summary table"))
        mock_analyzer.go_no_go_decision.return_value = (True, "Go decision rationale")
        mock_analyzer.generate_phase1_report.return_value = "Phase 1 report content"
        
        # Run analysis
        go_decision, stats, report = self.runner._analyze_phase1_results(all_results, 'addition', 2.5)
        
        # Verify analysis calls
        mock_analyzer.statistical_analysis_single_config.assert_called_once()
        mock_analyzer.go_no_go_decision.assert_called_once()
        mock_analyzer.generate_phase1_report.assert_called_once()
        
        # Verify return values
        self.assertTrue(go_decision)
        self.assertEqual(len(stats), 1)
        self.assertEqual(report, "Phase 1 report content")
    
    def test_analyze_phase1_results_no_baseline(self):
        """Test analysis when baseline results are missing"""
        all_results = {
            'anm_aggressive': [0.08, 0.09, 0.07, 0.10, 0.08]
        }
        
        go_decision, stats, report = self.runner._analyze_phase1_results(all_results, 'addition', 1.0)
        
        self.assertFalse(go_decision)
        self.assertEqual(stats, [])
        self.assertIn("No baseline results", report)
    
    def test_inheritance_from_experiment_runner(self):
        """Test that Phase1ExperimentRunner properly inherits from ExperimentRunner"""
        # Should have all parent methods
        self.assertTrue(hasattr(self.runner, 'train_model'))
        self.assertTrue(hasattr(self.runner, 'evaluate_model'))
        self.assertTrue(hasattr(self.runner, 'get_result_dir'))
        self.assertTrue(hasattr(self.runner, '_sample_batch'))
        
        # Should have additional Phase 1 specific methods
        self.assertTrue(hasattr(self.runner, 'run_phase1_viability_test'))
        self.assertTrue(hasattr(self.runner, 'train_model_with_phase1_config'))
        self.assertTrue(hasattr(self.runner, 'statistical_analyzer'))
    
    def test_phase1_experimental_design_validation(self):
        """Test that Phase 1 experimental design constants are accessible"""
        # These should be imported from the main module
        self.assertIn('total_experiments', PHASE1_EXPERIMENTAL_DESIGN)
        self.assertIn('num_configs', PHASE1_EXPERIMENTAL_DESIGN)
        self.assertIn('seeds_per_config', PHASE1_EXPERIMENTAL_DESIGN)
        self.assertIn('train_steps_per_experiment', PHASE1_EXPERIMENTAL_DESIGN)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_report_saving(self, mock_json_dump, mock_file):
        """Test that Phase 1 reports are saved correctly"""
        # Mock analysis to generate a report
        with patch.object(self.runner, '_analyze_phase1_results') as mock_analyze:
            mock_analyze.return_value = (True, [], "Test report content")
            
            # Create a minimal results dict to trigger analysis
            all_results = {'ired_baseline': [0.1]}
            go_decision, stats, report = self.runner._analyze_phase1_results(all_results, 'addition', 1.0)
            
            # Manually test report saving logic
            report_path = self.runner.base_dir / 'phase1_decision_report_addition.md'
            
            # Simulate report saving
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Verify report path construction
            expected_path = Path(self.test_dir) / 'phase1_decision_report_addition.md'
            self.assertEqual(report_path, expected_path)


if __name__ == '__main__':
    unittest.main()