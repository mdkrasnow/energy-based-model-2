import os
import os.path as osp

# Prevent numpy over multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from models import EBM, DiffusionWrapper
from models import SudokuEBM, SudokuTransformerEBM, SudokuDenoise, SudokuLatentEBM, AutoencodeModel
from models import GraphEBM, GraphReverse, GNNConvEBM, GNNDiffusionWrapper, GNNConvDiffusionWrapper, GNNConv1DEBMV2, GNNConv1DV2DiffusionWrapper, GNNConv1DReverse
from dataset import Addition, LowRankDataset, Inverse
from reasoning_dataset import FamilyTreeDataset, GraphConnectivityDataset, FamilyDatasetWrapper, GraphDatasetWrapper
from planning_dataset import PlanningDataset, PlanningDatasetOnline
from sat_dataset import SATNetDataset, SudokuDataset, SudokuRRNDataset, SudokuRRNLatentDataset
import torch
import numpy as np
import random

import argparse

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    print('Warning: MKL not initialized.')


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))


def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')

parser.add_argument('--dataset', default='inverse', type=str, help='dataset to evaluate')
parser.add_argument('--inspect-dataset', action='store_true', help='run an IPython embed interface after loading the dataset')
parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'mlp-reverse', 'sudoku', 'sudoku-latent', 'sudoku-transformer', 'sudoku-reverse', 'gnn', 'gnn-reverse', 'gnn-conv', 'gnn-conv-1d', 'gnn-conv-1d-v2', 'gnn-conv-1d-v2-reverse'])
parser.add_argument('--load-milestone', type=str, default=None, help='load a model from a milestone')
parser.add_argument('--batch_size', default=2048, type=int, help='size of batch of input to use')
parser.add_argument('--diffusion_steps', default=10, type=int, help='number of diffusion time steps (default: 10)')
parser.add_argument('--rank', default=20, type=int, help='rank of matrix to use')
parser.add_argument('--data-workers', type=int, default=None, help='number of workers to use for data loading')
parser.add_argument('--supervise-energy-landscape', type=str2bool, default=False)
parser.add_argument('--use-innerloop-opt', type=str2bool, default=False)
parser.add_argument('--cond_mask', type=str2bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--latent', action='store_true', default=False)
parser.add_argument('--ood', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
# ANM (Adversarial Negative Mining) arguments
parser.add_argument('--use-anm', action='store_true', default=False, help='Use adversarial negative mining')
parser.add_argument('--train-steps', type=int, default=2000, help='Total number of training steps')

# Curriculum configuration
parser.add_argument('--curriculum', type=str, default='aggressive', help='Name of curriculum to use (default, aggressive, conservative)')

# Override parameters (these override curriculum values if specified)
parser.add_argument('--anm-epsilon', type=float, default=None, help='ANM epsilon value (default: 0.1, scaled by curriculum)')
parser.add_argument('--anm-adversarial-steps', type=int, default=None, help='Number of gradient ascent steps for ANM (default: 5)')
parser.add_argument('--anm-learning-rate', type=float, default=None, help='Learning rate for model training (default: 1e-4)')
parser.add_argument('--anm-distance-penalty', type=float, default=None, help='Distance penalty weight for ANM (default: same as anm-epsilon)')
parser.add_argument('--anm-temperature', type=float, default=None, help='Temperature for loss scaling (default: varies by curriculum stage, typically 1.0-10.0)')
parser.add_argument('--anm-clean-ratio', type=float, default=None, help='Proportion of clean examples (default: varies by curriculum stage, 0.05-1.0)')
parser.add_argument('--anm-adversarial-ratio', type=float, default=None, help='Proportion of adversarial examples (default: varies by curriculum stage, 0.0-0.9)')
parser.add_argument('--anm-gaussian-ratio', type=float, default=None, help='Proportion of Gaussian noise examples (default: varies by curriculum stage, 0.05-0.1)')
parser.add_argument('--anm-hard-negative-ratio', type=float, default=None, help='Proportion of energy-based hard negative examples (default: varies by curriculum stage, 0.0-0.7)')
parser.add_argument('--anm-warmup-steps', type=int, default=None, help='Number of warmup steps before adversarial samples (default: 10% of train steps)')

# Hard negative mining specific parameters
parser.add_argument('--hnm-num-candidates', type=int, default=10, help='Number of HNM candidates to generate (default: 10)')
parser.add_argument('--hnm-refinement-steps', type=int, default=5, help='Energy descent steps per HNM candidate (default: 5)')
parser.add_argument('--hnm-lambda-weight', type=float, default=1.0, help='Balance between energy and error in HNM deception score (default: 1.0)')

# Random seed for reproducibility
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(FLAGS.seed)

    validation_dataset = None
    extra_validation_datasets = dict()
    extra_validation_every_mul = 10
    save_and_sample_every = 1000
    validation_batch_size = 256

    if FLAGS.dataset == "addition":
        dataset = Addition("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = dataset
        metric = 'mse'
    elif FLAGS.dataset == "inverse":
        dataset = Inverse("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = dataset
        metric = 'mse'
    elif FLAGS.dataset == "lowrank":
        dataset = LowRankDataset("train", FLAGS.rank, FLAGS.ood)
        validation_dataset = dataset
        metric = 'mse'
    elif FLAGS.dataset == 'parents':
        dataset = FamilyDatasetWrapper(FamilyTreeDataset((12, 12), epoch_size=int(1e5), task='parents'))
        metric = 'bce'
    elif FLAGS.dataset == 'uncle':
        dataset = FamilyDatasetWrapper(FamilyTreeDataset((12, 12), epoch_size=int(1e5), task='uncle'))
        metric = 'bce'
    elif FLAGS.dataset == 'connectivity':
        dataset = GraphDatasetWrapper(GraphConnectivityDataset((12, 12), 0.1, epoch_size=int(2048 * 1000), gen_method='dnc'))
        extra_validation_datasets = {
            'connectivity-13': GraphDatasetWrapper(GraphConnectivityDataset((13, 13), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-15': GraphDatasetWrapper(GraphConnectivityDataset((15, 15), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-18': GraphDatasetWrapper(GraphConnectivityDataset((18, 18), 0.1, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-20': GraphDatasetWrapper(GraphConnectivityDataset((20, 20), 0.1, epoch_size=int(1e3), gen_method='dnc'))
        }
        validation_batch_size = 64
        metric = 'bce'
    elif FLAGS.dataset == 'connectivity-2':
        dataset = GraphDatasetWrapper(GraphConnectivityDataset((12, 12), 0.2, epoch_size=int(2048 * 1000), gen_method='dnc'))
        extra_validation_datasets = {
            'connectivity-13': GraphDatasetWrapper(GraphConnectivityDataset((13, 13), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-15': GraphDatasetWrapper(GraphConnectivityDataset((15, 15), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-18': GraphDatasetWrapper(GraphConnectivityDataset((18, 18), 0.2, epoch_size=int(1e3), gen_method='dnc')),
            'connectivity-20': GraphDatasetWrapper(GraphConnectivityDataset((20, 20), 0.1, epoch_size=int(1e3), gen_method='dnc'))
        }
        validation_batch_size = 64
        metric = 'bce'
    elif FLAGS.dataset.startswith('parity'):
        dataset = SATNetDataset(FLAGS.dataset)
        metric = 'bce'
    elif FLAGS.dataset == 'sudoku':
        train_dataset = SudokuDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuDataset(FLAGS.dataset, split='val')
        extra_validation_datasets = {'sudoku-rrn-test': SudokuRRNDataset('sudoku-rrn', split='test')}
        dataset = train_dataset
        metric = 'sudoku'
        validation_batch_size = 64
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn':
        train_dataset = SudokuRRNDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNDataset(FLAGS.dataset, split='test')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku'
        assert FLAGS.cond_mask
    elif FLAGS.dataset == 'sudoku-rrn-latent':
        train_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='train')
        validation_dataset = SudokuRRNLatentDataset(FLAGS.dataset, split='validation')
        save_and_sample_every = 10000
        dataset = train_dataset
        metric = 'sudoku_latent'
    elif FLAGS.dataset == 'sort':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'sort-15': PlanningDataset(FLAGS.dataset + '-15', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'sort'
    elif FLAGS.dataset == 'sort-2':
        train_dataset = PlanningDatasetOnline('list-sorting-2', n=10)
        validation_dataset = PlanningDatasetOnline('list-sorting-2', n=10)
        extra_validation_datasets = {
            'sort-15': PlanningDatasetOnline('list-sorting-2', n=15)
        }
        dataset = train_dataset
        metric = 'sort-2'
    elif FLAGS.dataset == 'shortest-path':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=10000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=10000)
        dataset = train_dataset
        metric = 'bce'
    elif FLAGS.dataset == 'shortest-path-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-25': PlanningDataset('shortest-path-25-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 64
    elif FLAGS.dataset == 'shortest-path-10-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-15': PlanningDataset('shortest-path-15-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 128
    elif FLAGS.dataset == 'shortest-path-15-1d':
        train_dataset = PlanningDataset(FLAGS.dataset, split='train', num_identifier=100000)
        validation_dataset = PlanningDataset(FLAGS.dataset, split='validation', num_identifier=100000)
        extra_validation_datasets = {
            # it's fine to keep split=train because this is a generalization dataset.
            'shortest-path-20': PlanningDataset('shortest-path-1d', split='train', num_identifier=10000)
        }
        dataset = train_dataset
        metric = 'shortest-path-1d'
        validation_batch_size = 128
    else:
        assert False

    if FLAGS.inspect_dataset:
        from IPython import embed
        embed()
        exit()

    if FLAGS.model == 'mlp':
        model = EBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'mlp-reverse':
        model = EBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
            is_ebm = False,
        )
    elif FLAGS.model == 'sudoku':
        model = SudokuEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-latent':
        model = SudokuLatentEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-transformer':
        model = SudokuTransformerEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = DiffusionWrapper(model)
    elif FLAGS.model == 'sudoku-reverse':
        model = SudokuDenoise(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
    elif FLAGS.model == 'gnn':
        model = GraphEBM(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
        model = GNNDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-reverse':
        model = GraphReverse(
            inp_dim = dataset.inp_dim,
            out_dim = dataset.out_dim,
        )
    elif FLAGS.model == 'gnn-conv':
        model = GNNConvEBM(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
        model = GNNConvDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d':
        model = GNNConvEBM(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim, use_1d = True)
        model = GNNConvDiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d-v2':
        model = GNNConv1DEBMV2(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
        model = GNNConv1DV2DiffusionWrapper(model)
    elif FLAGS.model == 'gnn-conv-1d-v2-reverse':
        model = GNNConv1DReverse(inp_dim = dataset.inp_dim, out_dim = dataset.out_dim)
    else:
        assert False

    kwargs = dict()
    if FLAGS.baseline:
        kwargs['baseline'] = True

    if FLAGS.dataset in ['addition', 'inverse', 'lowrank']:
        kwargs['continuous'] = True

    if FLAGS.dataset in ['sudoku', 'sudoku_latent', 'sudoku-rrn', 'sudoku-rrn-latent']:
        kwargs['sudoku'] = True

    if FLAGS.dataset in ['connectivity', 'connectivity-2']:
        kwargs['connectivity'] = True

    if FLAGS.dataset in ['shortest-path', 'shortest-path-1d']:
        kwargs['shortest_path'] = True

    # Create curriculum config when ANM is used
    curriculum_config = None
    anm_epsilon = FLAGS.anm_epsilon
    anm_adversarial_steps = FLAGS.anm_adversarial_steps
    anm_learning_rate = FLAGS.anm_learning_rate
    anm_warmup_steps = FLAGS.anm_warmup_steps
    
    if FLAGS.use_anm:
        from curriculum_config import get_curriculum_by_name, CurriculumStage, CurriculumConfig
        
        def normalize_curriculum_ratios(stage, overrides):
            """
            Normalize curriculum ratios to ensure they sum to 1.0 when overrides are applied.
            
            Args:
                stage: Original CurriculumStage
                overrides: Dict of ratio overrides from command line flags
                
            Returns:
                Dict of normalized ratios that sum to 1.0
            """
            # Get original ratios
            original_ratios = {
                'clean_ratio': stage.clean_ratio,
                'adversarial_ratio': stage.adversarial_ratio,
                'gaussian_ratio': stage.gaussian_ratio,
                'hard_negative_ratio': stage.hard_negative_ratio
            }
            
            # Apply overrides with validation
            final_ratios = {}
            overridden_keys = set()
            override_sum = 0.0
            
            for key, original_value in original_ratios.items():
                if key in overrides and overrides[key] is not None:
                    override_value = overrides[key]
                    # Validate override ratio is non-negative
                    if override_value < 0:
                        raise ValueError(f"Ratio override for {key} must be non-negative, got {override_value}")
                    final_ratios[key] = override_value
                    overridden_keys.add(key)
                    override_sum += override_value
                else:
                    final_ratios[key] = original_value
            
            # Check if overrides alone exceed 1.0
            if override_sum > 1.0:
                print(f"Warning: Override ratios sum to {override_sum:.3f} > 1.0. "
                      f"All non-overridden ratios will be set to 0.0")
                # Set all non-overridden ratios to 0
                for key in original_ratios.keys():
                    if key not in overridden_keys:
                        final_ratios[key] = 0.0
                # Normalize overrides to sum to 1.0
                if override_sum > 0:
                    for key in overridden_keys:
                        final_ratios[key] = final_ratios[key] / override_sum
            else:
                # Calculate remaining capacity for non-overridden ratios
                remaining_capacity = 1.0 - override_sum
                
                # Get sum of non-overridden original ratios
                non_overridden_sum = sum(original_ratios[key] for key in original_ratios.keys() 
                                       if key not in overridden_keys)
                
                # Scale non-overridden ratios to fit remaining capacity
                if non_overridden_sum > 0 and remaining_capacity >= 0:
                    scale_factor = remaining_capacity / non_overridden_sum
                    for key in original_ratios.keys():
                        if key not in overridden_keys:
                            final_ratios[key] = original_ratios[key] * scale_factor
                    
                    # Log ratio adjustments if significant scaling occurred
                    if abs(scale_factor - 1.0) > 0.1:
                        non_overridden_names = [key.replace('_ratio', '') 
                                              for key in original_ratios.keys() 
                                              if key not in overridden_keys]
                        print(f"Note: Scaled {', '.join(non_overridden_names)} ratios by "
                              f"{scale_factor:.3f} to accommodate overrides")
                elif remaining_capacity < 0:
                    # This shouldn't happen due to the check above, but handle it
                    for key in original_ratios.keys():
                        if key not in overridden_keys:
                            final_ratios[key] = 0.0
            
            # Final validation and precision correction
            total_sum = sum(final_ratios.values())
            if abs(total_sum - 1.0) > 1e-10:  # Handle floating point precision
                # Normalize to exactly 1.0 if close enough
                if abs(total_sum - 1.0) < 0.01:  # Within 1%
                    for key in final_ratios:
                        final_ratios[key] = final_ratios[key] / total_sum
                else:
                    raise ValueError(f"Ratio normalization failed: final sum is {total_sum}, expected ~1.0")
            
            return final_ratios
        
        # Use the specified curriculum (default is 'aggressive')
        curriculum_config = get_curriculum_by_name(FLAGS.curriculum)
        curriculum_config.total_steps = FLAGS.train_steps
        # Disable validation gating for simplicity
        curriculum_config.enable_validation_gating = False
        
        # Apply overrides to curriculum stages if specified
        if any([FLAGS.anm_temperature is not None,
                FLAGS.anm_clean_ratio is not None,
                FLAGS.anm_adversarial_ratio is not None,
                FLAGS.anm_gaussian_ratio is not None,
                FLAGS.anm_hard_negative_ratio is not None]):
            # Create a new curriculum config with overridden values
            modified_stages = {}
            
            # Prepare override dictionary
            ratio_overrides = {
                'clean_ratio': FLAGS.anm_clean_ratio,
                'adversarial_ratio': FLAGS.anm_adversarial_ratio,
                'gaussian_ratio': FLAGS.anm_gaussian_ratio,
                'hard_negative_ratio': FLAGS.anm_hard_negative_ratio
            }
            
            for (start_pct, end_pct), stage in curriculum_config.stages.items():
                # Get normalized ratios that sum to 1.0
                normalized_ratios = normalize_curriculum_ratios(stage, ratio_overrides)
                
                # Create a new stage with normalized ratios
                modified_stages[(start_pct, end_pct)] = CurriculumStage(
                    name=stage.name,
                    clean_ratio=normalized_ratios['clean_ratio'],
                    adversarial_ratio=normalized_ratios['adversarial_ratio'],
                    gaussian_ratio=normalized_ratios['gaussian_ratio'],
                    hard_negative_ratio=normalized_ratios['hard_negative_ratio'],
                    epsilon_multiplier=stage.epsilon_multiplier,  # Keep from curriculum
                    temperature=FLAGS.anm_temperature if FLAGS.anm_temperature is not None else stage.temperature,
                    focus=stage.focus + " (with overrides)"
                )
            curriculum_config.stages = modified_stages
        
        # Set default values from curriculum if not overridden
        if anm_epsilon is None:
            # Will be determined dynamically by curriculum
            anm_epsilon = 0.1  # Default base value, will be scaled by curriculum
        
        if anm_adversarial_steps is None:
            # Default value if not specified
            anm_adversarial_steps = 5
        
        if anm_warmup_steps is None:
            # Default to 10% of training steps
            anm_warmup_steps = int(0.1 * FLAGS.train_steps)
            
        # anm_distance_penalty is the same as epsilon if not specified
        if FLAGS.anm_distance_penalty is None:
            anm_distance_penalty = anm_epsilon
        else:
            anm_distance_penalty = FLAGS.anm_distance_penalty

    # When ANM is not used, set defaults for the parameters
    if not FLAGS.use_anm:
        anm_adversarial_steps = 5
        anm_distance_penalty = 0.1
        anm_warmup_steps = 0
    
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 32,
        objective = 'pred_noise',  # Alternative pred_x0
        timesteps = FLAGS.diffusion_steps,  # number of steps
        sampling_timesteps = FLAGS.diffusion_steps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
        supervise_energy_landscape = FLAGS.supervise_energy_landscape,
        use_innerloop_opt = FLAGS.use_innerloop_opt,
        show_inference_tqdm = False,
        use_adversarial_corruption=FLAGS.use_anm,
        anm_adversarial_steps=anm_adversarial_steps,
        anm_distance_penalty=anm_distance_penalty,
        anm_warmup_steps=anm_warmup_steps,
        curriculum_config=curriculum_config,
        hnm_num_candidates=FLAGS.hnm_num_candidates,
        hnm_refinement_steps=FLAGS.hnm_refinement_steps,
        hnm_lambda_weight=FLAGS.hnm_lambda_weight,
        **kwargs
    )

    result_dir = osp.join('results', f'ds_{FLAGS.dataset}', f'model_{FLAGS.model}')
    if FLAGS.diffusion_steps != 100:
        result_dir = result_dir + f'_diffsteps_{FLAGS.diffusion_steps}'
    if FLAGS.use_anm:
        # Check if specific hyperparameters were provided
        if FLAGS.anm_epsilon is not None and FLAGS.anm_adversarial_steps is not None:
            # Use specific hyperparameter suffix for sweep
            result_dir = result_dir + f'_anm_eps{FLAGS.anm_epsilon}_steps{FLAGS.anm_adversarial_steps}'
            # Add distance penalty to directory name if explicitly provided
            if FLAGS.anm_distance_penalty is not None:
                result_dir = result_dir + f'_dp{FLAGS.anm_distance_penalty}'
        elif FLAGS.anm_adversarial_steps is not None:
            # Phase 1 experiments: use steps-only format
            result_dir = result_dir + f'_anm_steps{FLAGS.anm_adversarial_steps}'
        else:
            result_dir = result_dir + '_anm_curriculum'  # Default ANM
    
    # Add seed suffix for Phase 1 experiments
    # Phase 1 always expects seed suffix, including when seed=42
    # We distinguish Phase 1 usage by checking multiple indicators:
    # 1. ANM with explicit adversarial steps (Phase 1 ANM experiments)
    # 2. Non-default seed (other Phase 1 experiments with different seeds)
    # 3. Non-default train steps (Phase 1 uses 1000 steps vs default 2000)
    print(f"DEBUG: Path construction params - seed={FLAGS.seed}, train_steps={FLAGS.train_steps}, use_anm={FLAGS.use_anm}, anm_steps={FLAGS.anm_adversarial_steps}")
    is_phase1_experiment = (
        (FLAGS.use_anm and FLAGS.anm_adversarial_steps is not None) or 
        FLAGS.seed != 42 or 
        FLAGS.train_steps != 2000
    )
    print(f"DEBUG: is_phase1_experiment={is_phase1_experiment}")
    if is_phase1_experiment:
        result_dir = result_dir + f'_seed{FLAGS.seed}'
    print(f"DEBUG: Creating results directory: {result_dir}")
    try:
        os.makedirs(result_dir, exist_ok=True)
        print(f"DEBUG: Results directory created successfully: {result_dir}")
        
        # Verify the directory was actually created and is writable
        if not os.path.exists(result_dir):
            raise OSError(f"Directory creation appeared successful but path doesn't exist: {result_dir}")
        if not os.path.isdir(result_dir):
            raise OSError(f"Path exists but is not a directory: {result_dir}")
        if not os.access(result_dir, os.W_OK):
            raise OSError(f"Directory exists but is not writable: {result_dir}")
            
        print(f"DEBUG: Directory verification passed: {result_dir}")
        
    except Exception as e:
        print(f"ERROR: Failed to create results directory '{result_dir}': {e}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise

    if FLAGS.latent:
        # Load the decoder
        autoencode_model = AutoencodeModel(729, 729)
        ckpt = torch.load("results/autoencode_sudoku-rrn/model_mlp_diffsteps_10/model-1.pt")
        model_ckpt = ckpt['model']
        autoencode_model.load_state_dict(model_ckpt)
    else:
        autoencode_model = None

    # Use the learning rate override if specified, otherwise default to 1e-4
    train_lr = anm_learning_rate if anm_learning_rate is not None else 1e-4
    
    trainer = Trainer1D(
        diffusion,
        dataset,
        train_batch_size = FLAGS.batch_size,
        validation_batch_size = validation_batch_size,
        train_lr = train_lr,
        train_num_steps = FLAGS.train_steps,         # total training steps from command line
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        data_workers = FLAGS.data_workers,
        amp = False,                      # turn on mixed precision
        metric = metric,
        results_folder = result_dir,
        cond_mask = FLAGS.cond_mask,
        validation_dataset = validation_dataset,
        extra_validation_datasets = extra_validation_datasets,
        extra_validation_every_mul = extra_validation_every_mul,
        save_and_sample_every = save_and_sample_every,
        evaluate_first = FLAGS.evaluate,  # run one evaluation first
        latent = FLAGS.latent,  # whether we are doing reasoning in the latent space
        autoencode_model = autoencode_model
    )

    if FLAGS.load_milestone is not None:
        trainer.load(FLAGS.load_milestone)

    trainer.train()
    
    # Save training metadata for debugging and validation
    try:
        import json
        import datetime
        
        metadata = {
            'seed': FLAGS.seed,
            'train_steps': FLAGS.train_steps,
            'model_type': 'anm' if FLAGS.use_anm else 'baseline',
            'dataset': FLAGS.dataset,
            'diffusion_steps': FLAGS.diffusion_steps,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate,
            'rank': FLAGS.rank,
            'timestamp': datetime.datetime.now().isoformat(),
            'is_phase1_experiment': is_phase1_experiment
        }
        
        # Add ANM-specific parameters if used
        if FLAGS.use_anm:
            metadata.update({
                'anm_adversarial_steps': FLAGS.anm_adversarial_steps,
                'anm_epsilon': getattr(FLAGS, 'anm_epsilon', None),
                'anm_temperature': getattr(FLAGS, 'anm_temperature', None),
                'anm_clean_ratio': getattr(FLAGS, 'anm_clean_ratio', None),
                'anm_adversarial_ratio': getattr(FLAGS, 'anm_adversarial_ratio', None),
                'anm_gaussian_ratio': getattr(FLAGS, 'anm_gaussian_ratio', None)
            })
        
        # Try to get git commit info
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               stderr=subprocess.DEVNULL).decode().strip()
            metadata['git_commit'] = git_commit
        except:
            pass  # Git info optional
        
        metadata_path = os.path.join(result_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"DEBUG: Training metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"WARNING: Failed to save training metadata: {e}")
        # Don't fail training if metadata saving fails

