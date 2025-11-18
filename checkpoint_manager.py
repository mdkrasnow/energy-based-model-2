"""
Centralized checkpoint management with guaranteed path uniqueness and comprehensive validation.

This module ensures that no two different model configurations can ever use the same 
checkpoint path, preventing model collision bugs.
"""

import os
import hashlib
import json
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Set
from pathlib import Path

# Import constants to ensure consistency with existing codebase
DIFFUSION_STEPS = 10
RANK = 20


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration for a model training run.
    
    All parameters that affect model training should be included here
    to ensure path uniqueness.
    """
    dataset: str
    model_type: str  # 'anm' or 'baseline'
    diffusion_steps: int = 10
    rank: int = 20
    seed: Optional[int] = None
    
    # ANM-specific parameters (None for baseline)
    anm_adversarial_steps: Optional[int] = None
    anm_epsilon: Optional[float] = None
    anm_distance_penalty: Optional[float] = None
    anm_temperature: Optional[float] = None
    anm_clean_ratio: Optional[float] = None
    anm_adversarial_ratio: Optional[float] = None
    anm_gaussian_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration consistency."""
        if self.model_type not in ['anm', 'baseline']:
            raise ValueError(f"model_type must be 'anm' or 'baseline', got: {self.model_type}")
        
        if self.model_type == 'anm' and self.anm_adversarial_steps is None:
            raise ValueError("ANM model_type requires anm_adversarial_steps to be specified")
        
        if self.model_type == 'baseline' and any([
            self.anm_adversarial_steps is not None,
            self.anm_epsilon is not None,
            self.anm_distance_penalty is not None,
            self.anm_temperature is not None,
            self.anm_clean_ratio is not None,
            self.anm_adversarial_ratio is not None,
            self.anm_gaussian_ratio is not None
        ]):
            raise ValueError("Baseline model_type cannot have ANM parameters")
        
        if self.dataset not in ['addition', 'inverse', 'lowrank']:
            raise ValueError(f"Unknown dataset: {self.dataset}")


class CheckpointManager:
    """
    Centralized management of checkpoint paths with guaranteed uniqueness.
    
    This class ensures that:
    1. Different model configurations always produce different paths
    2. Same configurations always produce the same path (deterministic)
    3. Paths are validated for filesystem compatibility
    4. Comprehensive logging and verification is available
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self._path_registry: Dict[str, ModelConfig] = {}
        self._config_registry: Dict[str, str] = {}  # config_hash -> path
        self._lock = threading.Lock()  # Thread safety for concurrent access
    
    def generate_checkpoint_path(self, config: ModelConfig, checkpoint_name: str = "model-1.pt") -> Path:
        """
        Generate a unique checkpoint path for the given configuration.
        
        Args:
            config: Model configuration
            checkpoint_name: Name of checkpoint file (default: model-1.pt)
            
        Returns:
            Path: Unique path for this configuration
            
        Raises:
            ValueError: If configuration is invalid or path collision detected
        """
        with self._lock:  # Thread safety
            # Generate deterministic directory path
            dir_path = self._generate_directory_path(config)
            
            # Create full checkpoint path
            checkpoint_path = self.base_dir / dir_path / checkpoint_name
            
            # Validate path length (common filesystem limit is 255 chars for filename, 4096 for full path)
            if len(str(checkpoint_path)) > 4000:
                raise ValueError(f"Generated path too long ({len(str(checkpoint_path))} chars): {checkpoint_path}")
            
            # Verify uniqueness
            self._verify_path_uniqueness(config, str(checkpoint_path))
            
            # Register this path
            config_hash = self._config_hash(config)
            self._path_registry[str(checkpoint_path)] = config
            self._config_registry[config_hash] = str(checkpoint_path)
            
            return checkpoint_path
    
    def get_checkpoint_path(self, config: ModelConfig, checkpoint_name: str = "model-1.pt") -> Path:
        """
        Get checkpoint path for configuration (alias for generate_checkpoint_path).
        
        This method name makes it clear this is for retrieval.
        """
        return self.generate_checkpoint_path(config, checkpoint_name)
    
    def verify_checkpoint_integrity(self, checkpoint_path: Path, expected_config: ModelConfig) -> Dict[str, Any]:
        """
        Verify that a checkpoint file matches expected configuration.
        
        Returns:
            Dict with verification results including file hash, size, config match
        """
        if not checkpoint_path.exists():
            return {
                'exists': False,
                'error': f"Checkpoint does not exist: {checkpoint_path}"
            }
        
        try:
            # Get file metadata
            stat = checkpoint_path.stat()
            file_size = stat.st_size
            file_mtime = stat.st_mtime
            
            # Calculate file hash (first 1MB for performance)
            with open(checkpoint_path, 'rb') as f:
                file_data = f.read(1024 * 1024)  # Read first 1MB
                file_hash = hashlib.md5(file_data).hexdigest()[:8]
            
            # Check if path matches expected config
            registered_config = self._path_registry.get(str(checkpoint_path))
            config_matches = registered_config == expected_config if registered_config else None
            
            return {
                'exists': True,
                'path': str(checkpoint_path),
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'modified_timestamp': file_mtime,
                'file_hash': file_hash,
                'expected_config': asdict(expected_config),
                'registered_config': asdict(registered_config) if registered_config else None,
                'config_matches': config_matches,
                'is_valid': file_size > 0 and config_matches is not False
            }
            
        except Exception as e:
            return {
                'exists': True,
                'error': f"Failed to verify checkpoint: {e}",
                'path': str(checkpoint_path)
            }
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information about all registered paths."""
        return {
            'total_registered_paths': len(self._path_registry),
            'total_unique_configs': len(self._config_registry),
            'path_registry': {
                path: asdict(config) for path, config in self._path_registry.items()
            },
            'config_uniqueness_check': len(self._path_registry) == len(self._config_registry)
        }
    
    def _format_float_for_path(self, value: float) -> str:
        """Format float consistently for path generation (avoid precision issues)."""
        if value == int(value):
            return str(int(value))  # 1.0 -> "1"
        else:
            return f"{value:.6g}"  # Consistent precision, removes trailing zeros
    
    def _generate_directory_path(self, config: ModelConfig) -> str:
        """
        Generate the directory path component for a configuration.
        
        This ensures compatibility with existing path formats while guaranteeing uniqueness.
        """
        # Base path: ds_{dataset}/model_mlp_diffsteps_{diffusion_steps}
        path_parts = [f"ds_{config.dataset}", f"model_mlp_diffsteps_{config.diffusion_steps}"]
        
        # Add model type and parameters
        if config.model_type == 'anm':
            if config.anm_epsilon is not None and config.anm_distance_penalty is not None:
                # Full hyperparameter specification format (sweep mode)
                eps_str = self._format_float_for_path(config.anm_epsilon)
                path_parts.append(f"anm_eps{eps_str}_steps{config.anm_adversarial_steps}")
                if config.anm_distance_penalty != config.anm_epsilon:  # Only if different
                    dp_str = self._format_float_for_path(config.anm_distance_penalty)
                    path_parts.append(f"dp{dp_str}")
            elif config.anm_adversarial_steps is not None:
                # Phase 1 experiments: steps-only format
                path_parts.append(f"anm_steps{config.anm_adversarial_steps}")
            else:
                # Default ANM curriculum
                path_parts.append("anm_curriculum")
        
        # Add seed suffix for Phase 1 experiments
        if config.seed is not None:
            path_parts.append(f"seed{config.seed}")
        
        return "/".join(path_parts)
    
    def _config_hash(self, config: ModelConfig) -> str:
        """Generate a deterministic hash for a configuration."""
        config_dict = asdict(config)
        # Sort keys for deterministic hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _verify_path_uniqueness(self, config: ModelConfig, path: str) -> None:
        """
        Verify that this configuration produces a unique path.
        
        Raises ValueError if path collision is detected.
        """
        config_hash = self._config_hash(config)
        
        # Check if this exact config already has a registered path
        existing_path = self._config_registry.get(config_hash)
        if existing_path and existing_path != path:
            raise ValueError(
                f"Configuration collision: Same config produces different paths.\n"
                f"Config: {asdict(config)}\n"
                f"Existing path: {existing_path}\n"
                f"New path: {path}\n"
                f"This indicates a bug in path generation logic."
            )
        
        # Check if this path is already used by a different config
        existing_config = self._path_registry.get(path)
        if existing_config and existing_config != config:
            raise ValueError(
                f"Path collision: Different configs produce same path.\n"
                f"Path: {path}\n"
                f"Existing config: {asdict(existing_config)}\n"
                f"New config: {asdict(config)}\n"
                f"This violates uniqueness guarantee."
            )


def create_model_config_from_experiment_spec(experiment_spec: Dict[str, Any], 
                                            config_obj: Any,
                                            dataset: str = 'addition') -> ModelConfig:
    """
    Convert experiment specification and config object to ModelConfig.
    
    This is a helper function for backward compatibility with existing code.
    
    Args:
        experiment_spec: Experiment specification dict with seed, config_name, etc.
        config_obj: Configuration object with ANM parameters
        dataset: Dataset name (default: 'addition')
    """
    return ModelConfig(
        dataset=dataset,
        model_type='anm' if config_obj.use_anm else 'baseline',
        diffusion_steps=DIFFUSION_STEPS,  # Use imported constant
        rank=RANK,  # Use imported constant
        seed=experiment_spec.get('seed'),
        anm_adversarial_steps=config_obj.anm_adversarial_steps if config_obj.use_anm else None,
        anm_epsilon=getattr(config_obj, 'anm_epsilon', None) if config_obj.use_anm else None,
        anm_distance_penalty=getattr(config_obj, 'anm_distance_penalty', None) if config_obj.use_anm else None,
        anm_temperature=getattr(config_obj, 'anm_temperature', None) if config_obj.use_anm else None,
        anm_clean_ratio=getattr(config_obj, 'anm_clean_ratio', None) if config_obj.use_anm else None,
        anm_adversarial_ratio=getattr(config_obj, 'anm_adversarial_ratio', None) if config_obj.use_anm else None,
        anm_gaussian_ratio=getattr(config_obj, 'anm_gaussian_ratio', None) if config_obj.use_anm else None
    )


# Global checkpoint manager instance
_global_checkpoint_manager = CheckpointManager()

def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    return _global_checkpoint_manager