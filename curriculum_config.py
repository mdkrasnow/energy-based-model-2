"""
Comprehensive curriculum configuration system for adversarial training.

This module provides a flexible framework for defining and managing training
curricula that gradually introduce adversarial examples during the training process.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
import math


@dataclass
class CurriculumStage:
    """
    Defines a single stage in the adversarial training curriculum.
    
    Args:
        name: Descriptive name for this stage
        clean_ratio: Proportion of clean (original) examples [0.0, 1.0]
        adversarial_ratio: Proportion of adversarial examples [0.0, 1.0]
        gaussian_ratio: Proportion of Gaussian noise examples [0.0, 1.0]
        epsilon_multiplier: Multiplier for the base epsilon value [0.0, 1.0+]
        temperature: Temperature parameter for loss scaling
        focus: Description of what this stage focuses on
    """
    name: str
    clean_ratio: float
    adversarial_ratio: float
    gaussian_ratio: float
    epsilon_multiplier: float
    temperature: float
    focus: str

    def __post_init__(self):
        """Validate that ratios sum to approximately 1.0."""
        total_ratio = self.clean_ratio + self.adversarial_ratio + self.gaussian_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")


@dataclass
class CurriculumConfig:
    """
    Complete configuration for adversarial training curriculum.
    
    Args:
        total_steps: Total number of training steps
        stages: Mapping from (start_pct, end_pct) to CurriculumStage
        enable_validation_gating: Whether to use validation metrics to gate progression
        validation_threshold: Minimum validation accuracy improvement to advance
        rollback_steps: Number of steps to rollback on validation failure
    """
    total_steps: int
    stages: Dict[Tuple[float, float], CurriculumStage]
    enable_validation_gating: bool = True
    validation_threshold: float = 0.02
    rollback_steps: int = 5000

    def get_stage(self, current_step: int) -> CurriculumStage:
        """
        Get the appropriate curriculum stage for the current training step.
        
        Args:
            current_step: Current training step number
            
        Returns:
            CurriculumStage for the current training progress
        """
        if current_step < 0:
            current_step = 0
        if current_step > self.total_steps:
            current_step = self.total_steps
            
        progress = current_step / self.total_steps
        
        for (start_pct, end_pct), stage in self.stages.items():
            if start_pct <= progress < end_pct:
                return stage
        
        # If we're at exactly 100%, return the last stage
        if progress >= 1.0:
            last_stage = max(self.stages.items(), key=lambda x: x[0][1])
            return last_stage[1]
        
        # Fallback to first stage
        first_stage = min(self.stages.items(), key=lambda x: x[0][0])
        return first_stage[1]

    def get_smooth_epsilon(self, current_step: int, max_epsilon: float) -> float:
        """
        Calculate smoothly ramped epsilon value based on current stage.
        
        Args:
            current_step: Current training step
            max_epsilon: Maximum epsilon value to scale to
            
        Returns:
            Smoothed epsilon value for current step
        """
        stage = self.get_stage(current_step)
        base_epsilon = max_epsilon * stage.epsilon_multiplier
        
        # Apply smooth ramping within the stage
        progress = current_step / self.total_steps
        
        # Find current stage boundaries
        current_stage_bounds = None
        for (start_pct, end_pct), s in self.stages.items():
            if s == stage:
                current_stage_bounds = (start_pct, end_pct)
                break
        
        if current_stage_bounds:
            start_pct, end_pct = current_stage_bounds
            stage_progress = (progress - start_pct) / (end_pct - start_pct)
            # Smooth interpolation using cosine
            smooth_factor = 0.5 * (1 - math.cos(math.pi * stage_progress))
            return base_epsilon * smooth_factor
        
        return base_epsilon


# Default curriculum configuration
DEFAULT_CURRICULUM = CurriculumConfig(
    total_steps=100000,  # Default, should be overridden
    stages={
        (0.0, 0.20): CurriculumStage(
            name="warmup",
            clean_ratio=1.0,
            adversarial_ratio=0.0,
            gaussian_ratio=0.0,
            epsilon_multiplier=0.0,
            temperature=10.0,
            focus="Clean data training to establish baseline"
        ),
        (0.20, 0.35): CurriculumStage(
            name="introduction",
            clean_ratio=0.7,
            adversarial_ratio=0.2,
            gaussian_ratio=0.1,
            epsilon_multiplier=0.1,
            temperature=8.0,
            focus="Gentle introduction of adversarial examples"
        ),
        (0.35, 0.55): CurriculumStage(
            name="acceleration",
            clean_ratio=0.5,
            adversarial_ratio=0.4,
            gaussian_ratio=0.1,
            epsilon_multiplier=0.3,
            temperature=5.0,
            focus="Balanced mix with increasing adversarial strength"
        ),
        (0.55, 0.75): CurriculumStage(
            name="dominance",
            clean_ratio=0.3,
            adversarial_ratio=0.6,
            gaussian_ratio=0.1,
            epsilon_multiplier=0.7,
            temperature=2.0,
            focus="Adversarial examples become dominant"
        ),
        (0.75, 0.90): CurriculumStage(
            name="mastery",
            clean_ratio=0.15,
            adversarial_ratio=0.8,
            gaussian_ratio=0.05,
            epsilon_multiplier=1.0,
            temperature=1.5,
            focus="High-strength adversarial training"
        ),
        (0.90, 1.0): CurriculumStage(
            name="fine-tuning",
            clean_ratio=0.2,
            adversarial_ratio=0.75,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.8,
            temperature=1.0,
            focus="Final refinement with slight epsilon reduction"
        ),
    }
)

# Aggressive curriculum for faster adversarial introduction
AGGRESSIVE_CURRICULUM = CurriculumConfig(
    total_steps=100000,
    stages={
        (0.0, 0.10): CurriculumStage(
            name="warmup",
            clean_ratio=1.0,
            adversarial_ratio=0.0,
            gaussian_ratio=0.0,
            epsilon_multiplier=0.0,
            temperature=5.0,
            focus="Minimal clean training"
        ),
        (0.10, 0.25): CurriculumStage(
            name="rapid_introduction",
            clean_ratio=0.5,
            adversarial_ratio=0.4,
            gaussian_ratio=0.1,
            epsilon_multiplier=0.3,
            temperature=4.0,
            focus="Rapid adversarial introduction"
        ),
        (0.25, 0.50): CurriculumStage(
            name="aggressive_ramp",
            clean_ratio=0.2,
            adversarial_ratio=0.7,
            gaussian_ratio=0.1,
            epsilon_multiplier=0.7,
            temperature=2.0,
            focus="Aggressive adversarial ramping"
        ),
        (0.50, 0.80): CurriculumStage(
            name="high_intensity",
            clean_ratio=0.1,
            adversarial_ratio=0.85,
            gaussian_ratio=0.05,
            epsilon_multiplier=1.0,
            temperature=1.5,
            focus="High-intensity adversarial training"
        ),
        (0.80, 1.0): CurriculumStage(
            name="extreme_hardening",
            clean_ratio=0.05,
            adversarial_ratio=0.9,
            gaussian_ratio=0.05,
            epsilon_multiplier=1.2,
            temperature=1.0,
            focus="Extreme adversarial hardening"
        ),
    }
)

# Conservative curriculum for gradual introduction
CONSERVATIVE_CURRICULUM = CurriculumConfig(
    total_steps=100000,
    stages={
        (0.0, 0.30): CurriculumStage(
            name="extended_warmup",
            clean_ratio=1.0,
            adversarial_ratio=0.0,
            gaussian_ratio=0.0,
            epsilon_multiplier=0.0,
            temperature=15.0,
            focus="Extended clean training for stable foundation"
        ),
        (0.30, 0.45): CurriculumStage(
            name="gentle_introduction",
            clean_ratio=0.85,
            adversarial_ratio=0.1,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.05,
            temperature=12.0,
            focus="Very gentle adversarial introduction"
        ),
        (0.45, 0.60): CurriculumStage(
            name="gradual_increase",
            clean_ratio=0.7,
            adversarial_ratio=0.25,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.15,
            temperature=8.0,
            focus="Gradual adversarial increase"
        ),
        (0.60, 0.75): CurriculumStage(
            name="moderate_training",
            clean_ratio=0.55,
            adversarial_ratio=0.4,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.4,
            temperature=5.0,
            focus="Moderate adversarial training"
        ),
        (0.75, 0.90): CurriculumStage(
            name="balanced_hardening",
            clean_ratio=0.4,
            adversarial_ratio=0.55,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.7,
            temperature=3.0,
            focus="Balanced adversarial hardening"
        ),
        (0.90, 1.0): CurriculumStage(
            name="conservative_mastery",
            clean_ratio=0.3,
            adversarial_ratio=0.65,
            gaussian_ratio=0.05,
            epsilon_multiplier=0.9,
            temperature=2.0,
            focus="Conservative final mastery"
        ),
    }
)


def get_curriculum_by_name(name: str) -> CurriculumConfig:
    """
    Get a predefined curriculum configuration by name.
    
    Args:
        name: Name of the curriculum ('default', 'aggressive', 'conservative')
        
    Returns:
        CurriculumConfig object for the specified curriculum
        
    Raises:
        ValueError: If curriculum name is not recognized
    """
    curricula = {
        'default': DEFAULT_CURRICULUM,
        'aggressive': AGGRESSIVE_CURRICULUM,
        'conservative': CONSERVATIVE_CURRICULUM,
    }
    
    name = name.lower()
    if name not in curricula:
        available = ', '.join(curricula.keys())
        raise ValueError(f"Unknown curriculum '{name}'. Available: {available}")
    
    return curricula[name]


def calculate_adaptive_epsilon(
    loss_history: List[float],
    current_epsilon: float,
    target_loss_reduction: float = 0.02,
    adaptation_rate: float = 0.1,
    min_epsilon: float = 0.001,
    max_epsilon: float = 0.3
) -> float:
    """
    Calculate adaptive epsilon based on recent loss history.
    
    This function adjusts the epsilon value based on how well the model is
    learning. If loss is decreasing well, epsilon can be increased to make
    training more challenging. If loss is stagnating, epsilon is decreased.
    
    Args:
        loss_history: List of recent loss values (most recent last)
        current_epsilon: Current epsilon value
        target_loss_reduction: Target loss reduction rate per step
        adaptation_rate: How quickly to adapt epsilon
        min_epsilon: Minimum allowed epsilon value
        max_epsilon: Maximum allowed epsilon value
        
    Returns:
        Adapted epsilon value
    """
    if len(loss_history) < 10:
        return current_epsilon
    
    # Calculate recent loss trend
    recent_losses = loss_history[-10:]
    if len(recent_losses) < 2:
        return current_epsilon
    
    # Linear regression on recent losses to get trend
    n = len(recent_losses)
    x_mean = (n - 1) / 2
    y_mean = sum(recent_losses) / n
    
    numerator = sum((i - x_mean) * (loss - y_mean) for i, loss in enumerate(recent_losses))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    # Adapt epsilon based on loss trend
    if slope < -target_loss_reduction:  # Loss decreasing faster than target
        # Increase epsilon to make training more challenging
        new_epsilon = current_epsilon * (1 + adaptation_rate)
    elif slope > 0:  # Loss increasing
        # Decrease epsilon to make training easier
        new_epsilon = current_epsilon * (1 - adaptation_rate)
    else:
        # Loss trend is acceptable, small adjustment toward target
        adjustment = (target_loss_reduction + slope) * adaptation_rate
        new_epsilon = current_epsilon * (1 + adjustment)
    
    # Clamp to valid range
    new_epsilon = max(min_epsilon, min(new_epsilon, max_epsilon))
    
    return new_epsilon


def create_custom_curriculum(
    total_steps: int,
    stage_configs: List[Dict],
    enable_validation_gating: bool = True,
    validation_threshold: float = 0.02,
    rollback_steps: int = 5000
) -> CurriculumConfig:
    """
    Create a custom curriculum configuration.
    
    Args:
        total_steps: Total number of training steps
        stage_configs: List of dictionaries defining each stage
        enable_validation_gating: Whether to use validation gating
        validation_threshold: Validation threshold for progression
        rollback_steps: Steps to rollback on validation failure
        
    Returns:
        Custom CurriculumConfig object
        
    Example:
        stage_configs = [
            {
                'start_pct': 0.0, 'end_pct': 0.5, 'name': 'warmup',
                'clean_ratio': 1.0, 'adversarial_ratio': 0.0, 'gaussian_ratio': 0.0,
                'epsilon_multiplier': 0.0, 'temperature': 10.0,
                'focus': 'Clean training'
            },
            ...
        ]
    """
    stages = {}
    
    for config in stage_configs:
        stage_key = (config['start_pct'], config['end_pct'])
        stage = CurriculumStage(
            name=config['name'],
            clean_ratio=config['clean_ratio'],
            adversarial_ratio=config['adversarial_ratio'],
            gaussian_ratio=config['gaussian_ratio'],
            epsilon_multiplier=config['epsilon_multiplier'],
            temperature=config['temperature'],
            focus=config['focus']
        )
        stages[stage_key] = stage
    
    return CurriculumConfig(
        total_steps=total_steps,
        stages=stages,
        enable_validation_gating=enable_validation_gating,
        validation_threshold=validation_threshold,
        rollback_steps=rollback_steps
    )