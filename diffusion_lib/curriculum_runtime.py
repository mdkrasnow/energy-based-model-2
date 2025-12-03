from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
from functools import lru_cache

from curriculum_config import CurriculumConfig


@dataclass
class StageSnapshot:
    name: str
    clean_ratio: float
    adversarial_ratio: float
    gaussian_ratio: float
    hard_negative_ratio: float
    epsilon_multiplier: float
    temperature: float


class ValidationGatedCurriculum:
    """
    Manages validation-based gating (moved from diffusion module).
    Behavior preserved 1:1.
    """
    def __init__(self, config: CurriculumConfig, window_size: int = 10):
        self.config = config
        self.window_size = window_size

        self.val_loss_history = collections.deque(maxlen=window_size)
        self.val_accuracy_history = collections.deque(maxlen=window_size)
        self.val_loss_ema: Optional[float] = None
        self.val_accuracy_ema: Optional[float] = None

        self.stage_baselines = {}  # stage_name -> {'loss': float, 'accuracy': float}
        self.current_stage_baseline = None

        self.intensity_multiplier = 1.0
        self.consecutive_degradations = 0
        self.last_adjustment_step = 0
        self.rollback_count = 0

        self.performance_trend = "stable"  # informational only
        self.adjustment_history = []  # (step, adjustment_type, reason, old_intensity, new_intensity)

    def update_validation_metrics(self, val_loss: float, val_accuracy: float, step: int):
        self.val_loss_history.append(val_loss)
        self.val_accuracy_history.append(val_accuracy)

        alpha = 0.1
        if self.val_loss_ema is None:
            self.val_loss_ema = val_loss
            self.val_accuracy_ema = val_accuracy
        else:
            self.val_loss_ema = alpha * val_loss + (1 - alpha) * self.val_loss_ema
            self.val_accuracy_ema = alpha * val_accuracy + (1 - alpha) * self.val_accuracy_ema

    def set_stage_baseline(self, stage_name: str, force_update: bool = False):
        if (stage_name not in self.stage_baselines or force_update) and self.val_loss_ema is not None:
            self.stage_baselines[stage_name] = {
                'loss': self.val_loss_ema,
                'accuracy': self.val_accuracy_ema
            }
            self.current_stage_baseline = self.stage_baselines[stage_name]

    def detect_performance_degradation(self) -> Tuple[bool, str]:
        if (len(self.val_loss_history) < 3 or
                self.current_stage_baseline is None or
                self.val_loss_ema is None):
            return False, "insufficient_data"

        baseline_loss = self.current_stage_baseline['loss']
        baseline_accuracy = self.current_stage_baseline['accuracy']

        loss_increase = (self.val_loss_ema - baseline_loss) / baseline_loss
        if loss_increase > self.config.validation_threshold:
            return True, f"loss_degradation_{loss_increase:.3f}"

        accuracy_decrease = (baseline_accuracy - self.val_accuracy_ema) / baseline_accuracy
        if accuracy_decrease > self.config.validation_threshold:
            return True, f"accuracy_degradation_{accuracy_decrease:.3f}"

        return False, "performance_stable"

    def should_adjust_curriculum(self, step: int) -> Tuple[bool, str, str]:
        if not self.config.enable_validation_gating:
            return False, "none", "gating_disabled"

        if step - self.last_adjustment_step < 1000:
            return False, "none", "too_recent"

        is_degraded, degradation_reason = self.detect_performance_degradation()
        if is_degraded:
            self.consecutive_degradations += 1
            if self.consecutive_degradations >= 3:
                return True, "rollback", f"consecutive_degradations_{degradation_reason}"
            return True, "reduce_intensity", degradation_reason

        self.consecutive_degradations = 0
        if (len(self.val_loss_history) >= 5 and
                self.val_loss_ema < self.current_stage_baseline['loss'] * 0.95):
            return True, "increase_intensity", "performance_exceeding_baseline"

        return False, "none", "no_adjustment_needed"

    def adjust_intensity(self, adjustment_type: str, step: int, reason: str) -> float:
        old_intensity = self.intensity_multiplier
        if adjustment_type == "reduce_intensity":
            self.intensity_multiplier = max(0.1, self.intensity_multiplier * 0.7)
        elif adjustment_type == "increase_intensity":
            self.intensity_multiplier = min(2.0, self.intensity_multiplier * 1.2)
        elif adjustment_type == "rollback":
            self.intensity_multiplier = max(0.1, self.intensity_multiplier * 0.5)
            self.rollback_count += 1

        self.adjustment_history.append((step, adjustment_type, reason, old_intensity, self.intensity_multiplier))
        self.last_adjustment_step = step
        return self.intensity_multiplier

    def get_adjustment_log(self) -> str:
        if not self.adjustment_history:
            return "No curriculum adjustments made."
        recent = self.adjustment_history[-5:]
        lines = ["Recent curriculum adjustments:"]
        for step, adj_type, reason, old_i, new_i in recent:
            lines.append(f"  Step {step}: {adj_type} ({reason}) - intensity {old_i:.3f} -> {new_i:.3f}")
        return "\n".join(lines)


class CurriculumRuntime:
    """
    A thin runtime wrapper that encapsulates stage selection, epsilon smoothing,
    validation gating and rollback behavior. Now with atomic state management.
    """
    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        anm_distance_penalty: float,
        warmup_steps: int,
        use_adversarial_corruption: bool,
    ):
        self.config = curriculum_config
        self.anm_distance_penalty = anm_distance_penalty
        self.warmup_steps = warmup_steps
        self.use_adversarial_corruption = use_adversarial_corruption

        self.vg = ValidationGatedCurriculum(self.config) if self.config is not None else None

        # ATOMIC STATE MANAGEMENT - Centralized training step tracking
        self._state_lock = threading.RLock()  # Reentrant lock for nested calls
        self._atomic_step: int = 0  # Centralized training step counter
        
        # Core state variables - now protected by lock
        self.current_stage_name: Optional[str] = None
        self.stage_transition_step: int = 0
        self.effective_step: int = 0
        self.curriculum_rollback_target: Optional[int] = None
        
        # PARAMETER CACHING - Cache expensive get_params() calculations
        self._param_cache = {}
        self._cache_valid_step = -1
        
        # COMPLETE STATE SNAPSHOT for rollback - Store complete state for atomic rollback
        self._state_snapshots = {}  # step -> complete state dict
        self._max_snapshots = 100  # Keep last 100 snapshots for memory efficiency

    def _create_state_snapshot(self) -> dict:
        """Create a complete state snapshot for rollback purposes."""
        return {
            'atomic_step': self._atomic_step,
            'current_stage_name': self.current_stage_name,
            'stage_transition_step': self.stage_transition_step,
            'effective_step': self.effective_step,
            'curriculum_rollback_target': self.curriculum_rollback_target,
            'vg_intensity': self.vg.intensity_multiplier if self.vg else 1.0,
            'vg_consecutive_degradations': self.vg.consecutive_degradations if self.vg else 0,
            'vg_last_adjustment_step': self.vg.last_adjustment_step if self.vg else 0,
            'vg_rollback_count': self.vg.rollback_count if self.vg else 0,
        }

    def _restore_state_snapshot(self, snapshot: dict):
        """Restore complete state from snapshot atomically."""
        self._atomic_step = snapshot['atomic_step']
        self.current_stage_name = snapshot['current_stage_name']
        self.stage_transition_step = snapshot['stage_transition_step']
        self.effective_step = snapshot['effective_step']
        self.curriculum_rollback_target = snapshot['curriculum_rollback_target']
        
        if self.vg is not None:
            self.vg.intensity_multiplier = snapshot['vg_intensity']
            self.vg.consecutive_degradations = snapshot['vg_consecutive_degradations']
            self.vg.last_adjustment_step = snapshot['vg_last_adjustment_step']
            self.vg.rollback_count = snapshot['vg_rollback_count']

    def update_training_step(self, step: int):
        """Centralized training step management with atomic updates."""
        with self._state_lock:
            old_step = self._atomic_step
            self._atomic_step = step
            
            # Invalidate cache if step changed significantly
            if abs(step - self._cache_valid_step) > 1:
                self._param_cache.clear()
            
            # Save state snapshot periodically for rollback
            if step % 10 == 0:  # Save every 10 steps
                self._state_snapshots[step] = self._create_state_snapshot()
                
                # Cleanup old snapshots
                if len(self._state_snapshots) > self._max_snapshots:
                    oldest_step = min(self._state_snapshots.keys())
                    del self._state_snapshots[oldest_step]
            
            return old_step

    def get_current_training_step(self) -> int:
        """Thread-safe getter for current training step."""
        with self._state_lock:
            return self._atomic_step

    def get_params(self, step: int, base_distance_penalty: float) -> Tuple[StageSnapshot, float]:
        """
        Returns (stage snapshot, adjusted_epsilon).
        Now with parameter caching and atomic state management.
        """
        # Update centralized training step atomically
        self.update_training_step(step)
        
        with self._state_lock:
            # Check cache first
            cache_key = (step, base_distance_penalty)
            if (cache_key in self._param_cache and 
                step == self._cache_valid_step):
                return self._param_cache[cache_key]

            # ATOMIC STAGE TRANSITION - All state updates happen atomically
            effective_step = self._compute_effective_step(step)
            stage = self.config.get_stage(effective_step)
            base_epsilon = self.config.get_smooth_epsilon(effective_step, base_distance_penalty * 10)

            intensity_multiplier = 1.0
            if self.vg is not None:
                intensity_multiplier = self.vg.intensity_multiplier
                should_adjust, adj_type, reason = self.vg.should_adjust_curriculum(step)
                if should_adjust:
                    if adj_type == "rollback":
                        # Atomic rollback with complete state restoration
                        self._atomic_rollback_curriculum(step)
                        effective_step = self.effective_step
                        stage = self.config.get_stage(effective_step)
                        base_epsilon = self.config.get_smooth_epsilon(effective_step, base_distance_penalty * 10)
                    else:
                        old_i = self.vg.intensity_multiplier
                        new_i = self.vg.adjust_intensity(adj_type, step, reason)
                        if old_i != new_i:
                            print(f"[Curriculum Gating] Step {step}: {adj_type} - intensity {old_i:.3f} -> {new_i:.3f} (reason: {reason})")
                        intensity_multiplier = new_i

            adjusted_epsilon = base_epsilon * intensity_multiplier

            # Create stage snapshot
            s = StageSnapshot(
                name=stage.name,
                clean_ratio=stage.clean_ratio,
                adversarial_ratio=stage.adversarial_ratio,
                gaussian_ratio=stage.gaussian_ratio,
                hard_negative_ratio=stage.hard_negative_ratio,
                epsilon_multiplier=stage.epsilon_multiplier,
                temperature=stage.temperature,
            )
            
            # Apply intensity adjustments
            if intensity_multiplier < 1.0:
                clean_boost = (1.0 - intensity_multiplier) * 0.3
                s.clean_ratio = min(1.0, s.clean_ratio + clean_boost)
                s.adversarial_ratio = max(0.0, s.adversarial_ratio - clean_boost * 0.4)
                s.gaussian_ratio = max(0.0, s.gaussian_ratio - clean_boost * 0.2)
                s.hard_negative_ratio = max(0.0, s.hard_negative_ratio - clean_boost * 0.4)

            # ATOMIC STAGE TRANSITION - Handle stage transitions atomically
            if self.current_stage_name != s.name:
                self._atomic_stage_transition(s.name, step, effective_step, intensity_multiplier)

            # Store effective step in runtime
            self.effective_step = effective_step
            
            # Cache the result
            result = (s, adjusted_epsilon)
            self._param_cache[cache_key] = result
            self._cache_valid_step = step
            
            return result

    def _compute_effective_step(self, step: int) -> int:
        """Compute effective step considering rollback state."""
        effective_step = step
        if self.curriculum_rollback_target is not None:
            effective_step = min(step, self.curriculum_rollback_target + (step - self.stage_transition_step))
            if step >= self.curriculum_rollback_target + self.config.rollback_steps:
                self.curriculum_rollback_target = None
        return effective_step

    def _atomic_stage_transition(self, new_stage_name: str, step: int, effective_step: int, intensity_multiplier: float):
        """Perform stage transition atomically to prevent race conditions."""
        old_stage = self.current_stage_name
        
        # All stage transition updates happen atomically
        self.current_stage_name = new_stage_name
        self.stage_transition_step = step
        
        if self.vg is not None:
            self.vg.set_stage_baseline(new_stage_name, force_update=True)
            
        print(f"[Curriculum] Step {step}: Stage transition {old_stage} -> {new_stage_name} (effective_step: {effective_step}, intensity: {intensity_multiplier:.3f})")

    def update_validation_performance(self, training_step: int, val_loss: float, val_accuracy: Optional[float]):
        """Thread-safe validation performance updates."""
        if self.vg is None:
            return
        if val_accuracy is None:
            val_accuracy = max(0.0, 1.0 - val_loss)
            
        with self._state_lock:
            self.vg.update_validation_metrics(val_loss, val_accuracy, training_step)
            if self.current_stage_name is not None:
                self.vg.set_stage_baseline(self.current_stage_name)

    def should_adjust_curriculum(self, step: int) -> Tuple[bool, str, str]:
        with self._state_lock:
            if self.vg is None:
                return (False, "none", "no_validation_gating")
            return self.vg.should_adjust_curriculum(step)

    def adjust_curriculum_intensity(self, step: int, adjustment_type: Optional[str], reason: Optional[str]) -> float:
        with self._state_lock:
            if self.vg is None:
                return 1.0
            if adjustment_type is None:
                should_adjust, adjustment_type, reason = self.should_adjust_curriculum(step)
                if not should_adjust:
                    return self.vg.intensity_multiplier
            return self.vg.adjust_intensity(adjustment_type, step, reason)

    def rollback_curriculum(self, step: int, rollback_steps: Optional[int] = None) -> int:
        """Public interface for rollback - delegates to atomic implementation."""
        with self._state_lock:
            return self._atomic_rollback_curriculum(step, rollback_steps)

    def _atomic_rollback_curriculum(self, step: int, rollback_steps: Optional[int] = None) -> int:
        """ATOMIC ROLLBACK - Complete state rollback with all related variables."""
        if self.vg is None:
            return step
            
        if rollback_steps is None:
            rollback_steps = self.config.rollback_steps
            
        rollback_target_step = max(0, step - rollback_steps)
        
        # Find closest state snapshot for atomic rollback
        closest_snapshot_step = None
        for snapshot_step in sorted(self._state_snapshots.keys(), reverse=True):
            if snapshot_step <= rollback_target_step:
                closest_snapshot_step = snapshot_step
                break
        
        if closest_snapshot_step is not None:
            # Restore complete state atomically from snapshot
            snapshot = self._state_snapshots[closest_snapshot_step]
            self._restore_state_snapshot(snapshot)
            print(f"[Curriculum Rollback] Step {step}: Restored complete state from snapshot at step {closest_snapshot_step}")
        else:
            # Fallback to manual rollback if no snapshot available
            self.curriculum_rollback_target = rollback_target_step
            self.effective_step = rollback_target_step
            self.vg.intensity_multiplier = 1.0
            self.vg.consecutive_degradations = 0
            print(f"[Curriculum Rollback] Step {step}: Manual rollback {rollback_steps} steps to effective step {self.effective_step}")
        
        # Clear parameter cache to force recalculation
        self._param_cache.clear()
        
        return self.effective_step

    def get_validation_gating_summary(self) -> str:
        with self._state_lock:
            if self.vg is None:
                return "Validation gating: Disabled"
            lines = [
                f"Validation gating: {'Enabled' if self.config.enable_validation_gating else 'Disabled'}",
                f"Current intensity: {self.vg.intensity_multiplier:.3f}",
                f"Validation loss EMA: {self.vg.val_loss_ema:.4f}" if self.vg.val_loss_ema else "Validation loss EMA: Not available",
                f"Validation accuracy EMA: {self.vg.val_accuracy_ema:.4f}" if self.vg.val_accuracy_ema else "Validation accuracy EMA: Not available",
                f"Consecutive degradations: {self.vg.consecutive_degradations}",
                f"Total rollbacks: {self.vg.rollback_count}",
                f"Centralized training step: {self._atomic_step}",
                f"Parameter cache size: {len(self._param_cache)}",
                f"State snapshots: {len(self._state_snapshots)}"
            ]
            if self.curriculum_rollback_target is not None:
                lines.append(f"Active rollback: target step {self.curriculum_rollback_target}")
            adjustment_log = self.vg.get_adjustment_log()
            if adjustment_log != "No curriculum adjustments made.":
                lines.append(adjustment_log)
            return "\n".join(lines)

    def reset(self):
        """Reset all state atomically."""
        with self._state_lock:
            if self.vg is not None:
                self.vg = ValidationGatedCurriculum(self.config)
            self._atomic_step = 0
            self.effective_step = 0
            self.curriculum_rollback_target = None
            self.current_stage_name = None
            self.stage_transition_step = 0
            self._param_cache.clear()
            self._state_snapshots.clear()
            self._cache_valid_step = -1