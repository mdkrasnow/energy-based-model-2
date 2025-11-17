# File: /Users/mkrasnow/Desktop/energy-based-model-2/phase1_configs.py
# Phase 1 Statistical Viability Configuration Definitions
# Defines the 4 critical configurations for testing ANM effectiveness
from typing import Dict, Any, List, NamedTuple
from dataclasses import dataclass

@dataclass
class Phase1Config:
    """Configuration for a single Phase 1 experiment"""
    name: str
    description: str
    use_anm: bool
    anm_adversarial_steps: int = None
    anm_temperature: float = 1.0
    anm_clean_ratio: float = 0.1
    anm_adversarial_ratio: float = 0.8  
    anm_gaussian_ratio: float = 0.1
    use_random_noise: bool = False
    random_noise_scale: float = None
    random_noise_type: str = "gaussian"
    
    def to_train_args(self) -> List[str]:
        """Convert configuration to train.py command line arguments"""
        args = []
        
        if self.use_anm:
            args.extend([
                '--use-anm',
                '--anm-adversarial-steps', str(self.anm_adversarial_steps),
                '--anm-temperature', str(self.anm_temperature),
                '--anm-clean-ratio', str(self.anm_clean_ratio),
                '--anm-adversarial-ratio', str(self.anm_adversarial_ratio),
                '--anm-gaussian-ratio', str(self.anm_gaussian_ratio)
            ])
        return args
    
    def get_result_suffix(self) -> str:
        """Get unique suffix for result directory naming"""
        if self.use_anm:
            return f"anm_steps{self.anm_adversarial_steps}"
        else:
            return "baseline"

# Phase 1 Core Configurations: Exactly 4 configs for statistical testing
PHASE1_CONFIGURATIONS = {
    "ired_baseline": Phase1Config(
        name="ired_baseline",
        description="Standard IRED training without any adversarial negative mining",
        use_anm=False,
        use_random_noise=False
    ),
    
    "anm_best": Phase1Config(
        name="anm_best", 
        description="ANM with best-known hyperparameters from previous experiments",
        use_anm=True,
        anm_adversarial_steps=5,
        anm_temperature=1.0,
        anm_clean_ratio=0.1,
        anm_adversarial_ratio=0.8,
        anm_gaussian_ratio=0.1
    ),
    
    "anm_extreme": Phase1Config(
        name="anm_extreme",
        description="ANM with extreme simplification - single step",
        use_anm=True, 
        anm_adversarial_steps=1,  # Minimal steps
        anm_temperature=1.0,
        anm_clean_ratio=0.1,
        anm_adversarial_ratio=0.8,
        anm_gaussian_ratio=0.1
    )
}

# Validation: Ensure exactly 3 configurations now that random noise is removed
assert len(PHASE1_CONFIGURATIONS) == 3, f"Phase 1 requires exactly 3 configs, got {len(PHASE1_CONFIGURATIONS)}"

def get_phase1_config_names() -> List[str]:
    """Get list of all Phase 1 configuration names"""
    return list(PHASE1_CONFIGURATIONS.keys())

def get_phase1_config(name: str) -> Phase1Config:
    """Get specific Phase 1 configuration by name"""
    if name not in PHASE1_CONFIGURATIONS:
        raise ValueError(f"Unknown Phase 1 config: {name}. Available: {list(PHASE1_CONFIGURATIONS.keys())}")
    return PHASE1_CONFIGURATIONS[name]

def get_all_phase1_configs() -> List[Phase1Config]:
    """Get all Phase 1 configurations as a list"""
    return list(PHASE1_CONFIGURATIONS.values())

# Phase 1 Experimental Design Parameters
PHASE1_EXPERIMENTAL_DESIGN = {
    "num_configs": 3,
    "seeds_per_config": 5,
    "total_experiments": 15,  # 3 × 5 = 15 training runs
    "train_steps_per_experiment": 1000,  # Reduced for Phase 1 speed
    "statistical_alpha": 0.05,
    "bonferroni_alpha": 0.0167,  # 0.05 / 3 configs
    "effect_size_threshold": 0.3,  # Cohen's d > 0.3 (small-medium effect)
    "random_seeds": [42, 123, 456, 789, 999]  # Fixed seeds for reproducibility
}

def generate_experiment_matrix() -> List[Dict[str, Any]]:
    """
    Generate complete experimental matrix for Phase 1
    
    Returns:
        List of experiment specifications with config and seed combinations
    """
    experiments = []
    
    for seed in PHASE1_EXPERIMENTAL_DESIGN["random_seeds"]:
        for config_name, config in PHASE1_CONFIGURATIONS.items():
            experiment = {
                "config_name": config_name,
                "config": config,
                "seed": seed,
                "train_steps": PHASE1_EXPERIMENTAL_DESIGN["train_steps_per_experiment"],
                "experiment_id": f"{config_name}_seed{seed}"
            }
            experiments.append(experiment)
    
    return experiments

# Phase 1 Success Criteria
PHASE1_SUCCESS_CRITERIA = {
    "go_criteria": {
        "statistical_significance": "p_corrected < 0.0125", 
        "effect_size": "|Cohen's d| > 0.3",
        "logical_operator": "AND",
        "minimum_configs": 1,  # ANY config meeting criteria triggers GO
        "description": "ANY configuration shows both statistical significance AND meaningful effect size"
    },
    "no_go_criteria": {
        "condition": "ALL configs fail statistical significance OR effect size thresholds",
        "action": "STOP Phase 1, document negative results, pivot to alternatives",
        "time_limit": "5 days maximum",
        "compute_limit": "20 training runs maximum"
    }
}

# Phase 1 Configuration Rationale
PHASE1_CONFIG_RATIONALE = {
    "ired_baseline": {
        "purpose": "Reference condition for statistical comparison",
        "expectation": "Baseline performance to measure improvements against",
        "critical_finding": "If other configs don't beat this, ANM provides no benefit"
    },
    "anm_best": {
        "purpose": "Test ANM with previously optimized hyperparameters",  
        "expectation": "Should show best ANM performance if any exists",
        "critical_finding": "If this fails, ANM optimization is fundamentally limited"
    },
    "anm_extreme": {
        "purpose": "Test minimal ANM to isolate core mechanism",
        "expectation": "Pure energy maximization with minimal computational cost",
        "critical_finding": "If this succeeds, single-step ANM is sufficient"
    }
}

def print_phase1_experimental_design():
    """Print comprehensive overview of Phase 1 experimental design"""
    print("=" * 80)
    print("PHASE 1 STATISTICAL VIABILITY EXPERIMENTAL DESIGN")
    print("=" * 80)
    print()
    
    print("📊 EXPERIMENTAL MATRIX:")
    print(f"  • Configurations: {PHASE1_EXPERIMENTAL_DESIGN['num_configs']}")
    print(f"  • Seeds per config: {PHASE1_EXPERIMENTAL_DESIGN['seeds_per_config']}")
    print(f"  • Total experiments: {PHASE1_EXPERIMENTAL_DESIGN['total_experiments']}")
    print(f"  • Train steps per experiment: {PHASE1_EXPERIMENTAL_DESIGN['train_steps_per_experiment']:,}")
    print()
    
    print("🧪 CONFIGURATIONS TESTED:")
    for i, (name, config) in enumerate(PHASE1_CONFIGURATIONS.items(), 1):
        print(f"  {i}. {config.name.upper()}: {config.description}")
        if config.use_anm:
            print(f"     → steps={config.anm_adversarial_steps}")
        print()
    
    print("📈 STATISTICAL CRITERIA:")
    print(f"  • Significance threshold: p < {PHASE1_EXPERIMENTAL_DESIGN['bonferroni_alpha']} (Bonferroni corrected)")
    print(f"  • Effect size threshold: |Cohen's d| > {PHASE1_EXPERIMENTAL_DESIGN['effect_size_threshold']}")
    print(f"  • Go decision: ANY config meeting BOTH criteria")
    print(f"  • No-go decision: ALL configs fail statistical significance OR effect size")
    print()
    
    print("🎯 EXPECTED OUTCOMES:")
    for name, rationale in PHASE1_CONFIG_RATIONALE.items():
        print(f"  • {name.upper()}: {rationale['critical_finding']}")
    print()
    
    print("⏱️  TIMELINE:")
    print(f"  • Duration: 5 days maximum")
    print(f"  • Compute: ~3-4 hours per experiment × 20 = 60-80 hours total")
    print(f"  • Decision: Go/No-go within 1 week")
    print()
    
    print("=" * 80)

def validate_phase1_configs():
    """Validate Phase 1 configuration consistency and completeness"""
    print("🔍 Validating Phase 1 configurations...")
    
    # Check required configurations exist
    required_configs = ["ired_baseline", "anm_best", "anm_extreme"]
    for required in required_configs:
        assert required in PHASE1_CONFIGURATIONS, f"Missing required config: {required}"
    
    # Validate ANM configurations
    anm_configs = [c for c in PHASE1_CONFIGURATIONS.values() if c.use_anm]
    assert len(anm_configs) == 2, f"Expected 2 ANM configs, got {len(anm_configs)}"
    
    for config in anm_configs:
        assert config.anm_adversarial_steps is not None, f"ANM config {config.name} missing steps"
    
    # Validate baseline configuration
    baseline = PHASE1_CONFIGURATIONS["ired_baseline"]
    assert not baseline.use_anm, "Baseline config should not use ANM"
    assert not baseline.use_random_noise, "Baseline config should not use random noise"
    
    print("✓ All Phase 1 configurations validated successfully")

if __name__ == "__main__":
    validate_phase1_configs()
    print_phase1_experimental_design()
    
    print("\n🧪 EXPERIMENT MATRIX PREVIEW:")
    experiments = generate_experiment_matrix()
    print(f"Generated {len(experiments)} total experiments:")
    for exp in experiments[:8]:  # Show first 8 experiments
        print(f"  • {exp['experiment_id']}: {exp['config'].description}")
    print(f"  ... and {len(experiments) - 8} more experiments")
    print()
    print(f"✓ Phase 1 configuration system ready for execution (random noise removed)")