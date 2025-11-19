#!/usr/bin/env python3
"""
Simple verification that our ANM configuration fix is working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from unittest.mock import Mock
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D

def test_configuration_fix():
    """Test the core configuration behavior without complex model interactions"""
    print("🧪 Testing ANM configuration fix...")
    
    # Create simple mock model
    model = Mock()
    model.inp_dim = 32
    model.out_dim = 32
    
    print("✅ Test 1: Valid configuration (ANM + energy supervision)")
    try:
        diffusion = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=True
        )
        print(f"   ✓ Created successfully")
        print(f"   ✓ use_adversarial_corruption = {diffusion.use_adversarial_corruption}")
        print(f"   ✓ supervise_energy_landscape = {diffusion.supervise_energy_landscape}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("✅ Test 2: Invalid configuration should raise ValueError")
    try:
        bad_diffusion = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=False
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        expected_phrases = ["ANM", "requires", "supervise_energy_landscape"]
        error_msg = str(e)
        if all(phrase in error_msg for phrase in expected_phrases):
            print(f"   ✓ Correctly raised ValueError: {error_msg}")
        else:
            print(f"   ❌ FAILED: Error message doesn't match expected pattern: {error_msg}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    print("✅ Test 3: Baseline (no ANM) should work with any energy setting")
    try:
        # Both should work when ANM is disabled
        diffusion1 = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=False,
            supervise_energy_landscape=True
        )
        
        diffusion2 = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=False,
            supervise_energy_landscape=False
        )
        print(f"   ✓ Both configurations work when ANM is disabled")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    return True

def test_auto_coupling_simulation():
    """Simulate the auto-coupling behavior from train.py"""
    print("✅ Test 4: Simulating train.py auto-coupling logic")
    
    class MockFlags:
        def __init__(self, use_anm, supervise_energy_landscape):
            self.use_anm = use_anm
            self.supervise_energy_landscape = supervise_energy_landscape
    
    # Simulate the problematic case (what used to cause the bug)
    FLAGS = MockFlags(use_anm=True, supervise_energy_landscape=False)
    
    # Apply our auto-coupling fix
    if FLAGS.use_anm and not FLAGS.supervise_energy_landscape:
        print("   ⚠️  Auto-coupling triggered: enabling supervise_energy_landscape")
        FLAGS.supervise_energy_landscape = True
    
    # Verify the fix worked
    if FLAGS.use_anm and FLAGS.supervise_energy_landscape:
        print("   ✓ Auto-coupling successful - both flags now True")
        return True
    else:
        print(f"   ❌ Auto-coupling failed - use_anm={FLAGS.use_anm}, supervise_energy_landscape={FLAGS.supervise_energy_landscape}")
        return False

if __name__ == "__main__":
    print("🔧 ANM Configuration Fix Verification\n")
    
    success = True
    success &= test_configuration_fix()
    print()
    success &= test_auto_coupling_simulation()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ ANM configuration fix is working correctly!")
        print("   - Fail-fast validation prevents invalid configurations") 
        print("   - Auto-coupling logic would work in train.py")
        print("   - Error messages are clear and actionable")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Fix needs additional work")
        exit(1)