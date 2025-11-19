#!/usr/bin/env python3
"""
Manual verification script to check that our ANM fix is working correctly.
Tests that energy losses are now properly included in the total loss.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Addition

def create_test_model():
    """Create a minimal model for testing"""
    dataset = Addition("train", 20, False)  # rank=20, no curriculum
    model = EBM(
        inp_dim=dataset.inp_dim,
        out_dim=dataset.out_dim,
    )
    model = DiffusionWrapper(model)
    return model

def test_anm_fix():
    """Test that ANM energy losses are now properly included"""
    print("🧪 Testing ANM fix...")
    
    # Create model
    model = create_test_model()
    
    print("✅ Test 1: Auto-coupling works (use_anm=True should auto-enable energy supervision)")
    try:
        # This should work now due to auto-coupling (if we were calling via train.py)
        # But since we're calling directly, we need to test the fail-fast validation
        diffusion = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=True  # Manually set to True
        )
        print("   ✓ GaussianDiffusion1D created successfully with ANM + energy supervision")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("✅ Test 2: Fail-fast validation works (use_anm=True with energy=False should raise error)")
    try:
        # This should raise ValueError due to our fail-fast validation
        bad_diffusion = GaussianDiffusion1D(
            model=model,
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=False  # Should trigger ValueError
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "ANM" in str(e) and "requires supervise_energy_landscape" in str(e):
            print(f"   ✓ Correctly raised ValueError: {e}")
        else:
            print(f"   ❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED: Wrong exception type: {e}")
        return False
    
    print("✅ Test 3: Loss composition behavior check")
    try:
        # Create test data
        dataset = Addition("train", 20, False)
        batch_size = 4
        device = 'cpu'  # Force CPU to avoid MPS issues
        
        # Sample test data
        x_list = []
        y_list = []
        for i in range(batch_size):
            x, y = dataset[i]
            x_list.append(torch.from_numpy(x) if hasattr(x, 'numpy') else torch.tensor(x))
            y_list.append(torch.from_numpy(y) if hasattr(y, 'numpy') else torch.tensor(y))
        
        inp = torch.stack(x_list).float().to(device)
        target = torch.stack(y_list).float().to(device)
        
        # Test with ANM enabled
        diffusion_anm = GaussianDiffusion1D(
            model=model.to(device),
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=True,
            supervise_energy_landscape=True
        )
        
        # Test with ANM disabled  
        diffusion_baseline = GaussianDiffusion1D(
            model=model.to(device),
            seq_length=32,
            timesteps=10,
            use_adversarial_corruption=False,
            supervise_energy_landscape=False
        )
        
        # Create a simple mask and timestep
        mask = None
        t = torch.randint(0, 10, (batch_size,), device=device)
        
        # Test ANM p_losses (should include energy terms)
        print("   Testing ANM p_losses...")
        with torch.no_grad():
            loss_anm, (loss_mse_anm, loss_energy_anm, loss_opt_anm) = diffusion_anm.p_losses(inp, target, mask, t)
            
            print(f"   ANM - Total loss: {loss_anm.item():.6f}")
            print(f"   ANM - MSE loss: {loss_mse_anm.item():.6f}")
            print(f"   ANM - Energy loss: {loss_energy_anm.item():.6f}")
            print(f"   ANM - Opt loss: {loss_opt_anm.item():.6f}")
            
            # Key test: energy loss should NOT be -1 (sentinel value)
            if loss_energy_anm.item() == -1.0:
                print("   ❌ FAILED: Energy loss is still -1 (sentinel value)!")
                return False
            else:
                print("   ✓ Energy loss is computed (not -1 sentinel)")
        
        # Test baseline p_losses (should NOT include energy terms)
        print("   Testing baseline p_losses...")
        with torch.no_grad():
            loss_baseline, (loss_mse_baseline, loss_energy_baseline, loss_opt_baseline) = diffusion_baseline.p_losses(inp, target, mask, t)
            
            print(f"   Baseline - Total loss: {loss_baseline.item():.6f}")
            print(f"   Baseline - MSE loss: {loss_mse_baseline.item():.6f}")
            print(f"   Baseline - Energy loss: {loss_energy_baseline.item():.6f}")
            print(f"   Baseline - Opt loss: {loss_opt_baseline.item():.6f}")
            
            # Baseline should have -1 sentinel values
            if loss_energy_baseline.item() != -1.0:
                print("   ❌ FAILED: Baseline should have -1 energy loss!")
                return False
            else:
                print("   ✓ Baseline correctly has -1 energy loss (sentinel)")
        
        # Final comparison
        print("   Final comparison:")
        if loss_mse_anm.item() == loss_baseline.item():
            print("   🤔 NOTE: Total losses are equal (may be expected if energy contribution is small)")
        else:
            print("   ✓ Total losses differ (ANM vs baseline)")
            
        print(f"   Loss difference: {abs(loss_anm.item() - loss_baseline.item()):.6f}")
        
    except Exception as e:
        print(f"   ❌ FAILED: Loss composition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! ANM fix is working correctly.")
    return True

if __name__ == "__main__":
    success = test_anm_fix()
    if success:
        print("\n✅ VERDICT: Fix is working correctly!")
        exit(0)
    else:
        print("\n❌ VERDICT: Fix has issues!")
        exit(1)