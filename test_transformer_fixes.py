#!/usr/bin/env python3
"""
Test script to verify transformer fixes work correctly.
Tests all three condition types and ensures no errors occur.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_transformer_import():
    """Test that we can import the transformer module."""
    print("Testing transformer import...")
    try:
        from cfm_flowmp.models.transformer import FlowMPTransformer
        print("✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_transformer_initialization():
    """Test that we can create transformer instances."""
    print("\nTesting transformer initialization...")
    try:
        from cfm_flowmp.models.transformer import create_flowmp_transformer
        
        # Test all variants
        for variant in ["small", "base", "large"]:
            model = create_flowmp_transformer(variant=variant)
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {variant} variant: {params:,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False

def test_condition_types():
    """Test all three condition injection types."""
    print("\nTesting condition types...")
    try:
        from cfm_flowmp.models.transformer import FlowMPTransformer
        
        # Test parameters
        batch_size = 2
        seq_len = 32
        state_dim = 2
        
        for condition_type in ["adaln", "cross_attention", "token"]:
            print(f"  Testing {condition_type} mode...")
            
            model = FlowMPTransformer(
                state_dim=state_dim,
                max_seq_len=64,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                condition_type=condition_type,
            )
            model.eval()
            
            # Create dummy inputs
            x_t = torch.randn(batch_size, seq_len, 6)  # pos(2) + vel(2) + acc(2)
            t = torch.rand(batch_size)
            start_pos = torch.randn(batch_size, state_dim)
            goal_pos = torch.randn(batch_size, state_dim)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_t, t, start_pos, goal_pos)
            
            # Check output shape
            expected_shape = (batch_size, seq_len, 6)
            if output.shape != expected_shape:
                print(f"    ✗ Wrong output shape: {output.shape} vs {expected_shape}")
                return False
            
            print(f"    ✓ {condition_type} mode works correctly")
        
        return True
    except Exception as e:
        print(f"  ✗ Condition type test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dropout_parameter():
    """Test that dropout parameter is correctly stored and used."""
    print("\nTesting dropout parameter fix...")
    try:
        from cfm_flowmp.models.transformer import MultiHeadSelfAttention
        
        dropout_rate = 0.1
        attn = MultiHeadSelfAttention(
            hidden_dim=128,
            num_heads=4,
            dropout=dropout_rate,
        )
        
        # Check that dropout is stored correctly
        if not hasattr(attn, 'dropout'):
            print("  ✗ Missing dropout attribute")
            return False
        
        if attn.dropout != dropout_rate:
            print(f"  ✗ Wrong dropout value: {attn.dropout} vs {dropout_rate}")
            return False
        
        print("  ✓ Dropout parameter correctly stored")
        
        # Test forward pass with dropout
        attn.train()
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = attn(x)
        
        if out.shape != x.shape:
            print(f"  ✗ Wrong output shape: {out.shape} vs {x.shape}")
            return False
        
        print("  ✓ Forward pass with dropout works")
        return True
        
    except Exception as e:
        print(f"  ✗ Dropout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_contiguous_operations():
    """Test that contiguous operations work correctly."""
    print("\nTesting contiguous operations...")
    try:
        from cfm_flowmp.models.transformer import CrossAttention, MultiHeadSelfAttention
        
        # Test MultiHeadSelfAttention
        attn = MultiHeadSelfAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        
        with torch.no_grad():
            out = attn(x)
        
        if not out.is_contiguous():
            print("  ✗ MultiHeadSelfAttention output not contiguous")
            return False
        print("  ✓ MultiHeadSelfAttention output is contiguous")
        
        # Test CrossAttention
        cross_attn = CrossAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        cond_tokens = torch.randn(2, 4, 128)
        
        with torch.no_grad():
            out = cross_attn(x, cond_tokens)
        
        if not out.is_contiguous():
            print("  ✗ CrossAttention output not contiguous")
            return False
        print("  ✓ CrossAttention output is contiguous")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Contiguous test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\nTesting gradient flow...")
    try:
        from cfm_flowmp.models.transformer import FlowMPTransformer
        
        model = FlowMPTransformer(
            state_dim=2,
            max_seq_len=64,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        
        # Create dummy inputs
        x_t = torch.randn(2, 16, 6, requires_grad=True)
        t = torch.rand(2, requires_grad=True)
        start_pos = torch.randn(2, 2, requires_grad=True)
        goal_pos = torch.randn(2, 2, requires_grad=True)
        
        # Forward pass
        output = model(x_t, t, start_pos, goal_pos)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        if x_t.grad is None:
            print("  ✗ No gradient for x_t")
            return False
        
        if t.grad is None:
            print("  ✗ No gradient for t")
            return False
        
        # Check that model parameters have gradients
        no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
        if no_grad_params:
            print(f"  ✗ Parameters without gradients: {no_grad_params[:5]}")
            return False
        
        print("  ✓ Gradients flow correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Transformer Network Structure Tests")
    print("=" * 60)
    
    tests = [
        test_transformer_import,
        test_transformer_initialization,
        test_condition_types,
        test_dropout_parameter,
        test_contiguous_operations,
        test_gradient_flow,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
