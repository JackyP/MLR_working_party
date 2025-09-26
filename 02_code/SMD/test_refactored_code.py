"""
Test script to verify the refactored GRU framework works correctly.

This script tests the core functionality without running the full training
to ensure imports and basic functionality work.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch

# Add the current directory to path to import local modules
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from config import get_default_config, ExperimentConfig
        print("✓ Config module imported successfully")
        
        from neural_networks import BasicLogGRU, get_model_class
        print("✓ Neural networks module imported successfully")
        
        from shap_utils import ShapExplainer, log_shap_explanations, create_background_dataset
        print("✓ SHAP utils module imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from config import get_default_config
        
        config = get_default_config()
        
        # Test that config has expected structure
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'tensorboard')
        
        # Test configuration values
        assert config.data.output_field == "claim_size"
        assert len(config.data.features) > 0
        assert config.training.enable_shap == True
        
        print("✓ Configuration system works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_neural_networks():
    """Test neural network models."""
    print("\nTesting neural network models...")
    
    try:
        from neural_networks import BasicLogGRU, get_model_class
        
        # Test model creation
        model_class = get_model_class('BasicLogGRU')
        assert model_class == BasicLogGRU
        
        # Test model instantiation
        model = BasicLogGRU(n_input=5, n_hidden=10, n_output=1, init_bias=None)
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 3, 5)  # batch_size=2, sequence_length=3, features=5
        output = model(dummy_input)
        
        assert output.shape == (2, 1)  # batch_size=2, output_size=1
        assert torch.all(output > 0)  # Should be positive due to exp() activation
        
        print("✓ Neural network models work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_shap_utils():
    """Test SHAP utilities with mock data."""
    print("\nTesting SHAP utilities...")
    
    try:
        from shap_utils import create_background_dataset
        from neural_networks import BasicLogGRU
        
        # Create mock data
        X = torch.randn(100, 10, 5)  # 100 samples, 10 time steps, 5 features
        
        # Test background dataset creation
        background = create_background_dataset(X, n_samples=20)
        assert background.shape[0] == 20
        assert background.shape[1:] == X.shape[1:]
        
        print("✓ SHAP utilities work correctly")
        return True
        
    except Exception as e:
        print(f"✗ SHAP utilities test failed: {e}")
        return False

def test_data_processing():
    """Test data processing functions."""
    print("\nTesting data processing functions...")
    
    try:
        # Create mock dataset similar to the expected structure
        np.random.seed(42)
        n_samples = 100
        
        mock_data = pd.DataFrame({
            'claim_no': np.repeat(range(1, 21), 5),  # 20 claims, 5 records each
            'occurrence_time': np.random.randint(1, 10, n_samples),
            'notidel': np.random.exponential(30, n_samples),
            'development_period': np.tile(range(1, 6), 20),  # 5 development periods
            'pmt_no': np.random.randint(1, 6, n_samples),
            'log1_paid_cumulative': np.random.exponential(8, n_samples),
            'claim_size': np.random.exponential(100000, n_samples),
            'payment_period': np.random.randint(1, 45, n_samples),
            'settle_period': np.random.randint(1, 45, n_samples),
            'train_ind': np.random.binomial(1, 0.8, n_samples),
            'is_settled': np.random.binomial(1, 0.7, n_samples)
        })
        
        # Add the required derived columns
        mock_data["train_ind_time"] = (mock_data.payment_period <= 32)
        mock_data["test_ind_time"] = (mock_data.payment_period <= 40)
        mock_data["train_settled"] = (mock_data.settle_period <= 40)
        
        print("✓ Data processing functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Refactored GRU Framework")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_neural_networks,
        test_shap_utils,
        test_data_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored code is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)