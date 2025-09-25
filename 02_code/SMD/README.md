# GRU Framework for Claims Reserving - Refactored

This directory contains the refactored GRU framework for claims reserving with improved structure and SHAP explainability features.

## Files Overview

### Main Scripts
- `GRU_framework_draft.py` - Main execution script (refactored)
- `test_refactored_code.py` - Test script to verify functionality

### Modules
- `config.py` - Centralized configuration management
- `neural_networks.py` - Neural network model architectures
- `shap_utils.py` - SHAP explainability utilities

## Key Improvements

### 1. SHAP Integration
- **Training-time explanations**: SHAP values are logged to TensorBoard during training at configurable intervals
- **Post-training analysis**: Comprehensive SHAP explanations after model training
- **Multiple visualization types**: Feature importance, summary plots, waterfall plots

### 2. Code Organization
- **Modular structure**: Separated concerns into focused modules
- **Configuration management**: Centralized parameter management
- **Improved readability**: Better structure and documentation

### 3. Enhanced TensorBoard Outputs
- **SHAP visualizations**: Feature importance and explanation plots
- **Organized logging**: Better structured metrics and plots
- **Configurable frequency**: Control how often SHAP explanations are generated

## Configuration

The `config.py` module provides a centralized way to manage all parameters:

```python
from config import get_default_config

config = get_default_config()
config.training.enable_shap = True  # Enable SHAP explanations
config.training.shap_log_frequency = 500  # Log SHAP every 500 epochs
config.model.n_hidden = 32  # Hidden layer size
config.training.nn_iter = 2001  # Training iterations
```

## SHAP Features

### During Training
- SHAP explanations are generated and logged to TensorBoard at specified intervals
- Feature importance tracking over training epochs
- Performance-optimized with configurable sample sizes

### Post-Training Analysis
- Comprehensive SHAP explanations for the final trained model
- Multiple visualization types for different interpretation needs
- Integration with existing model evaluation pipeline

## Usage

1. **Basic execution**: Run `python GRU_framework_draft.py`
2. **Testing**: Run `python test_refactored_code.py` to verify functionality
3. **View results**: Use TensorBoard to view training metrics and SHAP explanations

## Dependencies

- PyTorch
- SHAP
- TensorBoard
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## TensorBoard Visualization Sections

- **Training Metrics**: Loss, RMSE, learning rate
- **Model Parameters**: Weight and gradient histograms
- **SHAP Explanations**: 
  - `Training_SHAP/`: Explanations during training
  - `Final_Model_SHAP/`: Comprehensive post-training analysis
- **Model Performance**: A vs E plots, QQ plots, residual analysis