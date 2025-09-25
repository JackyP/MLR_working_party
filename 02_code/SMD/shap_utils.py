"""
SHAP Utilities for Neural Network Explainability

This module provides utilities for generating SHAP explanations for PyTorch neural networks
and creating visualizations suitable for tensorboard logging.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union
import shap
from torch.utils.tensorboard import SummaryWriter


class ShapExplainer:
    """
    Wrapper class for SHAP explanations of PyTorch neural network models.
    """
    
    def __init__(self, model, background_data: torch.Tensor, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer for PyTorch model.
        
        Args:
            model: PyTorch model to explain
            background_data: Background dataset for SHAP explanations (tensor)
            feature_names: Optional list of feature names for better visualization
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(background_data.shape[-1])]
        
        # Create SHAP explainer using DeepExplainer for neural networks
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def get_shap_values(self, X: torch.Tensor, max_samples: int = 100) -> np.ndarray:
        """
        Calculate SHAP values for input data.
        
        Args:
            X: Input tensor to explain
            max_samples: Maximum number of samples to process (for computational efficiency)
            
        Returns:
            SHAP values as numpy array
        """
        # Limit samples for computational efficiency
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle different output formats from SHAP
        if isinstance(shap_values, list):
            # Multi-output case - take first output
            shap_values = shap_values[0]
            
        return shap_values
    
    def create_summary_plot(self, shap_values: np.ndarray, X: torch.Tensor, 
                          title: str = "SHAP Summary Plot") -> plt.Figure:
        """
        Create SHAP summary plot figure.
        
        Args:
            shap_values: SHAP values array
            X: Input data tensor
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        
        # Convert tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
            
        # Reshape if needed for 3D tensor (batch, sequence, features)
        if len(X_np.shape) == 3:
            X_np = X_np.reshape(-1, X_np.shape[-1])
            shap_values = shap_values.reshape(-1, shap_values.shape[-1])
        
        # Create summary plot
        shap.summary_plot(shap_values, X_np, feature_names=self.feature_names, 
                         show=False, plot_type="bar")
        plt.title(title)
        fig = plt.gcf()
        plt.close()
        return fig
    
    def create_waterfall_plot(self, shap_values: np.ndarray, X: torch.Tensor, 
                            sample_idx: int = 0, title: str = "SHAP Waterfall Plot") -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            shap_values: SHAP values array
            X: Input data tensor
            sample_idx: Index of sample to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Convert tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
            
        # Reshape if needed for 3D tensor
        if len(X_np.shape) == 3:
            X_np = X_np.reshape(-1, X_np.shape[-1])
            shap_values = shap_values.reshape(-1, shap_values.shape[-1])
        
        # Create waterfall plot for specific sample
        if sample_idx < shap_values.shape[0]:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=np.mean(shap_values),
                    data=X_np[sample_idx],
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title(title)
            fig = plt.gcf()
            plt.close()
            return fig
        else:
            # Return empty figure if sample_idx is out of bounds
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Sample index {sample_idx} out of bounds", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
    
    def create_feature_importance_plot(self, shap_values: np.ndarray, 
                                     title: str = "Feature Importance (SHAP)") -> plt.Figure:
        """
        Create feature importance plot based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values array
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Reshape if needed for 3D tensor
        if len(shap_values.shape) == 3:
            shap_values = shap_values.reshape(-1, shap_values.shape[-1])
        
        # Calculate mean absolute SHAP values for feature importance
        importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xlabel('Features')
        ax.set_ylabel('Mean |SHAP Value|')
        ax.set_title(title)
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
        
        plt.tight_layout()
        return fig


def log_shap_explanations(writer: SummaryWriter, explainer: ShapExplainer, 
                         X: torch.Tensor, epoch: int, prefix: str = "SHAP",
                         max_samples: int = 50) -> None:
    """
    Generate and log SHAP explanations to tensorboard.
    
    Args:
        writer: TensorBoard SummaryWriter
        explainer: ShapExplainer instance
        X: Input tensor to explain
        epoch: Current epoch number
        prefix: Prefix for tensorboard tags
        max_samples: Maximum number of samples for SHAP calculation
    """
    try:
        # Calculate SHAP values
        shap_values = explainer.get_shap_values(X, max_samples=max_samples)
        
        # Log feature importance plot
        importance_fig = explainer.create_feature_importance_plot(
            shap_values, title=f"{prefix} Feature Importance - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Feature_Importance', importance_fig, epoch)
        plt.close(importance_fig)
        
        # Log summary plot
        summary_fig = explainer.create_summary_plot(
            shap_values, X, title=f"{prefix} Summary - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Summary', summary_fig, epoch)
        plt.close(summary_fig)
        
        # Log waterfall plot for first sample
        waterfall_fig = explainer.create_waterfall_plot(
            shap_values, X, sample_idx=0, title=f"{prefix} Waterfall - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Waterfall', waterfall_fig, epoch)
        plt.close(waterfall_fig)
        
        # Log SHAP values as histogram
        writer.add_histogram(f'{prefix}/Values', shap_values.flatten(), epoch)
        
        # Log mean absolute SHAP values per feature
        if len(shap_values.shape) == 3:
            shap_values_2d = shap_values.reshape(-1, shap_values.shape[-1])
        else:
            shap_values_2d = shap_values
            
        mean_abs_shap = np.mean(np.abs(shap_values_2d), axis=0)
        for i, feature_name in enumerate(explainer.feature_names):
            writer.add_scalar(f'{prefix}/Feature_{feature_name}', mean_abs_shap[i], epoch)
            
    except Exception as e:
        print(f"Warning: SHAP explanation logging failed at epoch {epoch}: {str(e)}")


def create_background_dataset(X: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
    """
    Create background dataset for SHAP explainer from training data.
    
    Args:
        X: Full training dataset tensor
        n_samples: Number of background samples to use
        
    Returns:
        Background dataset tensor
    """
    if X.shape[0] > n_samples:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[indices]
    else:
        return X