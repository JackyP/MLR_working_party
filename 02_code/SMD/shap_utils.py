"""
SHAP Utilities for Neural Network Explainability

This module provides utilities for generating SHAP explanations for PyTorch neural networks
and creating visualizations suitable for tensorboard logging.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List
import shap
from torch.utils.tensorboard import SummaryWriter


SEED = 42 
rng = np.random.default_rng(SEED) 


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
        self.device = next(model.parameters()).device
        
        self.background_data = background_data.to(self.device)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(background_data.shape[-1])]
        
        self.explainer = shap.GradientExplainer(self.model, self.background_data)
    
    def get_shap_values(self, X: torch.Tensor, max_samples: int = 100) -> np.ndarray:
        """
        Calculate SHAP values for input data.
        
        Args:
            X: Input tensor to explain
            max_samples: Maximum number of samples to process (for computational efficiency)
            
        Returns:
            SHAP values as numpy array
        """
        if X.shape[0] > max_samples:
            #indices = np.random.choice(X.shape[0], max_samples, replace=False)
            indices = rng.choice(X.shape[0], size=max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        self.model.eval()
        X_sample = X_sample.to(self.device)
            
        shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Handle multi-dimensional output from the model
        if hasattr(shap_values, 'ndim') and shap_values.ndim == 4:
            shap_values = shap_values[..., 0]
            
        if isinstance(shap_values, torch.Tensor):
            return shap_values.cpu().numpy()
        return shap_values

    def create_summary_plot(self, shap_values: np.ndarray, X: torch.Tensor, 
                          title: str = "SHAP Summary Plot") -> plt.Figure:
        """
        Create SHAP summary plot figure (bar chart).
        """
        plt.figure(figsize=(10, 6))
        
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
            
        if len(X_np.shape) == 3:
            shap_values_reshaped = shap_values.reshape(-1, shap_values.shape[-1])
            X_np_reshaped = X_np.reshape(-1, X_np.shape[-1])
        else:
            shap_values_reshaped = shap_values
            X_np_reshaped = X_np

        shap.summary_plot(shap_values_reshaped, X_np_reshaped, feature_names=self.feature_names, 
                         show=False, plot_type="bar")
        plt.title(title)
        fig = plt.gcf()
        plt.close()
        return fig

    # --- NEW: Beeswarm Plot Function ---
    def create_beeswarm_plot(self, shap_values: np.ndarray, X: torch.Tensor,
                             title: str = "SHAP Beeswarm Summary Plot") -> plt.Figure:
        """
        Create SHAP beeswarm summary plot figure.
        """
        plt.figure(figsize=(10, 6))
        
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if len(X_np.shape) == 3:
            shap_values_reshaped = shap_values.reshape(-1, shap_values.shape[-1])
            X_np_reshaped = X_np.reshape(-1, X_np.shape[-1])
        else:
            shap_values_reshaped = shap_values
            X_np_reshaped = X_np

        shap.summary_plot(shap_values_reshaped, X_np_reshaped, feature_names=self.feature_names,
                          show=False, plot_type="dot") # "dot" creates the beeswarm plot
        plt.title(title)
        fig = plt.gcf()
        plt.tight_layout()
        plt.close()
        return fig

    # --- NEW: Partial Dependence Plot Function ---
    def create_dependence_plot(self, shap_values: np.ndarray, X: torch.Tensor,
                             feature_idx: int, title: str = "SHAP Dependence Plot") -> plt.Figure:
        """
        Create SHAP partial dependence plot for a single feature.
        """
        plt.figure(figsize=(10, 6))
        
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if len(X_np.shape) == 3:
            shap_values_reshaped = shap_values.reshape(-1, shap_values.shape[-1])
            X_np_reshaped = X_np.reshape(-1, X_np.shape[-1])
        else:
            shap_values_reshaped = shap_values
            X_np_reshaped = X_np

        #feature_name = self.feature_names[feature_idx]
        shap.dependence_plot(feature_idx, shap_values_reshaped, X_np_reshaped,
                             feature_names=self.feature_names, show=False)
        plt.title(title)
        fig = plt.gcf()
        plt.tight_layout()
        plt.close()
        return fig

    def create_waterfall_plot(self, shap_values: np.ndarray, X: torch.Tensor, 
                            sample_idx: int = 0, title: str = "SHAP Waterfall Plot") -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        """
        avg_shap_values_sample = shap_values[sample_idx].mean(axis=0)
        
        if isinstance(X, torch.Tensor):
            X_np_sample = X[sample_idx].detach().cpu().numpy().mean(axis=0)
        else:
            X_np_sample = X[sample_idx].mean(axis=0)

        base_value = self.explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = base_value[0]

        if sample_idx < shap_values.shape[0]:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=avg_shap_values_sample,
                    base_values=base_value,
                    data=X_np_sample,
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title(title)
            fig = plt.gcf()
            plt.close()
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Sample index {sample_idx} out of bounds", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
    
    def create_feature_importance_plot(self, shap_values: np.ndarray, 
                                     title: str = "Feature Importance (SHAP)") -> plt.Figure:
        """
        Create feature importance plot based on mean absolute SHAP values.
        """
        if len(shap_values.shape) == 3:
            shap_values_2d = shap_values.reshape(-1, shap_values.shape[-1])
        else:
            shap_values_2d = shap_values

        importance = np.mean(np.abs(shap_values_2d), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)
        
        ax.barh(range(len(importance)), importance[indices])
        ax.set_xlabel('Mean |SHAP Value| (Average impact on model output magnitude)')
        ax.set_ylabel('Features')
        ax.set_title(title)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        
        plt.tight_layout()
        return fig


def log_shap_explanations(writer: SummaryWriter, explainer: ShapExplainer, 
                         X: torch.Tensor, epoch: int, prefix: str = "SHAP",
                         max_samples: int = 50) -> None:
    """
    Generate and log SHAP explanations to tensorboard.
    """
    try:
        shap_values = explainer.get_shap_values(X, max_samples=max_samples)
        
        # --- MODIFIED: Calculate importance once for reuse ---
        if len(shap_values.shape) == 3:
            shap_values_2d = shap_values.reshape(-1, shap_values.shape[-1])
        else:
            shap_values_2d = shap_values
        mean_abs_shap = np.mean(np.abs(shap_values_2d), axis=0)

        # Log summary plot (bar)
        summary_fig = explainer.create_summary_plot(
            shap_values, X, title=f"{prefix} Summary - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Summary_Bar', summary_fig, epoch)
        plt.close(summary_fig)

        # --- NEW: Log beeswarm summary plot ---
        beeswarm_fig = explainer.create_beeswarm_plot(
            shap_values, X, title=f"{prefix} Beeswarm Summary - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Summary_Beeswarm', beeswarm_fig, epoch)
        plt.close(beeswarm_fig)

        # --- NEW: Log dependence plot for the most important feature ---
        #top_feature_idx = np.argsort(mean_abs_shap)[-1]
        #top_feature_name = explainer.feature_names[top_feature_idx]
        #dependence_fig = explainer.create_dependence_plot(
        #    shap_values, X, feature_idx=top_feature_idx,
        #    title=f"{prefix} Dependence Plot for '{top_feature_name}' - Epoch {epoch}"
        #)
        #writer.add_figure(f'{prefix}/Dependence_Plot_Top_Feature', dependence_fig, epoch)
        #plt.close(dependence_fig)


        num_features = len(explainer.feature_names)

        # --- NEW: Log dependence plots for all features ---
        for feature_idx in range(num_features):
            feature_name = explainer.feature_names[feature_idx]
    
            # Create the dependence plot for the current feature
            dependence_fig = explainer.create_dependence_plot(
                shap_values, X, feature_idx=feature_idx,
                title=f"{prefix} Dependence Plot for '{feature_name}' - Epoch {epoch}"
            )
    
            # Log the figure to the TensorBoard writer (or equivalent)
            # Note: Using the feature_name in the tag makes each plot unique
            writer.add_figure(f'{prefix}/Dependence_Plot/{feature_name}', dependence_fig, epoch)
    
            # Close the figure to free up memory
            plt.close(dependence_fig)


        # Log waterfall plot for first sample
        waterfall_fig = explainer.create_waterfall_plot(
            shap_values, X, sample_idx=0, title=f"{prefix} Waterfall - Epoch {epoch}"
        )
        writer.add_figure(f'{prefix}/Waterfall', waterfall_fig, epoch)
        plt.close(waterfall_fig)
        
        # Log SHAP values as histogram
        writer.add_histogram(f'{prefix}/Values', shap_values.flatten(), epoch)
        
        # Log mean absolute SHAP values per feature
        for i, feature_name in enumerate(explainer.feature_names):
            writer.add_scalar(f'{prefix}/Feature_Importance/{feature_name}', mean_abs_shap[i], epoch)
            
    except Exception as e:
        print(f"Warning: SHAP explanation logging failed at epoch {epoch}: {str(e)}")


def create_background_dataset(X: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
    """
    Create background dataset for SHAP explainer from training data.
    """
    if X.shape[0] > n_samples:
        #indices = np.random.choice(X.shape[0], n_samples, replace=False)
        indices = rng.choice(X.shape[0], size=n_samples, replace=False)
        return X[indices]
    else:
        return X