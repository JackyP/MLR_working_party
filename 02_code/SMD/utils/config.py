"""
Configuration Management for GRU Framework

This module centralizes all configuration parameters for the GRU framework,
making it easier to manage and modify hyperparameters and settings.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class DataConfig:
    """Data processing and loading configuration."""
    
    # File paths
    data_dir: str = "/home/nigel/git/MLR_working_party/01_data/"
    filename: str = "data_origframework_nofills_nosttl.csv"
    
    # Data processing parameters
    maxdev: int = 40
    cutoff: int = 40  # Hard-coded cut-off as set out previously
    cutoff1: int = 32
    
    # Feature configuration
    features: List[str] = None
    data_cols: List[str] = None
    output_field: str = "claim_size"
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                "occurrence_time", 
                "notidel", 
                "development_period", 
                "pmt_no",
                "log1_paid_cumulative",
                #"log1_cumulative_payment_to_prior_period",
            ]
        
        if self.data_cols is None:
            self.data_cols = self.features + ["claim_no"]


@dataclass 
class ModelConfig:
    """Neural network model configuration."""
    
    # Model architecture
    n_hidden: int = 32
    n_output: int = 1
    batch_norm: bool = False
    dropout: float = 0.0
    
    # Training parameters
    max_lr: float = 0.01
    weight_decay: float = 0.0
    l1_penalty: float = 0.0
    l1_applies_params: List[str] = None
    clip_value: Optional[float] = None
    
    # Early stopping and model selection
    keep_best_model: bool = False
    
    # Distributed/device settings
    device: str = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.l1_applies_params is None:
            self.l1_applies_params = ["linear.weight", "hidden.weight"]


@dataclass
class TrainingConfig:
    """Training process configuration."""
    
    # Iteration settings
    nn_iter: int = 2001  # 100 for experimentation, 500+ for convergence
    cv_runs: int = 24  # Testing: 4, proper runs: 24
    glm_iter: int = 500  # GLM gradient descent epochs
    nn_cv_iter: int = 100  # Lower for CV to save time
    mdn_iter: int = 1000  # MDN convergence requirement
    
    # Training behavior
    verbose: int = 1
    print_loss_every_iter: int = 100
    rebatch_every_iter: int = 1
    
    # SHAP configuration
    enable_shap: bool = True
    shap_log_frequency: int = 500  # Log SHAP every N epochs
    shap_max_samples: int = 50  # Max samples for SHAP calculation
    shap_background_samples: int = 100  # Background dataset size for SHAP


@dataclass
class TensorboardConfig:
    """Tensorboard logging configuration."""
    
    # Logging frequency
    log_weights_histograms: bool = True
    log_gradients: bool = True
    log_metrics_every: int = 1
    log_figures_every: int = 100
    
    # SHAP logging
    log_shap_explanations: bool = True
    shap_log_frequency: int = 500


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = None
    run_timestamp: str = None
    
    # Component configurations
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    tensorboard: TensorboardConfig = None
    
    def __post_init__(self):
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.experiment_name is None:
            self.experiment_name = f"gru_experiment_{self.run_timestamp}"
            
        # Initialize sub-configs if not provided
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()  
        if self.training is None:
            self.training = TrainingConfig()
        if self.tensorboard is None:
            self.tensorboard = TensorboardConfig()


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def load_config_from_dict(config_dict: dict) -> ExperimentConfig:
    """
    Load configuration from dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        
    Returns:
        ExperimentConfig instance
    """
    # Extract sub-configurations
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    tensorboard_config = TensorboardConfig(**config_dict.get('tensorboard', {}))
    
    # Create main config
    experiment_config = ExperimentConfig(
        experiment_name=config_dict.get('experiment_name'),
        run_timestamp=config_dict.get('run_timestamp'),
        data=data_config,
        model=model_config,
        training=training_config,
        tensorboard=tensorboard_config
    )
    
    return experiment_config