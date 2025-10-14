"""
Configuration Management for GRU Framework

This module centralizes all configuration parameters for the GRU framework,
making it easier to manage and modify hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Type
from datetime import datetime
import yaml
from pathlib import Path


def create_dynamic_dataclass(class_name: str, fields_dict: Dict[str, Any]) -> Type:
    """
    Dynamically creates a dataclass with attributes defined by the keys 
    in the input dictionary. All fields are set with their provided 
    value as the default.
    """
    
    # Prepare the class attributes dictionary with type annotations
    field_definitions = {}
    annotations = {}
    
    for name, value in fields_dict.items():
        # Infer type from the value
        field_type = type(value)
        
        # Handle mutable defaults (lists, dicts, sets) using default_factory
        if isinstance(value, (list, dict, set)):
            # Use lambda to create a factory function that returns a copy of the value
            field_definitions[name] = field(default_factory=lambda v=value: v.copy() if hasattr(v, 'copy') else list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else set(v))
        else:
            # Immutable types can use default directly
            field_definitions[name] = field(default=value)
        
        # Add type annotation
        annotations[name] = field_type

    # Add the annotations to the field definitions
    field_definitions['__annotations__'] = annotations
    
    # Use type() to create the new class dynamically
    DynamicClass = type(class_name, (object,), field_definitions)
    
    # Register the class as a dataclass
    return dataclass(DynamicClass)


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
    """Tensorboard logging configuration (dynamic)."""
    
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



def load_dynamic_configs(config_dict):
    """
    Dynamically creates dataclasses from the config dictionary.
    """
    
    instantiated_configs = {}
    
    for class_name, fields_dict in config_dict.items():
        # Skip non-dictionary values (like experiment_name, run_timestamp)
        if not isinstance(fields_dict, dict):
            # Store scalar values directly
            instantiated_configs[class_name] = fields_dict
            continue
            
        # 1. Dynamically CREATE the class (e.g., DataClass)
        DynamicClass = create_dynamic_dataclass(class_name, fields_dict)
        
        # 2. INSTANTIATE the class
        instance = DynamicClass(**fields_dict)
        
        # 3. STORE the instance
        instantiated_configs[class_name] = instance
    
    return instantiated_configs


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







def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
            
        if config_dict is None:
            config_dict = {}
            
        #return load_config_from_dict(config_dict)
        return load_dynamic_configs(config_dict)
        
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")



def save_config_to_yaml(config: ExperimentConfig, yaml_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: ExperimentConfig instance to save
        yaml_path: Path where to save the YAML file
    """
    yaml_path = Path(yaml_path)
    
    # Convert dataclass to dictionary
    config_dict = {
        'experiment_name': config.experiment_name,
        'run_timestamp': config.run_timestamp,
        'data': {
            'data_dir': config.data.data_dir,
            'filename': config.data.filename,
            'maxdev': config.data.maxdev,
            'cutoff': config.data.cutoff,
            'cutoff1': config.data.cutoff1,
            'features': config.data.features,
            'data_cols': config.data.data_cols,
            'output_field': config.data.output_field
        },
        'model': {
            'n_hidden': config.model.n_hidden,
            'n_output': config.model.n_output,
            'batch_norm': config.model.batch_norm,
            'dropout': config.model.dropout,
            'max_lr': config.model.max_lr,
            'weight_decay': config.model.weight_decay,
            'l1_penalty': config.model.l1_penalty,
            'l1_applies_params': config.model.l1_applies_params,
            'clip_value': config.model.clip_value,
            'keep_best_model': config.model.keep_best_model,
            'device': config.model.device
        },
        'training': {
            'nn_iter': config.training.nn_iter,
            'cv_runs': config.training.cv_runs,
            'glm_iter': config.training.glm_iter,
            'nn_cv_iter': config.training.nn_cv_iter,
            'mdn_iter': config.training.mdn_iter,
            'verbose': config.training.verbose,
            'print_loss_every_iter': config.training.print_loss_every_iter,
            'rebatch_every_iter': config.training.rebatch_every_iter,
            'enable_shap': config.training.enable_shap,
            'shap_log_frequency': config.training.shap_log_frequency,
            'shap_max_samples': config.training.shap_max_samples,
            'shap_background_samples': config.training.shap_background_samples
        },
        'tensorboard': {
            'log_weights_histograms': config.tensorboard.log_weights_histograms,
            'log_gradients': config.tensorboard.log_gradients,
            'log_metrics_every': config.tensorboard.log_metrics_every,
            'log_figures_every': config.tensorboard.log_figures_every,
            'log_shap_explanations': config.tensorboard.log_shap_explanations,
            'shap_log_frequency': config.tensorboard.shap_log_frequency
        }
    }
    
    # Create directory if it doesn't exist
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.dump(config_dict, file, default_flow_style=False, indent=2)


