"""
Neural Network Models for Claims Reserving

File: neural_networks.py

This module contains the neural network architectures used in the GRU framework
for claims reserving modeling.

The following classes are defined

- TabularNetRegressor:  A generic tabular regressor using PyTorch with options for L1 penalty,
- BasicLogGRU: A basic GRU model with log link for claims reserving.
- BasicLogLSTM: A basic LSTM model with log link for claims reserving.
- BasicLogRNN: A basic RNN model with log link for claims reserving.
- LogLinkForwardNet: A multi-layer feedforward network with log link for claims reserving.

- ColumnKeeper: A transformer to keep specified columns in a DataFrame.
- Make3D: A transformer to convert tabular data into 3D tensors for RNN input.


In addition, a model registry and a function to retrieve model classes by name are provided.
- get_model_class

"""
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from typing import Optional

# Local imports
from utils.config import get_default_config
from utils.shap import ShapExplainer, log_shap_explanations, create_background_dataset  


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configuration Setup
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load configuration
config = get_default_config()

# Set pandas display options
pd.options.display.float_format = '{:,.2f}'.format

SEED = 42 
rng = np.random.default_rng(SEED) 
writer = SummaryWriter()  

class TabularNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        module,
        criterion=nn.MSELoss(),
        max_iter=100,   
        max_lr=0.01,
        keep_best_model=False,
        batch_function=None,
        rebatch_every_iter=1,
        n_hidden=20,    
        #n_input=nfeatures,
        #n_output=1,
        l1_penalty=0.0,          # lambda is a reserved word
        l1_applies_params=["linear.weight", "hidden.weight"],
        weight_decay=0.0,
        batch_norm=False,
        interactions=False,
        dropout=0.0,
        clip_value=None,
        n_gaussians=3,
        verbose=1,                
        device="cpu", 
        init_bias=None,
        enable_shap=True,  # Enable SHAP explanations
        shap_log_frequency=500,  # Log SHAP every N epochs
        **kwargs
    ):
        """ Tabular Neural Network Regressor (for Claims Reserving)

        This trains a neural network with specified loss, Log Link and l1 LASSO penalties
        using Pytorch. It has early stopping and SHAP explainability.

        Args:
            module: pytorch nn.Module. Should have n_input and n_output as parameters and
                if l1_penalty, init_weight, or init_bias are used, a final layer 
                called "linear".

            criterion: pytorch loss function. Consider nn.PoissonNLLLoss for log link.

            max_iter (int): Maximum number of epochs before training stops.

            max_lr (float): Min / Max learning rate - we will use one_cycle_lr

            keep_best_model (bool): If true, keep and use the model weights with the best loss rather 
                than the final weights.

            batch_function (None or fn): If not None, used to get a batch from X and y

            rebatch_every_iter (int): redo batches every

            l1_penalty (float): l1 penalty factor. If not zero, is applied to 
                parameters in l1_applies_params.

            l1_applies_params (list): Parameters to apply l1 penalty to.

            weight_decay (float): L2 penalty to apply to all parameters

            batch_norm (bool): Whether to use batch normalization

            interactions (bool): Whether to use interaction terms

            dropout (float): Dropout probability

            clip_value (None or float): If not None, clip gradients to this value

            verbose (int): Verbosity level

            device (str): Device to use ('cpu', 'cuda', etc.)

            init_bias (None or float): If not None, initialize output layer bias to this value

            enable_shap (bool): Whether to enable SHAP explanations during training

            shap_log_frequency (int): Frequency of SHAP logging (every N epochs)
        """
        
        self.module = module
        self.criterion = criterion
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.keep_best_model = keep_best_model
        self.batch_function = batch_function
        self.rebatch_every_iter = rebatch_every_iter
        self.n_hidden = n_hidden
        #self.n_input = n_input
        #self.n_output = n_output
        self.l1_penalty = l1_penalty
        self.l1_applies_params = l1_applies_params
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm
        self.interactions = interactions
        self.dropout = dropout
        self.clip_value = clip_value
        self.n_gaussians = n_gaussians
        self.verbose = verbose
        self.device = device
        self.init_bias = init_bias
        self.enable_shap = enable_shap
        self.shap_log_frequency = shap_log_frequency
        self.kwargs = kwargs
        
        # Target device for tensors
        self.target_device = torch.device(device)
        
        # Training state
        self.module_ = None
        self.best_model = None
        self.shap_explainer = None
        self.print_loss_every_iter = kwargs.get('print_loss_every_iter', max(1, int(max_iter / 10)))

        
    def fix_array(self, y):
        """Need to be picky about array formats"""
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y.astype(np.float32)
        return y
        

    def setup_module(self, n_input, n_output):
         
        # Training new model
        self.module_ = self.module(
            n_input=n_input, 
            n_output=n_output,
            n_hidden=self.n_hidden,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
#            interactions=self.interactions,
#            n_gaussians=self.n_gaussians,
            init_bias=self.init_bias_calc if self.init_bias is None else self.init_bias,
            **self.kwargs
        ).to(self.target_device)
        

    def fit(self, X, y):
        # The main fit logic is in partial_fit
        # We will try a few times if numbers explode because NN's are finicky and we are doing CV
        n_input = X.shape[-1]
        n_output = 1 if y.ndim == 1 else y.shape[-1]
        self.init_bias_calc = np.log(y.mean()).astype(np.float32)
        self.setup_module(n_input=n_input, n_output=n_output)

        # Partial fit means you take an existing model and keep training 
        # so the logic is basically the same
        self.partial_fit(X, y)
        
        return self


    def partial_fit(self, X, y):
        # Check that X and y have correct shape
        check_consistent_length(X, y)

        # Convert to Pytorch Tensor
        X_tensor = X.to(self.target_device)
        y_tensor = torch.from_numpy(self.fix_array(y)).to(self.target_device)

        # Initialize SHAP explainer if enabled
        if self.enable_shap and self.shap_explainer is None:
            try:
                # Create background dataset for SHAP
                background_data = create_background_dataset(X_tensor, n_samples=100)
                feature_names = config.data.features
                self.shap_explainer = ShapExplainer(self.module_, background_data, feature_names)
                if self.verbose > 0:
                    print("SHAP explainer initialized successfully")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Failed to initialize SHAP explainer: {e}")
                self.enable_shap = False

        # Optimizer - the generically useful AdamW. Other options like SGD are also possible.
        optimizer = torch.optim.AdamW(
            params=self.module_.parameters(),
            lr=self.max_lr / 10,
            weight_decay=self.weight_decay
        )
        
        # Scheduler - one cycle LR
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            steps_per_epoch=1, 
            epochs=self.max_iter
        )
        
        # Loss Function
        try:
            loss_fn = self.criterion(log_input=False).to(self.target_device)  # Pytorch loss function
        except TypeError:
            loss_fn = self.criterion  # Custom loss function

        best_loss = float('inf') # set to infinity initially

        if self.batch_function is not None:
            X_tensor_batch, y_tensor_batch = self.batch_function(X_tensor, y_tensor)
        else:
            X_tensor_batch, y_tensor_batch = X_tensor, y_tensor

        # Training loop
        for epoch in range(self.max_iter):   # Repeat max_iter times

            self.module_.train()
            y_pred = self.module_(X_tensor_batch)  #  Apply current model

            # Tensorboard logging
            expected = y_pred.detach().cpu().numpy()
            ln_expected = np.log(expected + 1e-8)  # Add small epsilon to avoid log(0)
            ln_actual = np.log(y_tensor_batch.detach().cpu().numpy() + 1e-8)
            diff = y_tensor_batch.detach().cpu().numpy() - expected
            
            loss = loss_fn(y_pred, y_tensor_batch) #  What is the loss on it?

            # Tensorboard logging - basic metrics
            writer.add_scalar("Loss", loss, epoch)
            
            # Learning rate logging
            current_lr = scheduler.get_last_lr()[0]  # Assuming one parameter group
            writer.add_scalar('Learning Rate', current_lr, epoch)

            # Weights and biases logging
            for name, param in self.module_.named_parameters():       
                writer.add_histogram(name, param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)

            # SHAP explanations logging (less frequent to avoid performance issues)
            if (self.enable_shap and self.shap_explainer is not None and 
                epoch % self.shap_log_frequency == 0 and epoch > 0):
                try:

                    # Generate SHAP explanations for a sample of training data
                    sample_size = min(200, len( X_tensor_batch))
                    print(f'sample_size: {sample_size}')
                    #sample_indices = np.random.choice(len(X_tensor), sample_size, replace=False)
                    sample_indices = rng.choice(len(X_tensor_batch), sample_size, replace=False)
                    X_sample = X_tensor[sample_indices]

                    log_shap_explanations(
                        writer, self.shap_explainer, X_sample, epoch, 
                        prefix="Training_SHAP", max_samples=sample_size
                    )
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: SHAP logging failed at epoch {epoch}: {e}")
                
            if self.l1_penalty > 0.0:        #  Lasso penalty
                loss += self.l1_penalty * sum(
                    [
                        w.abs().sum()
                        for p, w in self.module_.named_parameters()
                        if p in self.l1_applies_params
                    ]
                )
                
            if self.keep_best_model & (loss.item() < best_loss):
                best_loss = loss.item()
                self.best_model = self.module_.state_dict()

            optimizer.zero_grad()            #  Reset optimizer            
            loss.backward()                  #  Apply back propagation
            
            # gradient norm clipping
            if self.clip_value is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip_value)
                # check if gradients have been clipped
                if (self.verbose >= 1) & (grad_norm > self.clip_value):
                    print(f'Gradient norms have been clipped in epoch {epoch}, value before clipping: {grad_norm}')    

            optimizer.step()                 #  Update model parameters
            scheduler.step()

            if torch.isnan(loss.data).tolist():
                raise ValueError('Error: nan loss')


            # Every self.print_loss_every_iter steps, print RMSE 
            if (epoch % self.print_loss_every_iter == 0) and (self.verbose > 0):
                self.module_.eval()                     # Eval mode 
                self.module_.point_estimates=True       # Distributional models - set to point                
                y_pred_point = self.module_(X_tensor)   # Get "real" model estimates
                assert(y_pred_point.size() == y_tensor.size())
                rmse = torch.sqrt(torch.mean(torch.square(y_pred_point - y_tensor)))
                self.module_.train()                     # back to training
                self.module_.point_estimates=False       # Distributional models - set to point
                
                print("Train RMSE: ", rmse.data.tolist(), " Train Loss: ", loss.data.tolist(), " Epoch: ", epoch)


                #Tensorboard
                writer.add_scalar("RMSE", rmse, epoch)
                writer.add_histogram('Expected', expected, epoch)
                writer.add_histogram('Diff', diff, epoch)

                fig, ax = plt.subplots()
                ax.scatter(y_tensor_batch, expected)
                ax.set_xlabel('Actual', fontsize=15)
                ax.set_ylabel('Expected', fontsize=15)
                ax.set_title('A vs E')               
                writer.add_figure('AvsE', fig, epoch)

                fig, ax = plt.subplots()                
                ax.scatter(ln_actual, ln_expected)
                ax.plot([0,16],[0,16])
                ax.set_xlabel('Actual', fontsize=15)
                ax.set_ylabel('Expected', fontsize=15)
                ax.set_title('A vs E Logged')               
                writer.add_figure('AvsE Logged', fig, epoch)

                
            if (self.batch_function is not None) & (epoch % self.rebatch_every_iter == 0):
                print(f"refreshing batch on epoch {epoch}")
                X_tensor_batch, y_tensor_batch = self.batch_function(X_tensor, y_tensor)
        
            if (epoch==self.max_iter-1):
                for name, param in self.module_.named_parameters():    
                    # Convert parameter tensor to numpy array and flatten it
                    param_np = param.detach().numpy().flatten()
                    fig, ax = plt.subplots()
                    ax.bar(range(len(param_np)), param_np)
                    ax.set_title(f'Parameters/{name}')
                    ax.set_xlabel('Parameter Node')
                    ax.set_ylabel('Value')
                    writer.add_figure(f'Parameters/{name}', fig, epoch)

        if self.keep_best_model:
            self.module_.load_state_dict(self.best_model)
            self.module_.eval()
            
        return self


    def predict(self, X, point_estimates=True):
        # Checks
        check_is_fitted(self)      # Check is fit had been called
        check_consistent_length(X)

        # Convert to Pytorch Tensor
        X_tensor = X.to(self.target_device)
      
        self.module_.eval()  # Eval (prediction) mode
        self.module_.point_estimates = point_estimates

        # Apply current model and convert back to numpy
        if point_estimates:
            y_pred = self.module_(X_tensor).cpu().detach().numpy()
            if y_pred.shape[-1] == 1: 
                return y_pred.ravel()
            else:
                return y_pred
        else:
            y_pred = self.module_(X_tensor)
            return y_pred


    def score(self, X, y):
        # Negative RMSE score (higher needs to be better)
        y_pred = self.predict(X)
        y = self.fix_array(y)
        return -np.sqrt(np.mean((y_pred - y)**2))

    def get_hidden_state(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X_tensor = torch.tensor(X).to(self.target_device)

        self.module_.eval()
        with torch.no_grad():
            _, hn = self.module_(X_tensor, return_hn=True)

        return hn.cpu().numpy()



class BasicLogGRU(nn.Module):
    """
    This BasicLogGRU class is a neural network for insurance claims reserving that predicts ultimate claim amounts using sequential payment data. Here's what each part does:

    Architecture Overview
    ========================
    The model processes sequential claim payment data through a GRU (Gated Recurrent Unit) and outputs predictions using a log-link transformation.

    Key Components
    ===============
    
    Constructor Parameters:
    --------------
    n_input: Number of input features per time step
    n_hidden: Size of the GRU hidden state
    n_output: Number of outputs (typically 1 for claim amount)
    init_bias: Initial bias value (often set to log of mean claim amount)
    batch_norm: Whether to apply batch normalization
    dropout: Dropout probability for regularization
    
    Neural Network Layers:
    ----------------------
    1 GRU Layer: nn.GRU(n_input, n_hidden, batch_first=True)
        Processes sequential payment history
        batch_first=True means input shape is (batch, sequence, features)
    2 Optional Batch Normalization: nn.BatchNorm1d(n_hidden)
        Normalizes the GRU output to stabilize training
    3 Dropout Layer: nn.Dropout(dropout)
        Prevents overfitting by randomly zeroing some neurons
    4 Output Layer: nn.Linear(n_hidden, n_output)
        Maps hidden state to final prediction
    
    Forward Pass Logic:
    --------------
    1 GRU Processing: h, _ = self.gru(x)
        Processes entire sequence, returns all hidden states
    2 Last Time Step: h = h[:, -1, :]
        Takes only the final hidden state (most recent information)
    3 Apply Regularization: Batch norm → Dropout
    4 Log Link: torch.exp(self.linear(h))
        Ensures positive predictions (claim amounts can't be negative)
        Common in insurance modeling

    Why This Design?
    --------------------
    Sequential Processing: Claims develop over time with multiple payments
    Log Link: Guarantees positive outputs, handles skewed claim distributions
    GRU: Captures temporal dependencies in payment patterns
    Regularization: Batch norm and dropout prevent overfitting
    
    This model predicts the ultimate claim amount based on the payment history pattern learned during training.
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_hidden: int, 
        n_output: int,
        init_bias: Optional[float] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super(BasicLogGRU, self).__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batch_norm = batch_norm
        self.point_estimates = True
        
        # GRU layer
        self.gru = nn.GRU(n_input, n_hidden, batch_first=True)
        
        # Batch normalization
        if batch_norm:
            self.batchn = nn.BatchNorm1d(n_hidden)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output layer
        self.linear = nn.Linear(n_hidden, n_output)
        
        # Initialize bias if provided
        if init_bias is not None:
            self.linear.bias.data.fill_(init_bias)
    
    def forward(self, x):
        # GRU forward pass
        h, _ = self.gru(x)
        
        # Take the last output from the sequence
        h = h[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.batch_norm:
            h = self.batchn(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        # Log link: Y = exp(XB)
        return torch.exp(self.linear(h))


class BasicLogLSTM(nn.Module):
    """
    Basic LSTM model with log link for claims reserving.
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_hidden: int, 
        n_output: int,
        init_bias: Optional[float] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super(BasicLogLSTM, self).__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batch_norm = batch_norm
        self.point_estimates = True
        
        # LSTM layer
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True)
        
        # Batch normalization
        if batch_norm:
            self.batchn = nn.BatchNorm1d(n_hidden)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(n_hidden, n_output)
        
        # Initialize bias if provided
        if init_bias is not None:
            self.linear.bias.data.fill_(init_bias)
    
    def forward(self, x):
        # LSTM forward pass
        h, _ = self.lstm(x)
        
        # Take the last output from the sequence
        h = h[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.batch_norm:
            h = self.batchn(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        # Log link: Y = exp(XB)
        return torch.exp(self.linear(h))


class BasicLogRNN(nn.Module):
    """
    Basic RNN model with log link for claims reserving.
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_hidden: int, 
        n_output: int,
        init_bias: Optional[float] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super(BasicLogRNN, self).__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batch_norm = batch_norm
        self.point_estimates = True
        
        # RNN layer
        self.rnn = nn.RNN(n_input, n_hidden, batch_first=True)
        
        # Batch normalization
        if batch_norm:
            self.batchn = nn.BatchNorm1d(n_hidden)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(n_hidden, n_output)
        
        # Initialize bias if provided
        if init_bias is not None:
            self.linear.bias.data.fill_(init_bias)
    
    def forward(self, x):
        # RNN forward pass
        h, _ = self.rnn(x)
        
        # Take the last output from the sequence
        h = h[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.batch_norm:
            h = self.batchn(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        # Log link: Y = exp(XB)
        return torch.exp(self.linear(h))


class LogLinkForwardNet(nn.Module):
    """
    Multi-layer feedforward network with log link for claims reserving.
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_hidden: int, 
        n_output: int,
        init_bias: Optional[float] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super(LogLinkForwardNet, self).__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batch_norm = batch_norm
        self.point_estimates = True
        
        # Hidden layers
        self.hidden = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        
        # Batch normalization layers
        if batch_norm:
            self.batchn = nn.BatchNorm1d(n_hidden)
            self.batchn2 = nn.BatchNorm1d(n_hidden)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(n_hidden, n_output)
        
        # Initialize bias if provided
        if init_bias is not None:
            self.linear.bias.data.fill_(init_bias)
    
    def forward(self, x):
        # Reshape input if it's 3D (batch, sequence, features) -> (batch*sequence, features)
        if len(x.shape) == 3:
            batch_size, seq_len, n_features = x.shape
            x = x.reshape(-1, n_features)
            reshape_output = True
        else:
            reshape_output = False
        
        # First hidden layer
        h = F.relu(self.hidden(x))
        if self.batch_norm:
            h = self.batchn(h)
        h = self.dropout(h)
        
        # Second hidden layer
        h2 = F.relu(self.hidden2(h))
        if self.batch_norm:
            h2 = self.batchn2(h2)
        h2 = self.dropout(h2)
        
        # Output with log link
        output = torch.exp(self.linear(h2))
        
        # Reshape output back if needed
        if reshape_output:
            output = output.reshape(batch_size, seq_len, -1)
        
        return output



# Model registry for easy access
MODEL_REGISTRY = {
    'BasicLogGRU': BasicLogGRU,
    'BasicLogLSTM': BasicLogLSTM,
    'BasicLogRNN': BasicLogRNN,
    'LogLinkForwardNet': LogLinkForwardNet,
}


def get_model_class(model_name: str):
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model class
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# classes to support modelling
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
class ColumnKeeper(BaseEstimator, TransformerMixin):
    """
    The ColumnKeeper class is a custom transformer that integrates with scikit-learn's preprocessing pipeline. Here's what it does:

    Purpose
    It selects and keeps only specified columns from a DataFrame while preserving the DataFrame format (rather than converting to a numpy array).

    How it works
    Initialization (__init__): 
     - Takes a list of column names (cols) that you want to keep
    Fit method (fit): 
     - Does nothing and returns self - this is required by scikit-learn's transformer interface but no actual fitting/learning is needed
    Transform method (transform):
     - Creates a copy of the input DataFrame (X.copy())
     - Selects only the columns specified in self.cols
     - Returns the filtered DataFrame
    
    Why it's useful
    Pipeline compatibility: Works seamlessly with scikit-learn pipelines
    DataFrame preservation: Unlike many sklearn transformers that return numpy arrays, this keeps the pandas DataFrame structure
    Column selection: Useful for feature selection or removing unwanted columns in a pipeline
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.copy()[self.cols]

        
class Make3D(BaseEstimator, TransformerMixin):
    """
    What the class is designed to do:
    This transformer converts tabular claims data into 3D tensors suitable for recurrent neural networks (like GRU/LSTM). Here's the process:

    Groups data by claim: Each claim has multiple rows (time sequence)
    Converts to tensors: Each group becomes a PyTorch tensor
    Pads sequences: Claims have different lengths, so shorter sequences are padded to match the longest

    Why this transformation is needed:
    Sequential data: Claims develop over time with multiple payments
    RNN input format: RNNs expect 3D input: (batch_size, sequence_length, num_features)
    Variable lengths: Different claims have different numbers of payments, requiring padding
    
    Example output shape:
    If you have 100 claims, max 20 time steps, and 5 features, the output tensor would be (100, 20, 5).

    This transformer is essential for preparing your claims data for the GRU model that predicts ultimate claim amounts based on payment history.
    """

    def __init__(self, data_cols):
        self.features = data_cols

    def fit(self, X, y=None):
        # No fitting necessary; return self
        return self

    def transform(self, X):
        # Group by 'claim_no'
        grouped = X.groupby('claim_no')
        # Convert each group to a tensor
        X_tensors = [torch.tensor(group[self.features].values, dtype=torch.float32) for _, group in grouped]
        # Pad sequences to the same length
        X_padded = pad_sequence(X_tensors, batch_first=True)
        return X_padded

