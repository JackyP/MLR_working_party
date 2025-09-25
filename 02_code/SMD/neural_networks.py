"""
Neural Network Models for Claims Reserving

This module contains the neural network architectures used in the GRU framework
for claims reserving modeling.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class BasicLogGRU(nn.Module):
    """
    Basic GRU model with log link for claims reserving.
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
        self.dropout = nn.Dropout(dropout)
        
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