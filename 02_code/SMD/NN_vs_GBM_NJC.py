import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


# Scikit-learn imports
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit


import chainladder as cl
import math
import random
import shap


# Local imports
from utils.config import ExperimentConfig, load_config_from_yaml
from utils.neural_networks import TabularNetRegressor, FeedForwardNet, ColumnKeeper, Make3D
from utils.data_engineering import load_data, process_data, create_train_test_datasets, process_data_davide
from utils.tensorboard import generate_enhanced_tensorboard_outputs, create_actual_vs_expected_plot
from utils.excel import save_df_to_excel



# Load from YAML file
config = load_config_from_yaml('configs/NN_v_GBM_NJC_config.yaml')

# Set pandas display options
pd.options.display.float_format = '{:,.2f}'.format

type(config['data'])

SEED = config['training'].seed 
rng = np.random.default_rng(SEED) 
#writer = SummaryWriter() 

# Create timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/log_NJC_NN_vs_GBM_outputs_{timestamp}.xlsx"

print(f"Experiment timestamp: {timestamp}")
print(f"Output file: {log_filename}")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Loading and Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load data
dat_orig = load_data(config)
save_df_to_excel(dat_orig, df_name="Original Data", filename=log_filename, mode='w')


dat = process_data_davide(config, dat_orig)
#save_df_to_excel(dat, df_name="Processed Data", filename=log_filename, mode='a')

data_cols = config['data'].data_cols

set(data_cols) - set(dat.columns)

features = data_cols + ["backdate_periods"]


def backdate(df, backdate_periods, keep_cols):
    dedupe = [*set(["claim_no", "occurrence_period", "development_period", "payment_period"] + keep_cols)]
    bd = df.loc[:, dedupe].copy()
    bd["development_period"]= bd.development_period + backdate_periods
    bd.rename(columns={"payment_period": "payment_period_as_at"}, inplace=True)
    df= df[["claim_no", "occurrence_period", "development_period", "train_ind", "payment_size", "payment_period", "occurrence_time", "notidel"]].assign(
        data_as_at_development_period = lambda df: df.development_period - backdate_periods, 
        backdate_periods = backdate_periods
    ).merge(
        bd,
        how='left',
        on=["claim_no", "occurrence_period", "development_period"],
        suffixes=[None, "_backdatedrop"]
    )
    return df.drop(df.filter(regex='_backdatedrop').columns, axis=1)

num_dev_periods = config['data'].cutoff - 1

backdated_data = [backdate(dat, backdate_periods=i, keep_cols=data_cols) for i in range(0, num_dev_periods)]

extra_data = (
    dat.loc[dat.train_ind == 1, [*set(["claim_no", "occurrence_period", "development_period", "payment_period", "train_ind"] + data_cols)]]  # Training data
        .groupby("claim_no").last()  # Last training record per claim
        .rename(columns={"payment_period": "payment_period_as_at", "development_period": "data_as_at_development_period"})
        .assign(
            development_period = num_dev_periods + 1,  # set dev period to be tail
            payment_period = lambda df: df.occurrence_period + num_dev_periods + 1,
            backdate_periods = lambda df: num_dev_periods + 1 - df.payment_period,
            payment_size = 0
        )
        .reset_index()
)


all_data = pd.concat(backdated_data + [extra_data], axis="rows")

a = set(all_data.columns.to_list())
b = set(extra_data.columns.to_list())

assert list(b - a) == []
assert list(a - b) == []

nn_train_full = all_data.loc[all_data.train_ind == 1].loc[lambda df: ~np.isnan(df.payment_period_as_at)]        # Filter out invalid payment period as at
nn_test = all_data.loc[all_data.train_ind == 0].loc[lambda df: df.payment_period_as_at==(config['data'].cutoff + 1)].fillna(0)       # As at balance date
features = data_cols + ["backdate_periods"]

nn_train = nn_train_full.groupby(["claim_no", "development_period"]).sample(n=1, random_state=42)
nn_train.index.size == dat[dat.train_ind==1].index.size

nn_dat = pd.concat([nn_train.assign(train_ind=True), nn_test.assign(train_ind=False)])
# Run below instead to not use these ideas for now:
# nn_train = dat.loc[dat.train_ind == 1]
# nn_test = dat.loc[dat.train_ind == 0]

features = data_cols + ["backdate_periods"]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CV Rolling origin
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RollingOriginSplit:
    def __init__(self, start_cut, n_splits):
        self.start_cut = start_cut
        self.n_splits = n_splits

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Payment period for splits
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        quantiles = pd.qcut(groups, self.start_cut + self.n_splits + 1, labels=False)

        for split_value in range(self.start_cut, self.start_cut + self.n_splits):
            yield np.where(quantiles <= split_value)[0], np.where(quantiles == split_value + 1)[0]

ps = RollingOriginSplit(5, 5).split(groups=dat.loc[dat.train_ind == 1].payment_period)
for tr, te in ps:
    print(len(tr), len(te), len(tr)+ len(te) )


def claim_sampler(X, y):
    indices = torch.tensor(
        nn_train_full[["claim_no", "development_period", "data_as_at_development_period"]]
        .reset_index()
        .groupby(["claim_no", "development_period"])
        .sample(n=1)
        .index
    )
    return torch.index_select(X, 0, indices), torch.index_select(y, 0, indices)

use_batching_logic=True  # Set to False to omit this logic.

nn_train_full.loc[lambda df: df.claim_no == 2000].sort_values(["development_period", "data_as_at_development_period"])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FFNN training
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parameters_nn = {
    "l1_penalty": [0.0, 0.001, 0.01, 0.1],
    "weight_decay": [0.0, 0.001, 0.01, 0.1],
    "n_hidden": [5, 10, 20],
    # "interactions": [0.0, 0.25, 0.5, 0.75, 1.0],
    "dropout": [0, 0.25, 0.5],
    "max_iter": [config['training'].nn_cv_iter],
    "max_lr": [0.05],
    "verbose": [0],
    "clip_value": [None, 3.0],
    "keep_best_model": [True] 
}  

def build_ffnn_pipeline(model):
    return Pipeline(
        steps=[
        ('scaler', ColumnTransformer(transformers=[('scale', MinMaxScaler(), features)], remainder='drop')),  # Scale only features, drop others
        # Add a DataFrame wrapper to preserve DataFrame structure after scaling
        ('to_df', FunctionTransformer(lambda x: pd.DataFrame(x, columns=features), validate=False)),
            ('zero_to_one_2', MinMaxScaler()),       # Important! Standardize deep learning inputs.
            ("model", model),
        ]
    )

model_ffnn_detailed = build_ffnn_pipeline(
    RandomizedSearchCV(
        TabularNetRegressor(FeedForwardNet),
        parameters_nn,
        n_jobs=4,  # Run in parallel (small model)
        n_iter=config['training'].cv_runs, # Models train slowly, so try only a few models
        cv=RollingOriginSplit(5,5).split(groups=nn_train.payment_period),
        random_state=42
    )
)

def set_seed(seed):
    """
    Imposta il seed per la riproducibilità in Python, NumPy e PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

model_ffnn_detailed.fit(
    nn_train,
    nn_train.loc[:, ["payment_size"]]
)

bst_det = model_ffnn_detailed["model"].best_params_
print("best parameters:", bst_det)

cv_results_detailed = pd.DataFrame(model_ffnn_detailed["model"].cv_results_)


# Refit best model for longer iters
model_ffnn_detailed = build_ffnn_pipeline(
    TabularNetRegressor(
        FeedForwardNet, 
        l1_penalty=bst_det["l1_penalty"],
        weight_decay=bst_det["weight_decay"],  
        n_hidden=bst_det["n_hidden"],   
        dropout=bst_det["dropout"],                             
        max_iter=config['training'].nn_iter, 
        max_lr=bst_det["max_lr"],                
        batch_function=claim_sampler if use_batching_logic else None,                
        rebatch_every_iter=config['training'].mdn_iter/10,  # takes over 1s to resample so iterate a few epochs per resample                
    )
)

train_data = nn_train_full if use_batching_logic else nn_train
train_labels = train_data.loc[:, ["payment_size"]]

model_ffnn_detailed.fit(train_data, train_labels)   pd.DataFrame(X_batched.numpy(), columns=features),
        pd.DataFrame(y_batched.numpy(), columns=["payment_size"])
    )
else: 
    model_ffnn_detailed.fit(
        nn_train,
        nn_train.loc[:, ["payment_size"]]
    )