import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime
import plotly.express as px  # Add this import
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
run_name = f"FFNN_vs_GBM_experiment_NJC_{timestamp}"  # Customize as needed

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{run_name}")
log_filename = f"logs/{run_name}.xlsx"


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


# Don't log to tensorboard in CV (because sklearn can't pickle the writer object)
# Don't use n_jobs >1 as sklearn can't read from utils when parallelizing
model_ffnn_detailed = build_ffnn_pipeline(
    RandomizedSearchCV(
        TabularNetRegressor(FeedForwardNet, device="cpu"),
        parameters_nn,
        n_jobs=1,  # Run in parallel (small model)
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

config['training'].enable_shap
config['tensorboard'].shap_log_frequency
config['training'].nn_iter

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
        rebatch_every_iter=config['training'].mdn_iter/10,  # takes over 1s to resample so iterate a few epochs per resample,
        enable_shap=config['training'].enable_shap,
        shap_log_frequency=config['tensorboard'].shap_log_frequency,
        device = "cpu",
        seed=SEED,
        config = config,
        writer=writer,  # TensorBoard writer                
    )
)



if use_batching_logic:
    model_ffnn_detailed.fit(
        nn_train_full,
        nn_train_full.loc[:, ["payment_size"]]
    )
else: 
    model_ffnn_detailed.fit(
        nn_train,
        nn_train.loc[:, ["payment_size"]]
    )



# Access the model data
regressor_model = model_ffnn_detailed["model"]

# Assuming you have your test data loaded and handled correctly
if use_batching_logic:
    X_test = nn_test
    y_test = nn_test.loc[:, ["payment_size"]]
else: 
    X_test = nn_test
    y_test = nn_test.loc[:, ["payment_size"]]

# 1. Accedi all'oggetto del regressore all'interno della pipeline
regressor_model = model_ffnn_detailed["model"]

# 2. Trasforma i dati di test usando solo i passaggi di pre-processing della pipeline


transformer_pipeline = Pipeline(
    steps=[
        ("keep", model_ffnn_detailed.named_steps["scaler"]),
        ('zero_to_one', model_ffnn_detailed.named_steps["zero_to_one_2"]),
    ]
)
X_test_transformed = transformer_pipeline.transform(X_test)

test_loss, testing_rmses = regressor_model.get_testing_losses(X_test_transformed, y_test)

# Create a DataFrame for the data
df_eval = pd.DataFrame({
    'epoch': regressor_model.testing_epochs,
    'training_rmse': regressor_model.training_rmses_history,
    'testing_rmse': testing_rmses,
    'training_loss': regressor_model.training_losses_history,
    'testing_loss': test_loss,
})

# --- PLOT 1: RMSE ---
fig = px.line(
    df_eval, 
    x='epoch', 
    y=['training_rmse', 'testing_rmse'], 
    title="RMSE over Epochs",
    labels={'value': 'RMSE', 'variable': 'Type'},
    color_discrete_map={'training_rmse': 'cyan', 'testing_rmse': 'red'}
)
# Add markers to all lines
fig.update_traces(mode='lines+markers')

# Change the testing_rmse line to dashed style
fig.update_traces(
    selector={'name': 'testing_rmse'},
    line={'dash': 'dash'}
)

# Move the legend to the top center, under the title
fig.update_layout(
    legend=dict(
        orientation="h",  # 'h' for horizontal layout
        yanchor="bottom", # Anchor the legend's bottom edge
        y=1.02,           # Position the legend just above the plot area (1.0)
        xanchor="center", # Anchor the legend's center
        x=0.5             # Center the legend horizontally
    )
)

fig.show()

## --- PLOT 2: LOSS ---
fig = px.line(
    df_eval, 
    x='epoch', 
    y=['training_loss', 'testing_loss'], 
    title="Loss over Epochs",
    labels={'value': 'Loss', 'variable': 'Type'},
    color_discrete_map={'training_loss': 'blue', 'testing_loss': 'orange'}
)

# Add markers to all lines
fig.update_traces(mode='lines+markers')

# Change the testing_rmse line to dashed style
fig.update_traces(
    selector={'name': 'testing_loss'},
    line={'dash': 'dash'}
)

# Move the legend to the top center, under the title
fig.update_layout(
    legend=dict(
        orientation="h",  # 'h' for horizontal layout
        yanchor="bottom", # Anchor the legend's bottom edge
        y=1.02,           # Position the legend just above the plot area (1.0)
        xanchor="center", # Anchor the legend's center
        x=0.5             # Center the legend horizontally
    )
)

fig.show()


# QQ plot for test set

# Function to make a dataset with train payments and test predictions, and resulting triangle
def make_pred_set_and_triangle(individual_model, train, test):
    dat_model_pred = pd.concat(
        [
            train,
            test.assign(payment_size = individual_model.predict(test))
        ], 
        axis="rows"
    )
    dat_model_pred["payment_size_cumulative"] = (
        dat_model_pred[["claim_no", "payment_size"]].groupby('claim_no').cumsum()
    )

    triangle_model_ind = (dat_model_pred
        .groupby(["occurrence_period", "development_period", "payment_period"], as_index=False)
        .agg({"payment_size_cumulative": "sum", "payment_size": "sum"})
        .sort_values(by=["occurrence_period", "development_period"])
    )

    return dat_model_pred, triangle_model_ind


dat_ffnn_det_pred, triangle_ffnn_detailed = make_pred_set_and_triangle(model_ffnn_detailed, nn_train, nn_test)

dat["pred_ffnn_claims"] = model_ffnn_detailed.predict(dat)
test_data = dat.loc[dat.train_ind == 0].copy()
test_data["pred_ffnn_claims_quantile"] = pd.qcut(test_data["pred_ffnn_claims"], 5000, labels=False, duplicates='drop')  # Group the test set predictions into 5000 quantiles
X_sum_test = test_data.groupby("pred_ffnn_claims_quantile").agg("mean").reset_index()  # Calculate the mean for each quantile in the test set


# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Test QQ Plot (Original Values)', 'Test QQ Plot (Log-transformed Values)'))

# --- Plot 1: QQ Plot with Original Values ---
max_val = max(X_sum_test['payment_size'].max(), X_sum_test['pred_ffnn_claims'].max())

fig.add_trace(
    go.Scatter(
        x=X_sum_test['payment_size'],
        y=X_sum_test['pred_ffnn_claims'],
        mode='markers',
        name='Average by Quantile',
        marker=dict(color='blue', opacity=0.7),
        showlegend=False  # Disable global legend
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Ideal Prediction (y=x)',
        line=dict(color='red', dash='dash'),
        showlegend=False  # Disable global legend
    ),
    row=1, col=1
)

# --- Plot 2: QQ Plot with Log Values ---
log_pred = np.log1p(X_sum_test['pred_ffnn_claims'])
log_actual = np.log1p(X_sum_test['payment_size'])
max_log_val = max(log_pred.max(), log_actual.max())

fig.add_trace(
    go.Scatter(
        x=log_actual,
        y=log_pred,
        mode='markers',
        #name='Average by Quantile (Log)',
        marker=dict(color='blue', opacity=0.7),
        showlegend=False  # Disable global legend
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=[0, max_log_val],
        y=[0, max_log_val],
        mode='lines',
        #name='Ideal Prediction (y=x) (Log)',
        line=dict(color='red', dash='dash'),
        showlegend=False  # Disable global legend
    ),
    row=1, col=2
)

# Add separate legend annotations in the top left of each subplot
fig.add_annotation(
    text="<b>Legend</b><br>● Average by Quantile<br>━ Ideal Prediction (y=x)",
    #xref="x1", yref="y1",  # Reference to first subplot's axes
    # Set the reference frame for x and y to 'paper'
    xref='paper', 
    yref='paper',
    x=0, y=1,  # Top left position
    showarrow=False,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1,
    align="left"
)

fig.add_annotation(
    text="<b>Legend</b><br>● Average by Quantile (Log)<br>━ Ideal Prediction (y=x) (Log)",
    xref='paper', 
    yref='paper',
    x=0.75, y=1,  # Top left position
    showarrow=False,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1,
    align="left"
)

# Update axes
fig.update_xaxes(title_text='Actual Average per Quantile', row=1, col=1, showgrid=True)
fig.update_yaxes(title_text='Predicted Average per Quantile', row=1, col=1, showgrid=True)
fig.update_xaxes(title_text='Actual Average per Quantile (log-transformed)', row=1, col=2, showgrid=True)
fig.update_yaxes(title_text='Predicted Average per Quantile (log-transformed)', row=1, col=2, showgrid=True)

fig.update_layout(height=500, width=1000, title_text="QQ Plots for Test Set")

fig.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
# got here
#
# There is an issue with the log_shap_explanations function in utils/tensorboard.py
# 
# The issue relates to the partial dependencies loop where seem to have different dimensions 
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#train_pred = generate_enhanced_tensorboard_outputs(model_ffnn_detailed, nn_train, config, writer=writer)


# Local imports
from utils.config import  ExperimentConfig
from utils.shap import ShapExplainer, log_shap_explanations, create_background_dataset  

## SHAP (SHapley Additive exPlanations) FFNN

model = model_ffnn_detailed
dat = nn_train


print("Generating enhanced tensorboard outputs...")
youtput = config['data'].output_field

# Training set analysis
train_pred = dat[config['data'].features + ["claim_no", youtput]].copy()
train_pred.loc[:,'pred_claims'] = model.predict(train_pred)

       
# Feature engineering for analysis
train_pred["log_pred_claims"] = train_pred["pred_claims"].apply(lambda x: np.log(x+1))
train_pred["log_actual"] = train_pred[youtput].apply(lambda x: np.log(x+1))
train_pred["rpt_delay"] = np.ceil(train_pred.notidel).astype(int)
train_pred["diff"] = train_pred[youtput] - train_pred["pred_claims"]
train_pred["diffp"] = (train_pred[youtput] - train_pred["pred_claims"]) / train_pred[youtput]



print("Generating SHAP explanations for trained model...")
            
# Get the underlying neural network model
nn_model = model.named_steps['model'].module_
            
# Transform features through the pipeline (excluding the final model step)  
pipeline_steps = model.steps[:-1]  # All steps except the model
feature_pipeline = Pipeline(pipeline_steps)
X_transformed = feature_pipeline.transform(train_features)
            
# Convert to tensor
X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
            
# Create SHAP explainer
background_data = create_background_dataset(X_tensor, n_samples=100)
feature_names = config['data'].features
shap_explainer = ShapExplainer(nn_model, background_data, feature_names)
            
# Generate SHAP explanations for a sample of training data
sample_size = min(200, len(X_tensor))
#sample_indices = np.random.choice(len(X_tensor), sample_size, replace=False)
sample_indices = rng.choice(len(X_tensor), sample_size, replace=False)
X_sample = X_tensor[sample_indices]
            
# Log SHAP explanations to tensorboard
log_shap_explanations(
    writer, shap_explainer, X_sample, epoch=9999,  # Use high epoch number for final analysis
    prefix="Final_Model_SHAP", max_samples=sample_size
)
            

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Log SHAP explanations to tensorboard
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
prefix="Final_Model_SHAP"
epoch=9999

explainer = shap_explainer
shap_values = explainer.get_shap_values(X_sample, max_samples=200)

print(f'DEBUG: log_shap_explanations/X shape: {X_sample.shape}')
print(f'DEBUG: shap_values.shape: {shap_values.shape}') 
print(f'DEBUG: shap_values[0] shape: {shap_values[0].shape}')

len(shap_values.shape)
shap_values_2d = shap_values.reshape(-1, shap_values.shape[-1])

mean_abs_shap = np.mean(np.abs(shap_values_2d), axis=0)
print(f'{prefix}/mean_abs_shap: {mean_abs_shap}')

import matplotlib

matplotlib.use('Agg')
summary_fig = explainer.create_summary_plot(shap_values, X_sample, title=f"{prefix} Summary - Epoch {epoch}")
summary_fig.savefig('shap_summary_plot.png')

fig = plt.gcf()

summary_fig
summary_fig.show()
type(summary_fig)

trained_model = model_ffnn_detailed["model"]



    
# Generate predictions
y_pred = model_ffnn_detailed.predict(nn_train)


type(trained_model)
type(model_ffnn_detailed)

nn_train.shape
dat.shape



print("Original feature list:", features)

feature_names = features

try:
    feature_names = model_ffnn_detailed["scaler"].features
    print("Features da ColumnKeeper:", feature_names)
except AttributeError:
    try:
        feature_names = model_ffnn_detailed["keep"].get_feature_names_out()
        print("Features da get_feature_names_out():", feature_names)
    except:
        feature_names = features 
        print("Features original list:", feature_names)

# data preparation
best_nn_model = model_ffnn_detailed["model"]



X_processed = model_ffnn_detailed["zero_to_one_2"].transform(
    model_ffnn_detailed["scaler"].transform(nn_train)
)

if hasattr(X_processed, 'values'):
    X_processed = X_processed.values

# check on dimensions
if len(feature_names) != X_processed.shape[1]:
    print(f"Attention: {len(feature_names)} features names but {X_processed.shape[1]} columns!")
    
    feature_names = feature_names[:X_processed.shape[1]]
    print("Feature names:", feature_names)


def predict_fn(X):
    if hasattr(X, 'values'):
        X = X.values
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    return best_nn_model.predict(X)

# SHAP
explainer = shap.KernelExplainer(predict_fn, X_processed[:100])
shap_values = explainer.shap_values(X_processed[:1000])

print("=== SHAP ANALYSIS ===")
print(f"Observation Number: {len(shap_values)}")
print(f"Features Number: {len(feature_names)}")

# Feature importance 
shap_df = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTOP 10 FEATURES:")
print(shap_df.head(10))

# Plot 
import plotly.express as px

# Prepare data for Plotly summary plot (approximating SHAP beeswarm)
df_shap = pd.DataFrame(shap_values, columns=feature_names)
df_shap_long = df_shap.melt(var_name='feature', value_name='shap_value')

X_df = pd.DataFrame(X_processed[:1000], columns=feature_names)
X_long = X_df.melt(var_name='feature', value_name='feature_value')

df_plot = pd.concat([df_shap_long, X_long[['feature_value']]], axis=1)

# Sort features by mean absolute SHAP value
mean_abs_shap = df_shap.abs().mean().sort_values(ascending=False)
df_plot['feature'] = pd.Categorical(df_plot['feature'], categories=mean_abs_shap.index, ordered=True)

fig = px.strip(df_plot, x='shap_value', y='feature', color='feature_value', orientation='h',
               title='SHAP Feature Importance - Neural Network')
fig.update_layout(height=700, width=1000)
fig.show()

# Bar plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_processed[:1000], 
                 feature_names=feature_names, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16)
plt.tight_layout()
plt.show()