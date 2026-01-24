import pandas as pd
import numpy as np
import pprint

import matplotlib
from matplotlib import pyplot as plt

from datetime import datetime
import plotly.io as pio

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter


# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


import chainladder as cl
import random
import shap


# Local imports
from utils.config import load_config_from_yaml
from utils.neural_networks import TabularNetRegressor, FeedForwardNet, ColumnKeeper
from utils.data_engineering import load_data, process_data_davide
from utils.excel import save_df_to_excel
from utils.charts import chart_epoch_loss, chart_dual_QQ
from utils.shap import create_background_dataset  


# Load from YAML file
config = load_config_from_yaml('configs/NN_v_GBM_NJC_config.yaml')
pprint.pprint(config)


# Set pandas display options
pd.options.display.float_format = '{:,.2f}'.format

# Ensure matplotlib uses the positron backend for interactive plots
matplotlib.use('module://positron.matplotlib_backend') # Or 'inline' if using a notebook console

SEED = config['training'].seed 
rng = np.random.default_rng(SEED) 

# Create timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"FFNN_vs_GBM_experiment_NJC_{timestamp}"  # Customize as needed

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{run_name}")
log_filename = f"logs/{run_name}.xlsx"


print(f"Experiment timestamp: {timestamp}")
print(f"Output file: {log_filename}")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Loading and Initial Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load data
dat_orig = load_data(config)
save_df_to_excel(dat_orig, df_name="Original Data", filename=log_filename, mode='w')

dat = process_data_davide(config, dat_orig)
#save_df_to_excel(dat, df_name="Processed Data", filename=log_filename, mode='a')

data_cols = config['data'].data_cols
set(data_cols) - set(dat.columns)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
## Chain Ladder - from aggregated triangle
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


triangle = cl.Triangle(
    data = dat.loc[dat.payment_period <= config['data'].cutoff],
    origin=["occurrence_date"],
    origin_format="%Y%Q",
    development=["payment_date"],
    development_format="%Y%Q",
    columns=["payment_size_cumulative"],
    cumulative=True,
)

triangle

basic_chain_ladder = cl.Pipeline(
    steps=[
    ('dev', cl.Development(average='volume')),
    ('model', cl.Chainladder())])

basic_chain_ladder.fit(triangle)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data augmentation for NN and GBM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


sample_df = dat.loc[dat.claim_no==2000, ["claim_no", "occurrence_period", "train_ind", "payment_size"] + data_cols]
sample_df

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

backdate(sample_df, backdate_periods=1, keep_cols=data_cols)

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


#Sampling to reduce increase in batch size due to data augmentation

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
# CV Rolling origin prior to NN and GBM training
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

fig = chart_epoch_loss(
        dataframe = df_eval,
        x_col_name='epoch',
        y_col_names=['training_rmse', 'testing_rmse'],
        y_col_colour_map={'training_rmse': 'cyan', 'testing_rmse': 'red'},
        y_col_line_styles = {'training_rmse': 'solid', 'testing_rmse': 'dash'},
        title="RMSE over Epochs",)
fig.show()


## --- PLOT 2: LOSS ---
fig = chart_epoch_loss(
    dataframe = df_eval,
    x_col_name='epoch',
    y_col_names=['training_loss', 'testing_loss'],
    y_col_colour_map={'training_loss': 'blue', 'testing_loss': 'orange'},
    y_col_line_styles = {'training_loss': 'solid', 'testing_loss': 'dash'},
    title="Loss over Epochs",)
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

fig = chart_dual_QQ(X_sum_test, x_col_name='payment_size', y_col_name='pred_ffnn_claims')
fig.show()


#train_pred = generate_enhanced_tensorboard_outputs(model_ffnn_detailed, nn_train, config, writer=writer)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# GBM
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


set_seed(SEED)

parameters_gbm = {
    "max_iter": [500, 1000, 2000, 5000], 
    "l2_regularization": [0, 0.001, 0.01, 0.1, 0.3],
    "loss": ["poisson"],
    "learning_rate": [0.03, 0.07, 0.1],
    "verbose": [1]  
  #  "early_stopping": [False]
}


gradient_boosting = Pipeline(
    steps=[
        ("keep", ColumnKeeper(features)),         
        ('gbm', RandomizedSearchCV(
                  HistGradientBoostingRegressor(verbose=1),
                  parameters_gbm,
                  n_jobs=-1, # Run in parallel
                  n_iter=config['training'].cv_runs, # Models train slowly, so try only a few models
                  cv=RollingOriginSplit(5,5).split(groups=nn_train.payment_period),
                  random_state=0)),
    ]
)

gradient_boosting.fit(
    nn_train,
    nn_train.payment_size
)

#print(gradient_boosting["gbm"].best_params_)

# Dopo il training, ottieni il miglior modello
best_model = gradient_boosting["gbm"].best_estimator_

gradient_boosting["gbm"].best_params_

# Le loss sono già salvate in questi attributi
train_losses = -best_model.train_score_
val_losses = -best_model.validation_score_  # Se c'è early stopping

print("Train losses:", train_losses)
print("Validation losses:", val_losses)

# Create a DataFrame for the data
df_eval = pd.DataFrame({
    'iteration': range(0,len(train_losses)),
    'training_loss': train_losses,
    'val_loss': val_losses,
})


## --- PLOT 2: LOSS ---
fig = chart_epoch_loss(
    dataframe = df_eval,
    x_col_name='iteration',
    y_col_names=['training_loss', 'val_loss'],
    y_col_colour_map={'training_loss': 'blue', 'val_loss': 'orange'},
    y_col_line_styles = {'training_loss': 'solid', 'val_loss': 'dash'},
    title="Loss over iterations",)
fig.show()


dat["pred_gbm_claims"] = gradient_boosting.predict(dat)

test_data = dat.loc[dat.train_ind == 0].copy()
test_data["pred_gbm_claims_quantile"] = pd.qcut(test_data["pred_gbm_claims"], 5000, labels=False, duplicates='drop') # Group the test set predictions into 1000 quantiles
X_sum_test = test_data.groupby("pred_gbm_claims_quantile").agg("mean").reset_index() # Calculate the mean for each quantile in the test set


fig = chart_dual_QQ(X_sum_test, x_col_name='payment_size', y_col_name='pred_gbm_claims')
fig.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# SHAP values GBM vs NN
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FFNN SHAP

feature_names_FFNN = features 
best_model_FFNN = model_ffnn_detailed["model"]

X_processed_FFNN = model_ffnn_detailed["zero_to_one_2"].transform(
    model_ffnn_detailed["scaler"].transform(nn_train)
)

X_tensor_processed_FFNN = torch.from_numpy(X_processed_FFNN).to(best_model_FFNN.device)
background_data_FFNN = create_background_dataset(X_tensor_processed_FFNN, n_samples=1000)
X_processed_1000_FFNN = background_data_FFNN.cpu().numpy()


def predict_fn_log(X):
    if hasattr(X, 'values'):
        X = X.values
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    raw_preds = best_model_FFNN.predict(X)
    
    # Take the log to move back to the additive space
    # Adding a tiny epsilon prevents log(0) errors
    return np.log(raw_preds + 1e-9)

# SHAP

explainer_FFNN = shap.KernelExplainer(predict_fn_log, X_processed_1000_FFNN)
shap_values_FFNN = explainer_FFNN.shap_values(X_processed_1000_FFNN)



print("=== SHAP ANALYSIS ===")
print(f"Observation Number: {len(shap_values_FFNN)}")
print(f"Features Number: {len(feature_names_FFNN)}")

# Feature importance 
shap_df_FFNN = pd.DataFrame({
    'feature': feature_names_FFNN,
    'importance': np.abs(shap_values_FFNN).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTOP 10 FEATURES:")
print(shap_df_FFNN.head(10))

# Plot 
plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values_FFNN, X_processed_1000_FFNN, 
                 feature_names=feature_names_FFNN, show=False)
plt.title('SHAP Feature Importance - Neural Network', fontsize=16)
plt.tight_layout()
plt.show()

# Bar plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_FFNN, X_processed_1000_FFNN, 
                 feature_names=feature_names_FFNN, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16)
plt.tight_layout()
plt.show()


# GBM SHAP

def create_sample_dataset(X: pd.DataFrame, n_samples: int = 100):
    """
    Create background dataset for SHAP explainer from training data.
    """
    if X.shape[0] > n_samples:
        SEED = 42 
        rng = np.random.default_rng(SEED) 
        
        indices = rng.choice(X.shape[0], size=n_samples, replace=False)
        return X.iloc[indices]  # Changed from X[indices] to X.iloc[indices]
    else:
        return X

nn_train_1000 = create_sample_dataset(nn_train, 1000)


best_model_GBM = gradient_boosting["gbm"].best_estimator_
X_processed_GBM = gradient_boosting["keep"].transform(nn_train_1000)

# SHAP expaliner creation
explainer_GBM = shap.Explainer(best_model_GBM, X_processed_GBM)

# SHAP Calculation
shap_values_GBM = explainer_GBM(X_processed_GBM)  # Prime 1000 osservazioni

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_GBM, X_processed_GBM, show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values_GBM, show=False)
plt.title('SHAP Feature Importance (Bar Plot)')
plt.tight_layout()
plt.show()


# SHAP waterfalll for first obs

# 1. Generate the explanation for the first observation
# We use the explainer you already created
first_obs_processed = X_processed_1000_FFNN[0:1] # Keep as 2D array
shap_values_single = explainer_FFNN.shap_values(first_obs_processed)

# 2. Manually create the Explanation object
# KernelExplainer.shap_values returns a list for multi-output or a numpy array.
# We need to extract the base value (expected_value) from the explainer.
exp = shap.Explanation(
    values=shap_values_single[0], 
    base_values=explainer_FFNN.expected_value, 
    data=X_processed_FFNN[0], 
    feature_names=feature_names_FFNN
)

# 3. Plot the waterfall
plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, show=False)
plt.title('SHAP FFNN Waterfall prediction plot for row 0')
plt.tight_layout()
plt.show()


# Waterfall plot for the first observation
shap.plots.waterfall(shap_values_GBM[0], show=False)
plt.title('SHAP GBM Waterfall prediction plot for row 0 ')
plt.tight_layout()
plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
## Chain Ladder - transform aggregated results to individual transactions
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# creating predictions at claim transaction level

231164341.99 / 	54288.11

# extract the ldf and cdf
df_ldf = basic_chain_ladder.named_steps.dev.ldf_.T.reset_index(drop=True)
df_ldf['cdf'] = basic_chain_ladder.named_steps.dev.cdf_.T.reset_index(drop=True)
df_ldf['pcnt_ult'] = 1 / df_ldf['cdf']
df_ldf['development_period'] = range(0, df_ldf.shape[0])

# Add a new row with development_period=39 and all other values=1
new_row = pd.DataFrame({col: 1 for col in df_ldf.columns}, index=[0])
new_row['development_period'] = 39.0
df_ldf = pd.concat([df_ldf, new_row], ignore_index=True)



last_known_payment = dat.loc[dat.train_ind == True] \
                          .groupby('claim_no') \
                          .agg(
                              last_payment_size=('payment_size_cumulative', 'last'),
                              last_dev_period=('development_period', 'last')
                          ).reset_index()

cl_predictions = last_known_payment.merge(df_ldf[['development_period', 'cdf', 'pcnt_ult']],
                                          how='left',
                                          left_on='last_dev_period',
                                          right_on='development_period')

cl_predictions['cl_ultimate_prediction'] = cl_predictions['last_payment_size'] * cl_predictions['cdf']

dat_with_cl = dat.merge(cl_predictions[['claim_no', 'cl_ultimate_prediction']], how='left', on='claim_no')
dat_with_cl = dat_with_cl.merge(df_ldf[['development_period', 'pcnt_ult']], how='left', on='development_period')
dat_with_cl['cl_cumulative_prediction'] = dat_with_cl['cl_ultimate_prediction'] * dat_with_cl['pcnt_ult']


dat_with_cl['cl_incr_prediction'] = dat_with_cl['cl_cumulative_prediction'] - dat_with_cl.groupby('claim_no')['cl_cumulative_prediction'].shift(1).fillna(0)


# Create the final granular dataset with all predictions
granular_predictions = dat_with_cl[[
    'claim_no', 'occurrence_period', 'development_period',
    'payment_period', 'payment_size', 'train_ind',
    'cl_incr_prediction', 'pred_ffnn_claims', 'pred_gbm_claims'
]].copy()

granular_predictions.rename(columns={
    'payment_size': 'actual_payment_incremental',
    'pred_ffnn_claims': 'ffnn_prediction_incremental',
    'pred_gbm_claims': 'gbm_prediction_incremental',
    'cl_incr_prediction': 'chain_ladder_prediction_incremental'}, inplace=True)

granular_predictions['paid_to_valn_date'] = granular_predictions['actual_payment_incremental']
granular_predictions.loc[granular_predictions['train_ind'] == False, 'paid_to_valn_date'] = 0

print("Final Granular Dataset with All Model Predictions:")
display(granular_predictions.head())



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
## Ultimate Summary - GBM VS NN
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# need to change this to be better paids vs ults etc.


# Filter for test data only (the part of the triangle we are predicting)
#test_data = granular_predictions[granular_predictions['train_ind'] == False].copy()

df_ultimate_summary = granular_predictions.groupby('occurrence_period').agg({
    'paid_to_valn_date': 'sum',
    'actual_payment_incremental': 'sum',
    'chain_ladder_prediction_incremental': 'sum',
    'ffnn_prediction_incremental': 'sum',
    'gbm_prediction_incremental': 'sum'
}).reset_index()

df_ultimate_summary

#
## Individual Claim Level Loss
#
# Here we evaluate the RMSE on the test part of the dataset to quantify the loss for each model.
# In this part we show the losses for each occurrence period and for each development month.Chain Ladder - as GLM from individual
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Filter for test data only (the part of the triangle we are predicting)
test_data = granular_predictions[granular_predictions['train_ind'] == False].copy()

if test_data.empty:
    print("The test set is empty. Performance evaluation is not possible.")
else:
    # Calculate the squared error for each model
    test_data['sq_err_cl'] = (test_data['chain_ladder_prediction_incremental'] - test_data['actual_payment_incremental'])**2
    test_data['sq_err_ffnn'] = (test_data['ffnn_prediction_incremental'] - test_data['actual_payment_incremental'])**2
    test_data['sq_err_gbm'] = (test_data['gbm_prediction_incremental'] - test_data['actual_payment_incremental'])**2
    
    # Aggregate the squared errors by 'occurrence_period' and calculate the RMSE
    rmse_by_period = test_data.groupby('occurrence_period')[['sq_err_cl', 'sq_err_ffnn', 'sq_err_gbm']].mean()

    # Aggregate the squared errors by 'development' and calculate the RMSE
    rmse_by_development = test_data.groupby('development_period')[['sq_err_cl', 'sq_err_ffnn', 'sq_err_gbm']].mean()
    
    rmse_by_period = np.sqrt(rmse_by_period)
    rmse_by_development = np.sqrt(rmse_by_development)

    rmse_by_period.rename(columns={
        'sq_err_cl': 'RMSE_Chain_Ladder',
        'sq_err_ffnn': 'RMSE_ffnn',
        'sq_err_gbm': 'RMSE_GBM'
    }, inplace=True)

    rmse_by_development.rename(columns={
        'sq_err_cl': 'RMSE_Chain_Ladder',
        'sq_err_ffnn': 'RMSE_ffnn',
        'sq_err_gbm': 'RMSE_GBM'
    }, inplace=True)
    

    # RMSE (on the entire test set)
    total_rmse = pd.DataFrame({
        'RMSE_Chain_Ladder': [np.sqrt(test_data['sq_err_cl'].mean())],
        'RMSE_ffnn': [np.sqrt(test_data['sq_err_ffnn'].mean())],
        'RMSE_GBM': [np.sqrt(test_data['sq_err_gbm'].mean())]
    }, index=['Total RMSE'])

    print("\n## Overall Performance Summary\n")
    display(total_rmse)

print("## RMSE by Occurrence Period (on test data)\n")
display(rmse_by_period)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
## Occurence and Dev period RMSE plots
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


style_mapping = {
    'RMSE_Chain_Ladder': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-.'},  # blue
    'RMSE_ffnn': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},  # orange
    'RMSE_GBM': {'color': '#2ca02c', 'marker': '*', 'linestyle': '-'},  # green
}

# Create figure with two side-by-side subplots (due in questo caso, uno sopra l'altro)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))


# Plot 1: RMSE by Occurrence Period

ax1.plot(rmse_by_period.index, rmse_by_period['RMSE_Chain_Ladder'],
         **style_mapping['RMSE_Chain_Ladder'], label='RMSE Chain Ladder')
ax1.plot(rmse_by_period.index, rmse_by_period['RMSE_ffnn'],
         **style_mapping['RMSE_ffnn'], label='RMSE FFNN')
ax1.plot(rmse_by_period.index, rmse_by_period['RMSE_GBM'],
         **style_mapping['RMSE_GBM'], label='RMSE GBM')

ax1.set_title('RMSE per Occurrence Period (Test Data)', fontsize=16)
ax1.set_xlabel('Occurrence Period', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_xticks(rmse_by_period.index)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

# RMSE by Development Period

ax2.plot(rmse_by_development.index, rmse_by_development['RMSE_Chain_Ladder'],
         **style_mapping['RMSE_Chain_Ladder'], label='RMSE Chain Ladder')
ax2.plot(rmse_by_development.index, rmse_by_development['RMSE_ffnn'],
         **style_mapping['RMSE_ffnn'], label='RMSE FFNN')
ax2.plot(rmse_by_development.index, rmse_by_development['RMSE_GBM'],
         **style_mapping['RMSE_GBM'], label='RMSE GBM')

ax2.set_title('RMSE per Development Period (Test Data)', fontsize=16)
ax2.set_xlabel('Development Period', fontsize=12)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_xticks(rmse_by_development.index)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

