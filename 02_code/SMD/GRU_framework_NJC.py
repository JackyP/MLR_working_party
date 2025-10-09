##########################################
# Python script file equivalent of GRU 
# notebook - Refactored with SHAP Integration
#
# This file has been refactored to improve efficiency, readability,
# and includes SHAP explanations for model interpretability.
###########################################

import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import time
from datetime import datetime

# PyTorch imports

from torch.utils.tensorboard import SummaryWriter

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


# Local imports
from utils.config import get_default_config, ExperimentConfig
from utils.neural_networks import TabularNetRegressor, BasicLogGRU, ColumnKeeper, Make3D

#from data_engineering import load_data, process_data_davide, create_train_test_datasets_davide
from utils.data_engineering import load_data, process_data, create_train_test_datasets
from utils.tensorboard import generate_enhanced_tensorboard_outputs, create_actual_vs_expected_plot

from utils.excel import save_df_to_excel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configuration Setup
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

matplotlib.use('Agg')

# Load configuration
config = get_default_config()

# Set pandas display options
pd.options.display.float_format = '{:,.2f}'.format

SEED = 42 
rng = np.random.default_rng(SEED) 
writer = SummaryWriter() 



# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_NJC_GRU_outputs_{timestamp}.xlsx"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Loading and Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load data
dat_orig = load_data(config)

save_df_to_excel(dat_orig, df_name="Original Data", filename=log_filename, mode='w')

#dat = process_data_davide(config, dat_orig)
dat = process_data(config, dat_orig)

save_df_to_excel(dat, df_name="Processed Data", filename=log_filename, mode='a')

# Create datasets
#trainx, y_train, testx, y_test = create_train_test_datasets_davide(dat, config)
trainx, y_train, testx, y_test = create_train_test_datasets(dat, config)

save_df_to_excel(trainx, df_name="x_train", filename=log_filename, mode='a')
save_df_to_excel(y_train, df_name="y_train", filename=log_filename, mode='a')
save_df_to_excel(testx, df_name="x_test", filename=log_filename, mode='a')
save_df_to_excel(y_test, df_name="y_test", filename=log_filename, mode='a')

# Extract configuration values for backwards compatibility
features = config.data.features
data_cols = config.data.data_cols
youtput = config.data.output_field

# Print dataset info
nclms = trainx['claim_no'].nunique()
print(f"Training dataset - DataFrame: {isinstance(trainx, pd.DataFrame)}, Claims: {nclms}")
nfeatures = len(features)
print(f"Number of features: {nfeatures}")  


 

preprocessor = ColumnTransformer(
    transformers=[('scale', MinMaxScaler(), features)],
    remainder='passthrough',
    verbose_feature_names_out=False  # Optional: keeps original names cleaner
    )

preprocessor.set_output(transform="pandas")


model_NN = Pipeline(
    steps=[
        ("keep", ColumnKeeper(data_cols)),   
        ('zero_to_one', preprocessor),       # Important! Standardize deep learning inputs.
        ('3Dtensor', Make3D(features)),
        ("model", TabularNetRegressor(
            BasicLogGRU, 
            #n_input=nfeatures, 
            n_hidden=config.model.n_hidden, 
            #n_output=config.model.n_output, 
            max_iter=config.training.nn_iter,
            enable_shap=config.training.enable_shap,
            shap_log_frequency=config.training.shap_log_frequency
        ))
    ]
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Model Training
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_model(model, trainx, y_train, config: ExperimentConfig):
    """
    Train the neural network model with timing.
    
    Args:
        model: The model pipeline to train
        trainx: Training features
        y_train: Training targets
        config: Experiment configuration
        
    Returns:
        Trained model and elapsed time
    """
    print("Starting model training...")
    start_time = time.time()
    
    model.fit(trainx, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed. Execution time: {elapsed_time:.6f} seconds")
    
    return model, elapsed_time

# Train the model
trained_model, training_time = train_model(model_NN, trainx, y_train, config)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Log outputs to tensorboard
# Generate enhanced outputs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_pred = generate_enhanced_tensorboard_outputs(trained_model, dat, config)

save_df_to_excel(train_pred, df_name="pred_train", filename=log_filename, mode='a')

y_pred=trained_model.predict(trainx)

save_df_to_excel(pd.DataFrame(y_pred, columns=['prediction']), df_name="y_pred_train", filename=log_filename, mode='a')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create plots TRAIN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# AvsE all - Train
fig, ax = create_actual_vs_expected_plot(
    train_pred, youtput, "pred_claims", 
    'Train AvsE all history', 
    writer, 'AvsE all Train'
)

# Logged AvsE all - Train
fig, ax = create_actual_vs_expected_plot(
    train_pred, "log_actual", "log_pred_claims", 
    'Logged Train AvsE All History', 
    writer, 'AvsE Logged All Train',
    max_val=20
)

# AvsE Ult - Train
dat_byclaim = train_pred.groupby("claim_no").last()
fig, ax = create_actual_vs_expected_plot(
    dat_byclaim, "claim_size", "pred_claims", 
    'Train AvsE Ult only', 
    writer, 'AvsE Ult only Train'
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# QQ plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_pred["pred_claims_20cile"] = pd.qcut(train_pred["pred_claims"], 20, labels=False, duplicates='drop')
X_sum = train_pred.groupby("pred_claims_20cile").agg("mean").reset_index()

# QQ - Train
fig, ax  = create_actual_vs_expected_plot(
    X_sum, "claim_size", "pred_claims", 
    'Train QQ plot 20', 
    writer, 'QQ plot Train'
)

# Logged QQ - Train
fig, ax  = create_actual_vs_expected_plot(
    X_sum, "log_actual", "log_pred_claims", 
    'Logged Train QQ plot 20', 
    writer, 'QQ plot Logged Train'
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create plots TEST
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test = (dat.loc[(dat.test_ind_time == 1) & (dat.train_ind == 0) & (dat.train_settled == 0)])

youtput="claim_size"
y_pred=model_NN.predict(test)

y_predx=model_NN.predict(testx)

#merge y_pred back into dat for each claim
claim_nos = test["claim_no"].drop_duplicates()
pred_df = pd.DataFrame({
    "claim_no": claim_nos.values,
    "pred_claims": y_pred
})

if "pred_claims" in test.columns:
    dat = dat.drop(columns=["pred_claims"])

test_pred = test.merge(pred_df, on="claim_no", how="left")

test_pred["log_pred_claims"]=test_pred["pred_claims"].apply(lambda x: np.log(x+1))
test_pred["log_actual"]=test_pred[youtput].apply(lambda x: np.log(x+1))

test_pred["rpt_delay"]=np.ceil(train_pred.notidel).astype(int)

test_pred["diff"]=test_pred[youtput]-train_pred["pred_claims"]
test_pred["diffp"]=(test_pred[youtput]-test_pred["pred_claims"])/test_pred[youtput]

save_df_to_excel(test_pred, df_name="pred_test", filename=log_filename, mode='a')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create plots TEST
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# AvsE all - Test
create_actual_vs_expected_plot(
    test_pred, youtput, "pred_claims", 
    'Test AvsE all history', 
    writer, 'AvsE all Test'
)

# Logged AvsE all - Test
fig, ax = create_actual_vs_expected_plot(
    test_pred, "log_actual", "log_pred_claims", 
    'Logged Test AvsE All History', 
    writer, 'AvsE Logged All Test',
    max_val=20
)

# AvsE Ult - Test
dat_byclaim = test_pred.groupby("claim_no").last()
fig, ax = create_actual_vs_expected_plot(
    dat_byclaim, "claim_size", "pred_claims", 
    'Test AvsE Ult only', 
    writer, 'AvsE Ult only Test'
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# QQ plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_pred["pred_claims_20cile"] = pd.qcut(test_pred["pred_claims"], 20, labels=False, duplicates='drop')
X_sum = test_pred.groupby("pred_claims_20cile").agg("mean").reset_index()

# QQ - Test
fig, ax  = create_actual_vs_expected_plot(
    X_sum, "claim_size", "pred_claims", 
    'Test QQ plot 20', 
    writer, 'QQ plot Test'
)

# Logged QQ - Test
fig, ax  = create_actual_vs_expected_plot(
    X_sum, "log_actual", "log_pred_claims", 
    'Logged Test QQ plot 20', 
    writer, 'QQ plot Logged Test'
)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Occurrence date plots
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

datTrainUlt=train_pred.groupby("claim_no").last()
datTestUlt=test_pred.groupby("claim_no").last()

datTrain_occ = datTrainUlt.groupby("occurrence_period").agg({youtput: "sum", "pred_claims": "sum"})
datTest_occ = datTestUlt.groupby("occurrence_period").agg({youtput: "sum", "pred_claims": "sum"})

plt.figure()

plt.plot(datTrain_occ.index, datTrain_occ[youtput])
plt.plot(datTrain_occ.index, datTrain_occ.pred_claims)

plt.plot(datTest_occ.index, datTest_occ[youtput])
plt.plot(datTest_occ.index, datTest_occ.pred_claims)

fig, ax = plt.subplots()
ax.plot(datTrain_occ.index, datTrain_occ[youtput], linestyle='--', label='Train Actual')
ax.plot(datTrain_occ.index, datTrain_occ.pred_claims, linestyle='--', label='Train Expected')
ax.plot(datTest_occ.index, datTest_occ[youtput], label='Test Actual')
ax.plot(datTest_occ.index, datTest_occ.pred_claims, label='Test Expected')
ax.set_yscale("log") 
ax.set_xlabel('Occurrence period', fontsize=15)
ax.set_ylabel('Total Ultimate claims', fontsize=15)
ax.set_title('by Occurrence Period')     
ax.legend()
writer.add_figure('by Occur Period', fig)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Development date plots
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

datTrain_dev = train_pred.groupby("development_period").agg({youtput: "sum", "pred_claims": "sum"})
datTest_dev = test_pred.groupby("development_period").agg({youtput: "sum", "pred_claims": "sum"})

plt.figure()

plt.plot(datTrain_dev.index, datTrain_dev[youtput])
plt.plot(datTrain_dev.index, datTrain_dev.pred_claims)

plt.plot(datTest_dev.index, datTest_dev[youtput])
plt.plot(datTest_dev.index, datTest_dev.pred_claims)


fig, ax = plt.subplots()
ax.plot(datTrain_dev.index, datTrain_dev[youtput], linestyle='--', label='Train Actual')
ax.plot(datTrain_dev.index, datTrain_dev.pred_claims, linestyle='--', label='Train Expected')
ax.plot(datTest_dev.index, datTest_dev[youtput], label='Test Actual')
ax.plot(datTest_dev.index, datTest_dev.pred_claims, label='Test Expected')
ax.set_xlabel('Development period', fontsize=15)
ax.set_ylabel('Total claims', fontsize=15)
ax.set_title('by Devevelopment Period')     
ax.legend()
writer.add_figure('by Dev Period', fig)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model weights and biases
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

keep_params = {}

fitted_model = model_NN.named_steps['model'].module_

for name, param in fitted_model.named_parameters():
    # Convert parameter tensor to numpy array and flatten it
    param_np = param.detach().numpy().flatten()

    # Create a DataFrame for the parameter
    keep_params[name] = pd.DataFrame(param_np, columns=[name])

for name, df in keep_params.items():
    # Flatten the DataFrame to a 1D array for easy plotting
    params = df.values.flatten()
    plt.bar(range(len(params)), params)
    plt.title(f'Parameters: {name}')
    plt.xlabel('Parameter Index')
    plt.ylabel('Value')
    plt.show()
    


for name, param in fitted_model.named_parameters():
    print(keep_params[name])



gradients = []

for param in fitted_model.parameters():
    if param.grad is not None:
        gradients.append(param.grad.view(-1).cpu().numpy())

plt.hist(np.concatenate(gradients), bins=100)
plt.title("Gradient Distribution")
plt.xlabel("Gradient Value")
plt.ylabel("Frequency")
plt.show()

