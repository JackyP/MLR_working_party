import pandas as pd
from typing import Tuple

from config import ExperimentConfig

def load_data(config: ExperimentConfig) -> pd.DataFrame:
    """
    Load the claims data.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Processed DataFrame
    """
    # Read in data
    #dat = pd.read_csv(
    #    config.data.data_dir + config.data.filename
    #)
    # https://github.com/agi-lab/SPLICE

    dat = pd.read_csv(
        f"https://raw.githubusercontent.com/agi-lab/SPLICE/main/datasets/complexity_1/payment_1.csv"
    )
    return dat

def process_data(config: ExperimentConfig, dat: pd.DataFrame) -> pd.DataFrame:
    """
    Process the claims data.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Processed DataFrame
    """
    
    # Data engineering - adds extra columns to dataset
    dat["train_ind_time"] = (dat.payment_period <= config.data.cutoff1)
    dat["test_ind_time"] = (dat.payment_period <= config.data.cutoff)
    dat["train_settled"] = (dat.settle_period <= config.data.cutoff)
    dat["settled_flag"] = (dat.settle_period <= config.data.cutoff1)
    
    dat['is_settled'] = dat['is_settled'].astype(int)
    dat["is_settled_future"] = (dat.is_settled)
    dat.loc[dat['payment_period'] > config.data.cutoff, 'is_settled_future'] = -1
    dat["future_flag"] = ~dat["train_ind_time"]
    
    dat["future_paid_cum"] = (dat.log1_paid_cumulative)
    dat.loc[dat['payment_period'] > config.data.cutoff, 'future_paid_cum'] = 12.3
    
    dat["L250k"] = 0
    dat.loc[dat['claim_size'] > 250000, 'L250k'] = 1
    
    # Create current development mappings
    currentdev = dat[dat['payment_period'] == config.data.cutoff].set_index('claim_no')['development_period'].to_dict()
    dat['curr_dev'] = dat['claim_no'].map(currentdev)
    #dat["curr_dev"].fillna(0, inplace=True)
    dat["curr_dev"] = dat["curr_dev"].fillna(0)
    
    currentpaid = dat[dat['payment_period'] == config.data.cutoff].set_index('claim_no')['log1_paid_cumulative'].to_dict()
    dat['curr_paid'] = dat['claim_no'].map(currentpaid)
    #dat["curr_paid"].fillna(0, inplace=True)
    dat["curr_paid"] = dat["curr_paid"].fillna(0)
    
    currentpmtno = dat[dat['payment_period'] == config.data.cutoff].set_index('claim_no')['pmt_no'].to_dict()
    dat['curr_pmtno'] = dat['claim_no'].map(currentpmtno)
    #dat["curr_pmtno"].fillna(0, inplace=True)
    dat["curr_pmtno"] = dat["curr_pmtno"].fillna(0)
    
    return dat



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset Creation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_train_test_datasets(dat: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create training and test datasets from processed data.
    
    Args:
        dat: Processed DataFrame
        config: Experiment configuration
        
    Returns:
        Tuple of (trainx, y_train, testx, y_test)
    """
    features = config.data.features
    youtput = config.data.output_field
    
    # Training data: settled claims within training time period
    trainx = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 1) & (dat.train_settled == 1), 
        features + ["claim_no"]
    ]
    y_train = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 1) & (dat.train_settled == 1)
    ].groupby('claim_no')[youtput].last()
    
    # Test data: unsettled claims not in training set
    testx = dat.loc[
        (dat.test_ind_time == 1) & (dat.train_ind == 0) & (dat.train_settled == 0),
        features + ["claim_no"]
    ]
    y_test = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 0) & (dat.train_settled == 0)
    ].groupby('claim_no')[youtput].last()
    
    return trainx, y_train, testx, y_test
