"""
Neural Network Models for Claims Reserving

File: data_engineering.py

This module contains helper functions to take simulated datasets from SPLICE and manipulate them into formats suitable
for claims reserving modeling.

The following functions are defined

- load_data: Load the claims data from a CSV file.
- process_data: Process the claims data into a structured format.
- create_train_test_datasets: Create training and test datasets from processed data.
- date_from_period: Convert a period number to a date string.
- process_data_davide: An alternative data processing function.
- create_train_test_datasets_davide: An alternative function to create training and test datasets.

"""


import pandas as pd
import numpy as np
from typing import Tuple
import datetime

from utils.config import ExperimentConfig


def load_data(config: ExperimentConfig) -> pd.DataFrame:
    """
    Load the claims data.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Processed DataFrame
    """
    # Read in data
    dat = pd.read_csv(
        config['data'].data_dir + config['data'].filename
    )
    
    return dat


def year_from_period(period_value: int, period_length_months = 3, base_year = 2000) -> int:
    """
    Given a zero indexed period_value returns a year.

    Args:
        period_value: An integer representing the period number (starting from 1)
        period_length_months: An integer specifying the length of the period in calendar months
        base_year: the start year assume we measure from the start of the start year!
    """
    year_from_period = base_year + ((period_value-1) * period_length_months) // 12
    
    return year_from_period


def quarter_from_period(period_value: int, period_length_months = 3) -> int:
    """
    Given a zero indexed period_value returns an integer specify the calendar quarter.

    Args:
        period_value: An integer representing the period number (starting from 0)
        period_length_months: An integer specifying the length of the period in calendar months
    """
    quarter_from_period = (((period_value-1) * period_length_months) % 12) // 3 + 1
    
    return quarter_from_period

def date_from_period(period: int, day = 15, base_year = 2000, return_format = "%d/%m/%Y") -> str:
    """
    Calculates a date (DD/MM/YYYY) based on a period number.

    The sequence starts on 15/02/2000 (Period 1) and progresses
    in quarterly steps (February, May, August, November) for subsequent periods.
    
    Args:
        period: An integer representing the period number (starting from 1).

    Returns:
        A string representing the date in 'DD/MM/YYYY' format.
    
    Raises:
        ValueError: If the period is less than 1.
    """
    if period < 1:
        raise ValueError("Period must be 1 or greater.")
      
    month_sequence = [2, 5, 8, 11] # Quarter-yearly months: Feb, May, Aug, Nov
    
    # Periods are 1-indexed, so we use (period - 1) for 0-indexed calculations
    zero_indexed_period = period - 1
    
    # Calculate the Year:
    # Every 4 periods (a full year cycle) the year increases by 1.
    # The first four periods (0, 1, 2, 3) fall in the base_year.
    year_offset = zero_indexed_period // len(month_sequence)
    year = base_year + year_offset
    
    # Calculate the Month:
    # The month cycles through the sequence [2, 5, 8, 11].
    # The index in the sequence is (period - 1) % 4.
    month_index = zero_indexed_period % len(month_sequence)
    month = month_sequence[month_index]
    
    # Create the datetime.date object
    date_obj = datetime.date(year, month, day)
    
    # Format and return the date as 'DD/MM/YYYY'
    return date_obj.strftime(return_format)


def process_data_davide(config: ExperimentConfig, dat: pd.DataFrame) -> pd.DataFrame:
    """
    Process the claims data.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Processed DataFrame
    """

    transactions = dat
    transactions["noti_period"] = np.ceil(transactions["occurrence_time"] + transactions["notidel"]).astype('int')
    transactions["settle_period"] = np.ceil(transactions["occurrence_time"] + transactions["notidel"] + transactions["setldel"]).astype('int')

    # Apply cut-off since some of the logic in this notebook assumes an equal set of dimensions
    transactions["development_period"] = np.minimum(transactions["payment_period"] - transactions["occurrence_period"], config['data'].cutoff)  
    num_dev_periods = config['data'].cutoff - 1  # (transactions["payment_period"] - transactions["occurrence_period"]).max()

    #transactions["occurence_date"] = transactions["occurrence_period"].apply(lambda x: date_from_period(x))
    #transactions["payment_date"] = transactions["payment_period"].apply(lambda x: date_from_period(x))
    #transactions["noti_date"] = (transactions["occurrence_period"] + transactions["noti_period"]).apply(lambda x: date_from_period(x))
    #transactions["settle_date"] = (transactions["occurrence_period"] + transactions["settle_period"]).apply(lambda x: date_from_period(x))

    # Transactions summarised by claim/dev:
    transactions_group = (transactions
            .groupby(["claim_no", "development_period"], as_index=False)
            .agg({"payment_size": "sum", "pmt_no": "max", "claim_size": "last"})
            .sort_values(by=["claim_no", "development_period"])
    )

    # This is varied from the original version:
    range_payment_delay = pd.DataFrame.from_dict({"development_period": range(0, num_dev_periods + 1)})

    # Claims header + development periods
    claim_head_expand_dev = (
        transactions
        .loc[:, ["claim_no", "occurrence_period", "occurrence_time", "noti_period", "notidel", "settle_period"]]
        .drop_duplicates()
    ).merge(
        range_payment_delay,
        how="cross"
    ).assign(
        payment_period=lambda df: (df.occurrence_period + df.development_period),
        is_settled=lambda df: (df.occurrence_period + df.development_period) >= df.settle_period,
        occurrence_date=lambda df: df.occurrence_period.apply(lambda x: date_from_period(x)),
        payment_date=lambda df: (df.occurrence_period + df.development_period).apply(lambda x: date_from_period(x)),

    )

    # create the dataset
    dat = claim_head_expand_dev.merge(
        transactions_group.drop(columns=['claim_size']),
        how="left",
        on=["claim_no", "development_period"],
    ).fillna(0)

    dat = dat.merge(
        transactions_group[['claim_no', 'claim_size']].drop_duplicates(),
        how="left",
        on=["claim_no"],
    )

    # Only periods after notification
    dat = dat.loc[dat.payment_period >= dat.noti_period]

    # Clean close to zero values
    dat["payment_size"] = np.where(abs(dat.payment_size) < 1e-2, 0.0, dat.payment_size)

    # Cumulative payments
    dat["payment_size_cumulative"] = dat[["claim_no", "payment_size"]].groupby('claim_no').cumsum()

    dat["payment_to_prior_period"] = dat["payment_size_cumulative"] - dat["payment_size"]
    dat["has_payment_to_prior_period"] = np.where(dat.payment_to_prior_period > 1e-2, 1, 0)
    dat["log1_payment_to_prior_period"] = np.log1p(dat.payment_to_prior_period)
    dat["log1_cumulative_payment_to_prior_period"] = np.log1p(dat.groupby("claim_no")["payment_size_cumulative"].shift(1).fillna(0))

    dat["pmt_no"] = dat.groupby("claim_no")["pmt_no"].cummax()
    dat["payment_count_to_prior_period"] = dat.groupby("claim_no")["pmt_no"].shift(1).fillna(0)
    
    dat["data_as_at_development_period"] = dat.development_period  # See data augmentation section
    dat["backdate_periods"] = 0
    dat["payment_period_as_at"] = dat.payment_period


    dat['occurrence_date'] = pd.to_datetime(dat['occurrence_date']).dt.to_period('Q')
    dat['payment_date'] = pd.to_datetime(dat['payment_date']).dt.to_period('Q')


    # Potential features for model later:
    #data_cols = [
    #    "notidel", 
    #    "occurrence_time", 
    #    "development_period", 
    #    "payment_period", 
    #    "has_payment_to_prior_period",
    #    "log1_payment_to_prior_period",
    #    "payment_count_to_prior_period",
    #    ]

    # The notebook is created on an MacBook Pro M1, which supports GPU mode in float32 only.
    dat[dat.select_dtypes(np.float64).columns] = dat.select_dtypes(np.float64).astype(np.float32)

    dat["train_ind"] = (dat.payment_period <= config['data'].cutoff)
    dat["cv_ind"] = dat.payment_period % 5  # Cross validate on this column

    return dat


def process_data(config: ExperimentConfig, dat: pd.DataFrame) -> pd.DataFrame:
    """
    Process the claims data.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Processed DataFrame
    """
    
    dat["noti_period"] = np.ceil(dat["occurrence_time"] + dat["notidel"]).astype('int')
    dat["settle_period"] = np.ceil(dat["occurrence_time"] + dat["notidel"] + dat["setldel"]).astype('int')

    dat["development_period"] = np.minimum(dat["payment_period"] - dat["occurrence_period"], config['data'].maxdev)  
    

    dat["occurrence_date"] = dat["occurrence_period"].apply(lambda x: date_from_period(x))
    dat["payment_date"] = dat["payment_period"].apply(lambda x: date_from_period(x))
    dat["noti_date"] = (dat["occurrence_period"] + dat["noti_period"]).apply(lambda x: date_from_period(x))
    dat["settle_date"] = (dat["occurrence_period"] + dat["settle_period"]).apply(lambda x: date_from_period(x))

    # Clean close to zero values
    dat["payment_size"] = np.where(abs(dat.payment_size) < 1e-2, 0.0, dat.payment_size)

    # payment_period=lambda df: (df.occurrence_period + df.development_period),
    dat["is_settled"]=((dat.occurrence_period + dat.development_period) >= dat.settle_period)
    dat = dat.loc[~dat.is_settled].copy()

    # Data engineering - adds extra columns to dataset

    dat["train_ind"] = (dat.claim_no % 10 >= 4)
    dat["train_ind_time"] = (dat.payment_period <= config['data'].cutoff)
    #dat["cv_ind"] = dat.payment_period % 5

    #dat["train_ind"] = (dat.payment_period <= config['data'].cutoff)
    
    #dat["train_ind_time"] = (dat.payment_period <= config['data'].cutoff1)
    dat["test_ind_time"] = (dat.payment_period <= config['data'].cutoff)
    dat["train_settled"] = (dat.settle_period <= config['data'].cutoff)
    dat["settled_flag"] = (dat.settle_period <= config['data'].cutoff1)
    
    # Cumulative payments
    dat["payment_size_cumulative"] = dat[["claim_no", "payment_size"]].groupby('claim_no').cumsum()
    dat["log1_paid_cumulative"] = np.log1p(dat.payment_size_cumulative)

    dat["pmt_no"] = dat.groupby("claim_no")["pmt_no"].cummax()

    # Second stage repetition
    
    dat["train_ind_time"] = (dat.payment_period <= config['data'].cutoff1)
    dat["test_ind_time"] = (dat.payment_period <= config['data'].cutoff)
    dat["train_settled"] = (dat.settle_period <= config['data'].cutoff)
    dat["settled_flag"] = (dat.settle_period <= config['data'].cutoff1)

    dat['is_settled'] = dat['is_settled'].astype(int)
    dat["is_settled_future"] = (dat.is_settled)
    dat.loc[dat['payment_period'] > config['data'].cutoff, 'is_settled_future'] = -1
    dat["future_flag"]= ~dat["train_ind_time"]

    dat["future_paid_cum"] = (dat.log1_paid_cumulative)
    dat.loc[dat['payment_period'] > config['data'].cutoff, 'future_paid_cum'] = 12.3

    dat["L250k"]=0
    dat.loc[dat['claim_size'] > 250000, 'L250k'] = 1

    
    currentdev = dat[dat['payment_period'] == config['data'].cutoff].set_index('claim_no')['development_period'].to_dict()
    dat['curr_dev'] = dat['claim_no'].map(currentdev).fillna(0)

    currentpaid = dat[dat['payment_period'] == config['data'].cutoff].set_index('claim_no')['log1_paid_cumulative'].to_dict()
    dat['curr_paid'] = dat['claim_no'].map(currentpaid).fillna(0)

    currentpmtno = dat[dat['payment_period'] == config['data'].cutoff].set_index('claim_no')['pmt_no'].to_dict()
    dat['curr_pmtno'] = dat['claim_no'].map(currentpmtno).fillna(0)
        
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
    features = config['data'].features
    y_output = config['data'].output_field
   
    # Training data: settled claims within training time period
    trainx = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 1) & (dat.train_settled == 1), 
        features + ["claim_no"]
    ]
    y_train = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 1) & (dat.train_settled == 1)
    ].groupby('claim_no')[y_output].last()
    
    # Test data: unsettled claims not in training set
    testx = dat.loc[
        (dat.test_ind_time == 1) & (dat.train_ind == 0) & (dat.train_settled == 0),
        features + ["claim_no"]
    ]
    y_test = dat.loc[
        (dat.train_ind_time == 1) & (dat.train_ind == 0) & (dat.train_settled == 0)
    ].groupby('claim_no')[y_output].last()
    
    return trainx, y_train, testx, y_test


def create_train_test_datasets_davide(dat: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create training and test datasets from processed data.
    
    Args:
        dat: Processed DataFrame
        config: Experiment configuration
        
    Returns:
        Tuple of (trainx, y_train, testx, y_test)
    """
    features = config['data'].features
    y_output = config['data'].output_field
    
    # Training data: settled claims within training time period

    trainx = dat.loc[
        (dat.payment_period <= config['data'].cutoff), 
        features + ["claim_no"]
    ]
    y_train = dat.loc[
        (dat.payment_period <= config['data'].cutoff)
    ].groupby('claim_no')[y_output].last()
    
    # Test data: unsettled claims not in training set
    testx = dat.loc[
        (dat.payment_period > config['data'].cutoff),
        features + ["claim_no"]
    ]
    y_test = dat.loc[
        (dat.payment_period > config['data'].cutoff)
    ].groupby('claim_no')[y_output].last()
    
    return trainx, y_train, testx, y_test
