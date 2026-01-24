"""
Neural Network Models for Claims Reserving

File: excel_utils.py

This module contains helper functions to log modelling outputs to excel.

The following classes are defined


"""

import pandas as pd
import os
    

def save_df_to_excel(df, df_name=None, filename="log_outputs.xlsx", mode='a'):
    """
    Save a DataFrame to an Excel file with the sheet name set to the DataFrame's name.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    df_name : str, optional
        Name for the sheet. If None, attempts to get the variable name
    filename : str, default "log_outputs.xlsx"
        Name of the Excel file
    mode : str, default 'a'
        'a' to append to existing file, 'w' to overwrite
    """
    
    # If df_name is not provided, use a default name
    if df_name is None:
        df_name = "Sheet1"
    
    # Clean sheet name (Excel has restrictions on sheet names)
    df_name = str(df_name)[:31]  # Max 31 characters
    invalid_chars = ['/', '\\', '?', '*', '[', ']', ':']
    for char in invalid_chars:
        df_name = df_name.replace(char, '_')
    
    # Handle existing file
    if mode == 'a' and os.path.exists(filename):
        # Append to existing file
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=df_name, index=False)
    else:
        # Create new file or overwrite
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=df_name, index=False)
    
    print(f"Saved DataFrame to '{filename}' as sheet '{df_name}'")