# src/cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fill_missing_median(df, columns):
    """
    Fills missing values in specified columns of a DataFrame with their respective medians.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list of str): A list of column names to fill.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
    """
    df_copy = df.copy()
    for col in columns:
        # Ensure the column is numeric before calculating the median
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            median_val = df_copy[col].median()
            df_copy[col].fillna(median_val, inplace=True)
    return df_copy

def drop_missing(df, threshold):
    """
    Drops rows from a DataFrame that have more missing values than a specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The minimum percentage of non-missing data a row must have (0.0 to 1.0).
                           For example, 0.5 means a row must be at least 50% full to be kept.

    Returns:
        pd.DataFrame: The DataFrame with sparse rows removed.
    """
    df_copy = df.copy()
    # Calculate the minimum number of non-NA values required for each row
    min_non_na = int(threshold * df_copy.shape[1])
    return df_copy.dropna(thresh=min_non_na)

def normalize_data(df, columns):
    """
    Applies Min-Max normalization to specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list of str): A list of column names to normalize.

    Returns:
        pd.DataFrame: The DataFrame with specified columns normalized.
    """
    df_copy = df.copy()
    scaler = MinMaxScaler()
    
    # The scaler expects a 2D array, so we pass the columns as a DataFrame slice
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy