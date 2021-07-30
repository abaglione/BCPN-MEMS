# imports
import sys
import os
import collections
import itertools
import numpy as np
import pandas as pd
import datetime
    
def reset_index(df):
    """
    An excellent workaround for resetting categorical indices
    
    Credit to @akielbowicz
    https://github.com/pandas-dev/pandas/issues/19136#issuecomment-380908428
    
    Args:
        df (DataFrame): A pandas DataFrame object
    
    Returns: 
        A pandas DataFrame with index as columns
    """
    index_df = df.index.to_frame(index=False)
    df = df.reset_index(drop=True)
    # In merge is important the order in which you pass the dataframes
    # if the index contains a Categorical. 
    # pd.merge(df, index_df, left_index=True, right_index=True) does not work
    return pd.merge(index_df, df, left_index=True, right_index=True)    

def match_value(df, col1, x, col2):
    """ 
    Match value x in df to a value in col1 in a given row, then return the first matched value in col2 of that same row.
    (pulled from StackOverflow)
    
    Useful for pulling info for a given participant id
    
    Args:
        df (DataFrame): A pandas DataFrame object
        col1 (str): The column to search
        x: The value to match in col1
        col2 (str): The column from which we want to return a value
        
    Returns:
        The value from col2, from the first matched item
    
    """
    filtered = df[df[col1] == x]
    if filtered.empty:
        return np.nan
    else:
        return filtered[col2].values[0]

# Get a combined datetime column
def get_datetime_col(x):
    date = None
    try:
        date = datetime.datetime.strptime(x['date'] + ' ' + x['time'], '%m/%d/%Y %H:%M:%S') 
    except:
        date = np.nan

    return date


def mean_days_between_dates(x):
    dates = [date for date in x if type(date)!= pd._libs.tslibs.nattype.NaTType]
    dates = sorted(dates)
    if len(dates) > 1:
        mean_days_betw = sum((b - a).days for a, b in zip(dates, dates[1:])) / (len(dates) - 1)
        return mean_days_betw
    else:
        return np.NaN

def get_epochs(df, start_date_col, pid_col, time_of_day_bins=None, time_of_day_labels=None):
    '''
        Extracts common epochs of interest (day, week, month, time of day, etc)
        
        Args:
            df: A Pandas DataFrame
            
            start_date_col: Name of column that stores a participant's study start date
            
            pid_col: Name of column that stores a participants's unique identifier (pid)
            
            time_of_day_bins (optional): User-specified list of integers [-1...n], where n < 24, and each two consecutive integers form a lower and upper bound of time (in hours). Datetimes are compared against the list, and datetimes whose `hour`s fall within one of these lower and upper bound sets are considered to be in the same "time of day".
                
                Example: An array [-1, 5, 23] creates two time divisions, where `datetime`s with hour between 0 (12am) and 5 (5am) are in one "time of day" bin, and `datetime`s with hour between 5 (5:01am) and 23 (11:59pm) are in another "time of day" bin.
                                        
            time_of_day_labels (optional): User-specified labels for each time division created in the time_of_day_bins list
    
    '''
    df['hour'] = df['datetime'].dt.hour

    if time_of_day_bins:
        df['time_of_day'] = pd.cut(df['datetime'].dt.hour, time_of_day_bins, labels=time_of_day_labels)
    
    df['weekday'] = df['datetime'].dt.day_name() # e.g., "Monday"
    
    df['study_day'] = (df['datetime'] - df[start_date_col]).dt.days
    df['study_week'] = np.floor((df['datetime']- df[start_date_col]).dt.days / 7.0)
    
    # https://stackoverflow.com/a/151211
    df['study_month'] = 12 * (df['datetime'].dt.year - df[start_date_col].dt.year) + (df['datetime'].dt.month - df[start_date_col].dt.month)
    
    return df

def calc_standard_static_metrics(df, cols, col_prefix):
    
    df[col_prefix + 'mean'] = df[cols].mean(axis=1)
    df[col_prefix + 'std'] = df[cols].std(axis=1)
    df[col_prefix + 'min'] = df[cols].min(axis=1)
    df[col_prefix + 'max'] = df[cols].max(axis=1)
    
    newcols = [col_prefix + metric for metric in ['mean', 'std', 'min', 'max']]
    return df, newcols

def calc_standard_temporal_metrics(df, groupby_cols, datetime_col):
    # TODO: Move to consts
    SECONDS_IN_HOUR = 3600.0
    
    res = df.groupby(groupby_cols)[datetime_col].agg({
        'event_time_mean': lambda x: np.floor(
            x.dt.hour.mean()
        ),
        'event_time_std': lambda x: np.floor(
            x.dt.hour.std()
        ),
        'event_time_min': lambda x: np.floor(
            x.dt.hour.min()
        ),
        'event_time_max': lambda x: np.floor(
            x.dt.hour.max()
        ),    
        'between_event_time_mean': lambda x: np.floor(
            abs(x.diff().mean().total_seconds() / SECONDS_IN_HOUR)
        ),
        'between_event_time_std': lambda x: np.floor(
            x.diff().mean().total_seconds() / SECONDS_IN_HOUR
        ),
        'between_event_time_min': lambda x: np.floor(
            x.diff().min().total_seconds() / SECONDS_IN_HOUR
        ),
        'between_event_time_max': lambda x: np.floor(
            x.diff().max().total_seconds() / SECONDS_IN_HOUR
        )
    }).reset_index()
    return res
    