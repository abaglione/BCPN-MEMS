# imports
import sys
import os
import collections
import itertools
import numpy as np
import pandas as pd
import datetime

from ..consts import DAYS_IN_WEEK
    
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

def get_temporal_feats(df, start_date_col, id_col, time_of_day_props):
    '''
        Extracts common temporal features of interest (day, week, month, time of day, etc)
        
        Args:
            df: A Pandas DataFrame
            
            start_date_col: Name of column that stores a participant's study start date
            
            id_col: Name of column that stores a participants's unique identifier (pid)
            
            time_of_day_bins (optional): User-specified list of integers [-1...n], where n < 24, and each two consecutive integers form a lower and upper bound of time (in hours). Datetimes are compared against the list, and datetimes whose `hour`s fall within one of these lower and upper bound sets are considered to be in the same "time of day".
                
                Example: An array [-1, 5, 23] creates two time divisions, where `datetime`s with hour between 0 (12am) and 5 (5am) are in one "time of day" bin, and `datetime`s with hour between 5 (5:01am) and 23 (11:59pm) are in another "time of day" bin.
                                        
            time_of_day_props (optional):
    
    '''
    df['hour'] = df['datetime'].dt.hour

    if time_of_day_props:
        df['time_of_day'] = pd.cut(df['datetime'].dt.hour, time_of_day_props['bins'], labels=time_of_day_props['labels'])

    df['is_weekday'] = df['datetime'].apply(
        lambda x: 1 if x.day_name() != "Saturday" and x.day_name() != "Sunday" else 0
    )
                                      
    df['study_day'] = (df['date'] - df[start_date_col]).dt.days
    df['day_of_week'] = df['date'].dt.day_name()
    df['study_week'] = np.floor((df['date']- df[start_date_col]).dt.days / DAYS_IN_WEEK)
    
    # Estimate of month
    df['study_month'] = np.floor((df['date']- df[start_date_col]) / np.timedelta64(1, 'M'))
    
    for col in ['time_of_day', 'day_of_week']:
        
        # Explicitly set dtype
        df[col] = df[col].astype('category') 

    return df

def calc_standard_static_metrics(df, cols, col_prefix):
    
    df[col_prefix + 'mean'] = df[cols].mean(axis=1)
    df[col_prefix + 'std'] = df[cols].std(axis=1)
#     df[col_prefix + 'min'] = df[cols].min(axis=1)
#     df[col_prefix + 'max'] = df[cols].max(axis=1)
    
    newcols = [col_prefix + metric for metric in ['mean', 'std', 'min', 'max']]
    return df, newcols

def calc_standard_temporal_metrics(df, groupby_cols, datetime_col):
    # TODO: Move to consts
    SECONDS_IN_HOUR = 3600.0
    
    # res = df.groupby(groupby_cols)[datetime_col].agg({
    #     'event_time_mean': lambda x: np.floor(
    #         x.dt.hour.mean()
    #     ),
    #     'event_time_std': lambda x: np.floor(
    #         x.dt.hour.std()
    #     ),
    #     'between_event_time_mean': lambda x: np.floor(
    #         abs(x.diff().mean().total_seconds() / SECONDS_IN_HOUR)
    #     ),
    #     'between_event_time_std': lambda x: np.floor(
    #         x.diff().std().total_seconds() / SECONDS_IN_HOUR
    #     )
    # }).reset_index()
    res = df.groupby(groupby_cols).agg(
        event_time_mean=(
            datetime_col, 
            lambda x: np.floor(
                x.dt.hour.mean()
            )
        ),
        event_time_std=(
            datetime_col, 
            lambda x: np.floor(
                x.dt.hour.std()
            )
        ),
        between_event_time_mean=(
            datetime_col, 
            lambda x: np.floor(
                abs(x.diff().mean().total_seconds() / SECONDS_IN_HOUR)
            )
        ),
        between_event_time_std= (
            datetime_col, 
            lambda x: np.floor(
                x.diff().std().total_seconds() / SECONDS_IN_HOUR
            )
        )
    ).reset_index().fillna(0)
    return res
    