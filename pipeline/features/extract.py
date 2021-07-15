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

def gen_categories(df, col_properties):
    ''' 
        Create column(s) that categorize observations based on cutoff values for existing column(s)
        Example:
        
    '''
    for col, props in col_properties.items():
        df[col + '_cat'] = pd.cut(df[col], props['bins'], labels=props['labels'])
    
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

# --------- LEGACY FUNCTION --------- 
def weekly_epoch_breakdown(df, metric, merge_cols):
    
    """
    Create long-form (?) version of weekly feature vectors with one column per epoch-measure
    - Example: produces column `evening_duration`
    - Useful for graphing later on
        
    Args:
        df (DataFrame): A pandas DataFrame object
        metric (str): The name of the engagement metric (e.g. 'duration' or 'frequency')
        merge_cols ([str]): A list of common columns used as identity columns for merging 
            in future operations (e.g. the participant id column, or the week of study column); 
            these should never be renamed!
    Returns:
        df (Dataframe): The pandas dataframe, in long-form (?)
    
    """
    
    # Expand using reset_index workaround above
    df = df.unstack()
    df = reset_index(df)
   
     # Custom reordering of epochs to be consistent with times of the day
    df = df[merge_cols + EPOCHS['labels'][1:] + [EPOCHS['labels'][0]]]
    
    # Append metric label to each epoch column 
    #  (e.g. "morning_frequency")
    df.rename(columns=lambda x: x.lower().replace(" ", ""), inplace=True)
    df.rename(columns=lambda x: x +'_' + metric if x not in merge_cols else x, inplace=True)
    
    # Replace missing data
    df.replace(np.nan,-1, inplace=True)
    return df

# TODO: Add options for specifying which metrics to calculate? Which values to fill missing values with?
def calc_duration_has_epoch(dataset, groupbycols):
    """
    Calculate measurements specific to duration (sum, mean, std, etc) for a dataframe
    that HAS been divided into epochs. 
        
    Args:
        dataset (DataFrame): A pandas DataFrame object; contains one entry per participant
            per timestamped app launch
        groupbycols ([str]): The names of the columns by which to group, for each calculation
            (e.g. the participant id column)
    
    Returns:
        df (Dataframe): A pandas dataframe with one column per duration-related measurement
    """
    
    merge_cols = groupbycols[:-1]
    
    # Get dataframe `df` with target measure
    df = dataset.groupby(groupbycols)['duration'].sum()
    
    # Get dataframe with epoch-measure columns from existing dataframe, and assign to new (resulting) dataframe `res`
    res = weekly_epoch_breakdown(df, 'duration', merge_cols) 
    
    # No need to do any merging this first time...
    
    # Repeat process above for next measure
    df = dataset.groupby(groupbycols)['duration'].mean()
    
    # This time, get the dataframe with epoch-measure columns by modifying the existing dataframe `df`; do not assign result to new dataframe, but rather to old `df`
    df = weekly_epoch_breakdown(df, 'duration_mean', merge_cols)

    # Set resulting dataframe as a merged result of updated dataframe `df` and previous resulting dataframe `res`
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )
    
    # And on we go for as many measures as we like...

    df = dataset.groupby(groupbycols)['duration'].std()
    df = weekly_epoch_breakdown(df, 'duration_std', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = dataset.groupby(groupbycols)['duration'].min()
    df = weekly_epoch_breakdown(df, 'duration_min', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = dataset.groupby(groupbycols)['duration'].max()
    df = weekly_epoch_breakdown(df, 'duration_max', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )
    
    df = dataset.groupby(groupbycols)['date'].apply(
            lambda x: x.diff().mean().total_seconds()
    )
   
    df = df.round(0)
    df = weekly_epoch_breakdown(df, 'betweenlaunch_duration_mean', merge_cols) 
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = dataset.sort_values('date').groupby(groupbycols)['date'].apply(
            lambda x: x.diff().std().total_seconds()
    )
    df = weekly_epoch_breakdown(df, 'betweenlaunch_duration_std', merge_cols)   
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )
    
# Old strategy for indicating missing vars
#     res.replace(0.,-1, inplace=True)

    # Set missing values to be a very high value (in this case, the number of seconds in a 24-hour span)
    # Helps us find missing alues later during imputation
    res.replace(np.nan,3600*24, inplace=True)
    
    return res

'''
TODO: 
- Modify to allow for any time span?
- Optional param: list of names of apps to iterate over

'''
def construct_feature_vectors(dataset, survey_df, timediv):
    vectors = None
    timediv_col = None
    
    # Get a list of apps over which to iterate
    apps = list(dataset['package'].unique())
    
    if timediv == 'daily':
        # Let our daily individual aggregate features be our starting dataframe
        vectors = pd.read_csv('features/app_dly_ind_agg.csv')  
        
        # Specify the column for our time division (e.g., daily or weekly)
        timediv_col = 'dayofstudy'
        
        # Specify the files containing applevel features
        # NOTE: Had problem last time with "for each" with only one list entry...hmm
        applevel_featurefiles = ['features/app_dly_ind_applevel.csv']
        
    elif timediv == 'wkly':
        # Let our weekly individual aggregate features be our starting dataframe
        vectors = pd.read_csv('features/wkly_agg.csv')
        timediv_col = 'weekofstudy'
        applevel_featurefiles = ['features/wkly_applevel.csv', 'features/wkly_epoch_applevel.csv']
        
    # for pid in list(vectors['pid'].unique()):
#     for week in range(2, 8):
#         series = pd.Series([pid, week], index = vectors.columns)
#         vectors = vectors.append(series, ignore_index=True)

    # Implement a sorting scheme, to more easily visualize participants' progression
    #   through the study
    vectors = vectors.sort_values(by=['pid', timediv_col])
    vectors.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Obtain the day and week each app launch occured
    df = dataset[['pid', 'dayofstudy','weekofstudy']]

    # TODO: Fix this merge - introduces duplicates :( which need to be manually removed in Excel
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")
    
    # Remove any entries that fall outside the study window
    vectors = vectors[vectors['dayofstudy'] > 0]
    
    # Create new column for each measure, for each app
    # For instance, we want to add a column for the "Worryknot" app's total frequency of use
    #   We also want to add a column for the "Worryknot" app's frequency of use for each epoch
    # Note that we also include the file with epoch- and app-specific metrics.
    #   For instance, we want to add a column for the "worryknot" app's total frequency for the morning
    to_merge = []
    for f in applevel_featurefiles:
        df = pd.read_csv(f)
        df.drop(columns=['Unnamed: 0'], inplace=True)

        for app in apps:
            filtered = df[df['package'] == app].drop(columns=['package'])
            metric_cols = [col for col in filtered.columns if col not in ['pid', timediv_col]]
            for col in metric_cols:
                filtered.rename(index=str, columns={col: col + '_' +  app}, inplace=True)
            to_merge.append(filtered)

    for df_to_merge in to_merge:
        vectors = pd.merge(vectors, df_to_merge, on=['pid', timediv_col], how="left")
       
    # Calculate total number of apps used during time division (e.g., per week), per user
    df = dataset.groupby(['pid', timediv_col])['package'].nunique().reset_index(name='num_apps_used')
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")

    # Find the name(s) of the most used app(s) each day, per user
    df = dataset.groupby(['pid', timediv_col])['package'].agg(pd.Series.mode).reset_index(name='most_used_app')

    # Create one column per most-used app (descending order)
    df = pd.concat([df, df['most_used_app'].apply(pd.Series).add_prefix('most_used_app_')], axis = 1)
    
    # No longer need the column we started with
    df.drop(columns=['most_used_app'], inplace=True)
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")
    
    # Finally, let's add survey features
    # For daily features, daily survey features such as mood scores will be associated with the single weekly feature
    df = survey_df[['pid', 'weekofstudy', 'cope_alcohol_tob', 'physical_pain', 'connected', 'receive_support', 'support_others', 'active', 'healthy_food']]
    
    vectors = pd.merge(vectors, df, on=['pid', 'weekofstudy'], how="left")
    vectors.replace(np.nan,-1, inplace=True) 
    
    return vectors
        
    