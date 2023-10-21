import numpy as np
import pandas as pd
from itertools import compress

from ..consts import TARGET_HORIZONS

class Featureset:
    def __init__(self, df, name, id_col, nominal_cols=None, target_col=None, horizon=None, n_lags=None):
        self.df = df
        self.name = name
        self.id_col = id_col
        self.target_col = target_col
        self.horizon = horizon
        self.n_lags = n_lags
        
        '''Used to store list of non-continuous columns (e.g., yes/no columns or
        ones that have been one-hot encoded) '''
        self.nominal_cols = []
        if nominal_cols:
            self.nominal_cols += nominal_cols
        
    def prune_nominals(self):
        print('Pruning the nominal columns.')
        nominal_cols = [col for col in self.nominal_cols if 
                        col in self.df.columns
                        and col != self.id_col
                        and col != self.target_col]

        self.nominal_cols = nominal_cols
        
    def one_hot_encode(self):
        print('Doing one-hot encoding.')

        '''One-hot encode categoricals
           We'll want to add any new columns to our list of nominal columsn - use python magic to make it happen
        '''
        cols_og = self.df.columns
        self.df = pd.get_dummies(self.df, columns=self.df.select_dtypes('category').columns)
        cols_all = self.df.columns
        self.nominal_cols += list(set(cols_all) - set(cols_og))

        # Exclude datetimes /non-numerics
        self.df = self.df.select_dtypes('number') # Assumes target col is numeric
        
        self.prune_nominals()
        
    def get_lagged_featureset(self, n_lags):
        print('Getting lagged features.')
        '''Generate lagged observations for temporal data, for each subject '''
        rows = []

        for unique_id in self.df[self.id_col].unique(): 

            # Filter by subject
            subset = self.df[self.df[self.id_col] == unique_id]

            # Sort by horizon
            subset.sort_values(by=self.horizon, ascending=True)

            # Get features as supervised learning df
            # Temporal features will be lagged by a window of size 3
            agg = series_to_supervised(subset.iloc[:, 1:], time_col = self.horizon, target_col = self.target_col, 
                                       n_in=n_lags, n_out=1) 

            # Be sure to add the unique id column back in, at the very beginning
            agg.insert(0, self.id_col, unique_id)

            # Add to list of dfs to concatenate together
            rows.append(agg)

        # Get all subjects' lagged features together
        res = pd.concat(rows, axis=0)
        
        # Be sure to reset the index!
        res = res.reset_index(drop=True)
        
        # Finally, get a new list of nominal feats that mirrors the lagged structure
        mask = [any(col_og in col for col_og in self.nominal_cols) for col in res.columns]
        nominal_cols = list(compress(list(res.columns), mask))
        self.prune_nominals()

        return Featureset(df=res, name=self.name, nominal_cols=nominal_cols, 
                          id_col=self.id_col, target_col=self.target_col, n_lags=n_lags)
    
    def handle_multicollinearity(self):
        print('Handling multicollinearity...')
        cols = [col for col in self.df.columns if col != self.id_col 
                and col != self.target_col
                and col not in TARGET_HORIZONS]
        
        corr_matrix = self.df[cols].corr()

        # --- Credit to Chris Albon ---
        # https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.85
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        
        if len(to_drop) > 0:
            self.df.drop(columns=to_drop, axis=1, inplace=True)

        self.prune_nominals()

    def prep_for_modeling(self, n_lags=None, reduce_collinearity=False):
        print('Preparing feature set for modeling.')
    
        # One hot encode categoricals
        self.one_hot_encode()
        
        if reduce_collinearity:
            self.handle_multicollinearity()

        # If this is a temporal fs
        if n_lags:
        
            # Get new, lagged featureset
            fs = self.get_lagged_featureset(n_lags)
        else:
            fs = self

        # Should have already been done - this is just a safeguard
        fs.prune_nominals()

        # Ensure target column is last
        if fs.target_col:
            end_col = fs.df.pop(fs.target_col)
            fs.df[fs.target_col] = end_col

        return fs

    def __repr__(self):    
        rep = '\n'.join([
            f'Name: { self.name }',
            f'Number of features: {self.df.shape[1]-1}', # Id column is not a feature 
            f'Number of observations: {self.df.shape[0]}'
        ])
        
        if self.horizon:
            rep = rep + f'\nHorizon: { self.horizon }'
        
        if self.n_lags:
            rep = rep + f'\nNumber of lags: { self.n_lags }'
        
        if self.target_col:
            rep = rep + f'\nTarget: { self.target_col }'
        return rep
        
def series_to_supervised(df, time_col, target_col, n_in=1, n_out=1, dropnan=True):
    """
    Thank you to Jason Brownlee, who created this solution 
    (which doesn't rely on TensorFlow). This is adapted from his guide here:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    
    Frame a time series as a supervised learning dataset.
    Arguments:
        df: A pandas DataFrame object.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
        label_column: Name of target var we want to predict later
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = len(df.columns)
    cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s (t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s (t)' % df.columns[j]) for j in range(n_vars)]
        else:
            names += [('%s (t+%d)' % (df.columns[j], i)) for j in range(n_vars)]
    
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
   
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    # Drop all time columns (e.g. 'study_week')
    agg.drop(columns = [col for col in agg.columns if time_col in col], inplace=True)
    
    ''' For the all other columns (the value we want to predict) retain only the observations 
    before time t'''
    agg.drop(columns = [col for col in agg.columns if '(t-' not in col 
                        and target_col not in col], 
             inplace=True)
    
    # drop the (t) suffix in the target column
    # TODO - this won't work if we are predicting more than one time step
    agg.rename(columns = {target_col+' (t)': target_col}, inplace=True)
    return agg


             
             
