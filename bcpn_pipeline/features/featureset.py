import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Featureset:
    def __init__(self, df, name, id_col, target_col=None, epoch=None):
        self.df = df
        self.name = name
        self.id_col = id_col
        self.target_col = target_col
        self.epoch = epoch
        
    def create_combined_featureset(self, fs):
        # Assumes they have the same id_col
        df = self.df.merge(fs.df, on=[self.id_col])
        return Featureset(df=df, name=self.name + ' - ' + fs.name, id_col=self.id_col, 
                          target_col=self.target_col,
                          epoch=self.epoch)
        
    def transform(self):
        print('Doing imputation, one-hot encoding, and scaling...')
        
        # Impute numerics and categoricals
        categoricals = self.df.select_dtypes('category')
        for col in categoricals.columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        imputer = IterativeImputer(random_state=5)
        numerics = list(set(list(self.df.select_dtypes('number').columns)) -\
                        set([self.id_col]))
        self.df[numerics] = imputer.fit_transform(self.df[numerics])

        # One-hot encode categoricals
        self.df = pd.get_dummies(self.df, columns=self.df.select_dtypes('category').columns)

        # Exclude datetimes /non-numerics
        self.df = self.df.select_dtypes('number') # Assumes target col is numeric

        # Perform Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        to_scale = [col for col in self.df.columns if col != self.id_col and col != self.target_col]
        self.df[to_scale] = scaler.fit_transform(self.df[to_scale]) 
        
        # Final sanity check to ensure imputation / transformations worked - none should be null!
        assert self.df.isnull().values.any() == False, "Imputation failed! Investigate your dataframe."
        
    def get_lagged_featureset(self, n_lags):
        print('Getting lagged features...')
        '''Generate lagged observations for temporal data, for each subject '''
        rows = []

        for unique_id in self.df[self.id_col].unique(): 

            # Filter by subject
            subset = self.df[self.df[self.id_col] == unique_id]

            # Sort by epoch
            subset.sort_values(by=self.epoch, ascending=True)

            # Get features as supervised learning df
            # Temporal features will be lagged by a window of size 3
            agg = series_to_supervised(subset.iloc[:, 1:], time_col = self.epoch, target_col = self.target_col, 
                                       n_in=n_lags, n_out=1) 

            # Be sure to add the unique id column back in, at the very beginning
            agg.insert(0, self.id_col, unique_id)

            # Add to list of dfs to concatenate together
            rows.append(agg)

        # Get all subjects' lagged features together
        res = pd.concat(rows, axis=0)

        return Featureset(df=res, name=self.name, id_col=self.id_col, target_col=self.target_col)
    
    def handle_multicollinearity(self):
        print('Handling multicollinearity...')
        corr_matrix = self.df.corr()

        # --- Credit to Chris Albon ---
        # https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.85
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        
        if len(to_drop) > 0:
            self.df.drop(columns=to_drop, axis=1, inplace=True)
    
    def prep_for_modeling(self, n_lags=None):
        
        # Transform current featureset
        self.transform()
        
        fs = self
        
        # If this is a temporal fs
        if n_lags:
            # Get new, lagged featureset
            fs = fs.get_lagged_featureset(n_lags)
        
        fs.handle_multicollinearity()
        return fs

            
    def __repr__(self):    
        rep = '\n'.join([
            f'Name: { self.name }',
            f'Number of features: {self.df.shape[1] - 2}', 
            f'Number of observations: {self.df.shape[0]}'
        ])
        
        if self.epoch:
            print(self.epoch)
            rep = rep + f'\nEpoch: { self.epoch }'
        
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
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % df.columns[j]) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (df.columns[j], i)) for j in range(n_vars)]
    
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
    agg.rename(columns = {target_col+'(t)': target_col}, inplace=True)
    
    return agg


             
             