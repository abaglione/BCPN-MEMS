import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Featureset(df, name, id_col, target_col=None):
    self.df = df
    self.name = name
    self.id_col = id_col
    self.target_col = target_col

    def prep_for_modeling(self, categoricals):

         # One-hot encode categoricals
        self.df = pd.get_dummies(self.df, columns=categoricals) 

        # Exclude datetimes /non-numerics
        self.df = self.df.select_dtypes('number') # Assumes target col is numeric

        # Perform Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        to_scale = [col for col in self.df.columns if col != self.id_col and col != self.target_col]
        self.df[to_scale] = scaler.fit_transform(self.df[to_scale]) 

        import pandas as pd
        
    def get_lagged_df(self, epoch, n_lags):
        '''Generate lagged observations for temporal data, for each subject '''

        rows = []

        for unique_id in self.df[self.id_col].unique(): 

            # Filter by subject
            subset = self.df[self.df[self.id_col] == unique_id]

            # Sort by epoch
            subset.sort_values(by=epoch, ascending=True)

            # Get features as supervised learning df
            # Temporal features will be lagged by a window of size 3
            agg = series_to_supervised(subset.iloc[:, 1:], time_col = epoch, target_col = self.target_col, 
                                       n_in=n_lags, n_out=1) 

            # Be sure to add the unique id column back in, at the very beginning
            agg.insert(0, self.id_col, unique_id)

            # Add to list of dfs to concatenate together
            rows.append(agg)

        # Get all subjects' lagged features together
        res = pd.concat(rows, axis=0)

        return res

def impute(df, id_col, numerics, categoricals=None):
    if categoricals:
        numerics = list(set(numerics) - set(categoricals))
        for col in categoricals:
            df[col].fillna(df[col].mode()[0], inplace=True)

    imputer = IterativeImputer(random_state=5)
    df[numerics] = imputer.fit_transform(df[numerics])
    return df
        
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
    
    ''' For the target column (the value we want to predict) retain only the observations at time t onwards, 
    since we don't want these as input features '''
    agg.drop(columns = [col for col in agg.columns if target_col in col][:n_in], inplace=True)

    return agg


             
             