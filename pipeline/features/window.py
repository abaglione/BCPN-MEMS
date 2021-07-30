import numpy as np
import pandas as pd

def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
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

    return agg