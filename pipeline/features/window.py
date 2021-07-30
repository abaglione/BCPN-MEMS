import numpy as np
import pandas as pd

class WindowGenerator():
    def __init__(self, df, n_in=1, n_out=1):
        '''Arguments: 
            df: A pandas DataFrame object.
            n_in: Number of time steps to be used as input (X) (e.g., 15 weeks).
            n_out: Number of time steps into the future for which we want to generate observations
        '''     
        self.data = df
        self.n_in = n_in
        self.n_out=n_out
        self.featureset, self.train, self.test = None

    def make_featureset(self, n_test, dropnan=True):
        '''
        Thank you to Jason Brownlee! This function is an adaptation of separate functions 
        from the following guides: 
        https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
        https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        
        Frame a time series as a supervised learning dataset.
        Arguments:
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            train and test sets for supervised learning
        '''
        n_vars = len(self.df.columns)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(self.df.shift(i))
            names += [('%s(t-%d)' % (self.df.columns[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(self.df.shift(-i))
            if i == 0:
                names += [('%s(t)' % self.df.columns[j]) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (self.df.columns[j], i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
       
        self.featureset = agg # Store featureset in a new var

        # split into train and test sets and return these
        train, test = self.featureset.values[0:-n_test], self.featureset.values[-n_test:]
        return train, test