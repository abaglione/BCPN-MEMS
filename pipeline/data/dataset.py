import pandas as pd
import numpy as np
import itertools

class Dataset:
    def __init__(self, df, id_cols):
        
        # Set main DataFrame
        self.df = df
        
        # Set id columns (used for reference when modifying the df)
        self.id_cols = id_cols
        
    def set_dtypes(self, dtypes_dict):
        ''' Set dtypes on feature columns '''
        for dtype, cols in dtypes_dict.items():
            if dtype == 'datetime':
                for col in cols:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Don't try to coerce the columns that fail
            elif dtype == 'numeric': 
                for col in cols:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
            else:
                for col in cols:
                    self.df[col] = self.df[col].astype(dtype, errors='ignore') 
    
    def clean(self, to_rename=None, to_drop=None, one_hots=None):
        ''' Clean the dataset - E.g., rename columns, eliminate useless columns
            set initial dtypes, etc, etc.
        '''

        # Rename specific cols
        if to_rename:
            self.df.rename(columns=to_rename, inplace=True)

        # Drop specific cols
        # Need to add a safeguard so you can't drop ID columns!
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            
        # Replace all-whitespace cells with NaNs
        self.df.replace(['^\s+$'], np.nan, regex = True, inplace=True)

        # Remove all columns that are completely empty
        self.df.dropna(axis=1, how='all', inplace=True)

        # Get number of unique values for each column
        counts = self.df.nunique()

        # Record additional columns to drop (those with only one unique value
        to_del = [i for i,v in enumerate(counts) if v == 1]

        # Drop useless columns
        self.df.drop(self.df.columns[to_del], axis=1, inplace=True)
#         print(self.df.shape)

        ''' Handle the special case of categoricals that were recoded as one-hot vectors and 
        that have cells that are strings '''
        if one_hots:
            for col in one_hots:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: convert_onehots(x))

        print('Cleaning complete.')
    
    def build_df_from_features(self, feat_cols):
        ''' Return a dataframe with only the specified features '''
        return self.df[self.id_cols + list(itertools.chain(*[v for k,v in feat_cols.items()]))]
        
    def __repr__(self):
        return '\n'.join([
            f'Number of features: {self.df.shape[1]}'
        ])

def convert_onehots(x):
    try:
        ''' try casting to int - if the column has mixed strings and numbers, this will fail for both
        string 'nan's and other strings (e.g., 'hello' would fail)'''
        float(x)
        return x
    except:
        # decide if we have a nan or not 
        if x.lower() == 'nan':
            return np.nan
        # otherwise, data is present in this cell - code it as such
        else:
            return 1

