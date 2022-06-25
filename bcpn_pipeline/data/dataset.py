import pandas as pd
import numpy as np
import itertools

class Dataset:
    def __init__(self, df, id_col, feature_categories=None):
        
        # Set main DataFrame
        self.df = df
        
        # Set id column (used for reference when modifying the df)
        self.id_col = id_col
        
        # Store a dictionary of category, list of column pairs
        self.feature_categories = feature_categories

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
            elif dtype == 'categorical':
                for col in cols:
                    self.df[col] = pd.Categorical(self.df[col], ordered=False)
            else:
                for col in cols:
                    self.df[col] = self.df[col].astype(dtype, errors='ignore') 
    
    def clean(self, to_rename=None, to_drop=None, to_map = None, to_binarize=None, onehots_to_reverse=None):
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

        if to_map: # Numeric to strings     
            ''' Convert category numbers to labels, so we can have
            meaningful column names later on'''
            for col in list(to_map.keys()):
                self.df[col] = pd.to_numeric(self.df[col]).map(to_map[col])

        if to_binarize:
            for col in to_binarize:
                if col in self.df.columns:
                    
                    ''' Handle the special case of columns that were recoded as strings but
                    need to be binary (1 or 0)''' 
                    self.df[col] = self.df[col].apply(lambda x: binarize_col(x))    
                    
        if onehots_to_reverse:
            for prefix in onehots_to_reverse:
                cols = [col for col in self.df.columns if prefix in col]
                self.df[prefix.rstrip('_')] = self.df[cols].apply(
                    pd.to_numeric, errors='coerce'
                ).idxmax(1).apply(
                    # Tried lstrip, but it strips the leading letter for some categories!
                    lambda x: x.replace(prefix, '') if type(x) == type('string') else x # Guards against NaNs
                ) 
                self.df.drop(columns=cols, inplace=True)
                
        print('Cleaning complete.')

def build_df_from_feature_categories(df, feat_categories, id_col):
    # Note - will fail if col(s) not in df
    return df[[id_col] + list(itertools.chain(*[v for k,v in feat_categories.items()]))]

def __repr__(self):

    n_cand_feats = 0 if not self.feature_categories else \
                    len(list(itertools.chain(*[v for v in self.feature_categories.values()])))

    return '\n'.join([
        f'Number of columns: {self.df.shape[1]}',
        f'Number of candidate features: { n_cand_feats }'
    ])

def binarize_col(x):
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

