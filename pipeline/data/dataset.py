import pandas as pd
import numpy as np
import itertools

class Dataset:
    def __init__(self, df, id_cols):
        
        # Set main DataFrame
        self.df = df
        
        # Set id columns (used for reference when modifying the df)
        self.id_cols = id_cols
        
        # Set feature columns and dtypes for quick access
        self.features = {}

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
    
    def clean(self, rename_dict=None, drop_dict=None):
        ''' Clean the dataset - E.g., rename columns, eliminate useless columns
            set initial dtypes, etc, etc.
        '''

        # Rename specific cols
        if rename_dict:
            self.df.rename(columns=rename_dict, inplace=True)

        # Drop specific cols
        # Need to add a safeguard so you can't drop ID columns!
        if drop_dict:
            self.df.drop(columns=drop_dict, inplace=True)
            
        # Replace all-whitespace cells with NaNs
        self.df.replace(['^\s+$'], np.nan, regex = True, inplace=True)

        # Remove all columns that are completely empty
        self.df.dropna(axis=1, how='all', inplace=True)
        
        # print(self.df.shape)

        # Get number of unique values for each column
        counts = self.df.nunique()

        # Record these additional columns to drop
        to_del = [i for i,v in enumerate(counts) if v == 1]

        # Drop useless columns
        self.df.drop(self.df.columns[to_del], axis=1, inplace=True)
#         print(self.df.shape)
        
        ''' 
        Set dtypes on remaining columns
        For now, naively assume we only have numerics or datetimes
        Dtype setting will ignore errors for, say, objects we try to cast to numerics
        '''
        dtypes_dict = {
            'numeric': [col for col in self.df.columns if 'date' not in col.lower()],
            'datetime': [col for col in self.df.columns if 'date' in col.lower()]
        }
        
        self.set_dtypes(dtypes_dict)
        
        print('Cleaning complete.')
    
    def update_features(self, features_dict, dtypes_dict=None):
        ''' Add new features and set dtypes on them, if desired''' 
        
        for feature_cat, feature_cols in features_dict.items():
            if feature_cat not in self.features.keys():
                self.features[feature_cat] = feature_cols
            else:
                self.features[feature_cat] = self.features[feature_cat] + feature_cols
                self.features[feature_cat] = list(set(self.features[feature_cat])) # Eliminate duplicates
        
        if dtypes_dict:
            self.set_dtypes(dtypes_dict)
        
    def build_df_from_features(self, id_cols):
        ''' Return a dataframe with only the features defined in self.features '''
        feat_cols = list(itertools.chain(*[v for k,v in self.features.items()]))
        return self.df[id_cols + feat_cols]
        
    def __repr__(self):
        return '\n'.join([
            f'Number of columns: {self.df.shape[1]}', 
            f'Number of candidate features (selected columns): {sum([len(x) for x in self.features.values()])}'
        ])


