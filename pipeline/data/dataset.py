import pandas as pd
import numpy as np
import itertools

class Dataset:
    def __init__(self, df):
        
        # Set main DataFrame
        self.df = df
        
        # Set feature columns and dtypes for quick access
        self.features = {}
        
    def clean(self, rename_dict=None, drop_dict=None):
        ''' Clean the dataset - E.g., rename columns, eliminate useless columns, etc.
        '''
        
        # Replace all-whitespace cells with NaNs
        self.df.replace(['^\s+$'], np.nan, regex = True, inplace=True)

        # Rename selected cols
        if rename_dict:
            self.df.rename(columns=rename_dict, inplace=True)

        # Drop selected cols
        if drop_dict:
            self.df.drop(columns=drop_dict, inplace=True)

        # Remove all columns that are completely empty
        self.df.dropna(axis=1, how='all', inplace=True)

    def set_dtypes(self, dtypes_dict):
        ''' Set dtypes on feature columns '''
        
        for dtype, cols in dtypes_dict.items():
            if dtype == 'datetime':
                self.df[cols] = self.df[cols].apply(pd.to_datetime, errors='coerce')
#                 for col in cols:
#                     self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            elif dtype == 'numeric':
                self.df[cols] = self.df[cols].apply(pd.to_numeric, errors='coerce')
#                 for col in cols:
#                     self.df[col] = pd.to_numeric(self.df[col])
            else:
#                 for col in cols:
#                     self.df[col] = self.df[col].fillna(-1).astype(dtype, errors='ignore') 
                self.df[cols] = self.df[cols].fillna(-1).astype(dtype, errors='ignore')
    
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


