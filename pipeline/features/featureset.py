import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute(df, id_col, numerics, categoricals=None):
    if categoricals:
        numerics = list(set(numerics) - set(categoricals))
        for col in categoricals:
            df[col].fillna(df[col].mode()[0], inplace=True)

    imputer = IterativeImputer(random_state=5)
    df[numerics] = imputer.fit_transform(df[numerics])
    return df

def prep_for_modeling(df, id_col, target_col, categoricals):
    
     # One-hot encode categoricals
    df = pd.get_dummies(df, columns=categoricals) 
    
    # Exclude datetimes /non-numerics
    df = df.select_dtypes('number') # Assumes target col is numeric
    
    # Perform Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    to_scale = [col for col in df.columns if col != id_col and col != target_col]
    df[to_scale] = scaler.fit_transform(df[to_scale]) 
    
    return df