import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute(df, id_col):
    print('Doing imputation.')

    # Impute numerics and categoricals
    categoricals = df.select_dtypes('category')
    for col in categoricals.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    imputer = IterativeImputer(random_state=5)

    # Excluding id_col here is unnecessary, since it will always be present...remove sometime
    numerics = list(set(list(df.select_dtypes('number').columns)) -\
                    set([id_col]))
    df[numerics] = imputer.fit_transform(df[numerics])
    return df
