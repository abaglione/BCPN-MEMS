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

