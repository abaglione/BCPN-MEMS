import pandas as pd
import numpy as np

def impute(df, id_col, imputer):
    print('Imputing missing data.')
     
    # Impute numerics and categoricals
    categoricals = df.select_dtypes('category')
    for col in categoricals.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Excluding id_col here is unnecessary, since it will always be present...remove sometime
    numerics = list(set(list(df.select_dtypes('number').columns)) -\
                    set([id_col]))
    df[numerics] = imputer.fit_transform(df[numerics])
    return df

def upsample(X, y, id_col, upsampler):
    print('Upsampling the minority class.')
    X_upsampled, y_upsampled = upsampler.fit_resample(X, y)
    cols = X.columns
    X = pd.DataFrame(X_upsampled, columns=cols, dtype=float)
    
    # Save the upsampled groups array
    upsampled_groups = X[id_col]

    return X, y_upsampled, upsampled_groups

def scale(X, scaler):
    print('Scaling input features.')
    
    ''' Perform Scaling
        Thank you for your guidance, @Miriam Farber
        https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
    '''
    X_scaled = scaler.fit_transform(X)
    index = X.index
    cols = X.columns
    X = pd.DataFrame(X_scaled, index=index, columns=cols)
    return X