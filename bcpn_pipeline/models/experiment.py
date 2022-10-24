import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from statsmodels.regression.mixed_linear_model import MixedLM

from ..consts import OUTPUT_PATH_LAGS, OUTPUT_PATH_PRED, OUTPUT_PATH_LMM
from .predict import predict
from .transform import impute
# from .helpers import to_csv_async


def tune_lags(fs):
    
    output_path = OUTPUT_PATH_LAGS
    
    # Exclude first month (ramp-up period during which time users were getting used to the MEMS caps)
    if fs.horizon == 'study_day':
        exclusion_thresh = 30
    elif fs.horizon == 'study_week':
        exclusion_thresh = 4
    elif fs.horizon == 'study_month':
        exclusion_thresh = 1
    
    fs.df = fs.df[fs.df[fs.horizon] > exclusion_thresh]
    
    # Ensure we don't end up with a tiny feature set!
    if fs.horizon == 'study_month':
        lag_range = range(1, 5)
    else:
        lag_range = range(1, 8)
    
    
    for n_lags in lag_range:
        print('For ' + str(n_lags) + ' lags.')

        #Perform final encoding, scaling, etc
        all_feats = fs.prep_for_modeling(n_lags)
        
        # Also tune the tree depth - will help us with gridsearch later on
        for max_depth in range(1, 6):
            print('Using tree with max_depth of %i.' % (max_depth))
            models = {
                'RF': RandomForestClassifier(max_depth=max_depth, random_state=max_depth)
            }

            predict(fs=all_feats, output_path=output_path,
                    select_feats=False, tune=False, 
                    importance=False, models=models, max_depth=max_depth) # Pass in max_depth so it gets recorded...dont' ask me why I designed it this way.

def predict_from_mems(fs, n_lags, **kwargs):

    output_path = OUTPUT_PATH_PRED


    # Get a set of lagged features that's ready to go!
    fs_lagged = fs.prep_for_modeling(n_lags)

    # Do a non-tuned and a tuned run, for comparison's sake
    predict(fs_lagged, output_path=output_path, select_feats=True, tune=True, importance=True, **kwargs)
    # predict(fs_lagged, output_path=output_path, select_feats=False, tune=False, importance=False, **kwargs) 

def gen_mixed_lm(fs, feats_explanatory, alpha=0.5, random_state=7):

    df = fs.df
    df = df.loc[:, (df!=0).any(0)] # Drop columns with all 0's
    feats_explanatory = [col for col in feats_explanatory if col in df.columns]

    # Do imputation
    imputer = IterativeImputer(random_state=random_state)
    imputer.fit(df)
    df = impute(df, imputer)

    # Add the intercept column
    df['intercept'] = 1

    # Set the random seed prior to fitting and predicting
    np.random.seed(random_state)

    # Fit the model
    print('Fitting linear mixed model...')
    # lmm = MixedLM(endog=df[fs.target].astype(float), exog=df[feats_explanatory].astype(
    #     float), groups=df[fs.id_col], exog_re=df['intercept'])

    lmm = MixedLM(endog=df[fs.target_col].astype(float), 
                  exog=df[feats_explanatory].astype(float), 
                  groups=df[fs.id_col], 
                  exog_re=df['intercept']
                  )

    # Fit the model, penalizing the fixed effects
    fit_res = lmm.fit(method='lbfgs', alpha=alpha, disp=1, max_iter=50)
    
    # Create the predictive results DataFrame, storing important info such as the alpha and feature set name
    pred_res = pd.DataFrame({'explanatory_var': fit_res.params.index, 
                             'coef': fit_res.params,
                             't': fit_res.tvalues,
                             'p': fit_res.pvalues}
                            ).reset_index()

    pred_res.drop(columns=['index'], inplace=True)
    pred_res['feature_set'] = fs.horizon
    pred_res['alpha'] = alpha
    pred_res['target'] = fs.target_col

    # Generate final predictions
    print('Making predictions...')
    pred_df = pd.DataFrame(
        fit_res.predict(
            df[feats_explanatory]
        ), 
        columns=[f'pred_{fs.target_col}'])

    pred_df[fs.target_col] = df[fs.target_col]
    pred_df['diff'] = pred_df['pred_'+ fs.target_col] - pred_df[fs.target_col]
    pred_res['rmse'] = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])

    # Save outputs
    pred_res.to_csv(Path.joinpath(OUTPUT_PATH_LMM, f'lmm.csv'))
    return(pred_res)
