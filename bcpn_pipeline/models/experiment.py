from ..consts import OUTPUT_PATH_PRIMARY
from .predict import predict
from .transform import impute
# from .helpers import to_csv_async

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM

def tune_lags(fs):
    
    output_path = OUTPUT_PATH_PRIMARY + 'tuned_lags/'
    
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

    output_path = OUTPUT_PATH_PRIMARY + 'prediction_task/'


    # Get a set of lagged features that's ready to go!
    fs_lagged = fs.prep_for_modeling(n_lags)

    # Do a non-tuned and a tuned run, for comparison's sake
    predict(fs_lagged, output_path=output_path, select_feats=True, tune=True, importance=True, **kwargs)
    predict(fs_lagged, output_path=output_path, select_feats=False, tune=False, importance=False, **kwargs) 
    

# def gen_mixed_lm(fs, explanatory_feats, alpha=0.5, random_state=7):
#     output_path = OUTPUT_PATH_PRIMARY + '/lmm_task/'
    
#     df = fs.df

#     # Do imputation
#     imputer = IterativeImputer(random_state=random_state)
#     imputer.fit(df)
#     df = impute(df, imputer)

#     # Add the intercept column
#     df['intercept'] = 1
#     np.random.seed(random_state)

#     # Fit the model
#     print('Fitting linear mixed model...')
#     lmm = MixedLM(endog=df[fs.target].astype(float), exog=df[explanatory_feats].astype(
#         float), groups=df[fs.id_col], exog_re=df['intercept'])

#     fit_res = lmm.fit_regularized(method='l1', alpha=alpha, disp=1)
    
#     # Create the predictive results DataFrame, storing important info such as the alpha and feature set name
#     pred_res = pd.DataFrame({'explanatory_var': fit_res.params.index, 'coef': fit_res.params,
#                         't_value': fit_res.t_values, 'p_value': fit_res.p_values}).fit_reset_index()
#     pred_res.drop(columns=['index'], inplace=True)
#     pred_res['feature_set'] = fs.horizon
#     pred_res['alpha'] = alpha
#     pred_res['target'] = fs.target

#     # Generate final predictions
#     print('Making predictions...')
#     pred_df = pd.DataFrame(fit_res.predict(
#         df[explanatory_feats].astype(float)), columns=['pred_'+ fs.target])
#     pred_df[fs.target] = df[fs.target]
#     pred_df['diff'] = pred_df['pred_'+ fs.target] - pred_df[fs.target]

#     pred_res['rmse'] = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])

#     # No need to wrap this in an event loop, since just one call
#     to_csv_async(pred_res, fp=output_path + 'pred.csv')

#     return(pred_res)