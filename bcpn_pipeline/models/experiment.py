from ..consts import OUTPUT_PATH_PRIMARY
from .predict import predict
from sklearn.ensemble import RandomForestClassifier

def tune_lags(fs):
    
    output_path = OUTPUT_PATH_PRIMARY + '/tuned_lags/'
    
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
        lag_range = range(1, 17)
    
    
    for n_lags in lag_range:
        print('For ' + str(n_lags) + ' lags.')

        #Perform final encoding, scaling, etc
        all_feats = fs.prep_for_modeling(n_lags)
        
        # Ensure we got a lagged series as expected
#         print(all_feats.df)
#         print(all_feats.nominal_cols)

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

    output_path = OUTPUT_PATH_PRIMARY + '/prediction_task/'

    # Get a set of lagged features that's ready to go!
    fs_lagged = fs.prep_for_modeling(n_lags)

    # Do a non-tuned and a tuned run, for comparison's sake
    predict(fs_lagged, output_path=output_path, select_feats=False, tune=False, importance=False, **kwargs) 
    predict(fs_lagged, output_path=output_path, select_feats=True, tune=True, importance=True, **kwargs)