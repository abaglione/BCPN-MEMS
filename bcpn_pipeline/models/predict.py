import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import shap
import pickle
import xgboost
from scipy import interp
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, roc_curve, auc, confusion_matrix
from sklearn.pipeline import Pipeline
from tune_sklearn import TuneGridSearchCV
import matplotlib.pyplot as plt

import transform

def get_mean_roc_auc(tprs, aucs, fpr_mean):
    tpr_mean = np.mean(tprs, axis=0)
    tpr_mean[-1] = 1.0
    auc_mean = auc(fpr_mean, tpr_mean)
    auc_std = np.std(aucs)

    return {'tpr_mean': tpr_mean, 'fpr_mean': fpr_mean}, {'auc_mean': auc_mean, 'auc_std': auc_std} 

def get_agg_auc(y_all, y_probas_all):

    y_all = np.concatenate(y_all)
    y_probas_all = np.concatenate(y_probas_all)
    
    # https://stackoverflow.com/questions/57756804/roc-curve-with-leave-one-out-cross-validation-in-sklearn
    fpr, tpr, thresholds = roc_curve(y_all, y_probas_all)
    return {'auc': auc(fpr, tpr)}, tpr, fpr
    
def get_performance_metrics(df, actual='actual', pred='pred'):
    stats = {}

    stats['accuracy'] = accuracy_score(y_true=df[actual], y_pred=df[pred])
    
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true=df[actual], y_pred=df[pred], average='macro'
    )
    
    stats.update({'precision': precision, 'recall': recall, 
                  'f1_score': f1_score, 'support': support
                 })

    return stats

def calc_shap(X_train, X_test, model, method):
    shap_values = None
    
    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    elif method == 'SVM':
        X_train_sampled = shap.sample(X_train, 5)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(X_test)

    return shap_values


def gather_shap(X, method, shap_values, test_indices):
    print('Gathering SHAP stats.')

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    test_indices_all = test_indices[0]
    shap_values_all = np.array(shap_values[0])

    for i in range(1, len(test_indices)):
        test_indices_all = np.concatenate((test_indices_all, test_indices[i]), axis=0)
        
        if method == 'RF' or method == 'SVM': # classifiers with multiple outputs
            shap_values_all = np.concatenate(
                (shap_values_all, np.array(shap_values[i])), axis=1)
        else:
            shap_values_all = np.concatenate((shap_values_all, shap_values[i]), axis=0)

    # Bring back variable names
    X_test = pd.DataFrame(X.iloc[test_indices_all], columns=X.columns)

    return X_test, shap_values_all

# Thank you to Lee Cai, who bootstrapped a similar function in a diff project
# Modifications have been made to suit this project.
def optimize_params(X, y, groups, method, random_state):
    n_jobs = -1
    if method == 'LogisticR':
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1'],  # Use LASSO for feature selection
            'solver': ['liblinear'],
            'max_iter': [3000, 6000, 9000]
        }
        model = LogisticRegression(random_state=random_state)

    elif method == 'RF':
        param_grid = {
            'n_estimators': [50, 100, 250, 500],
            'max_depth': [1],
            'min_samples_leaf': [1, 2, 3]
        }
        model = RandomForestClassifier(oob_score=True, random_state=random_state)

    elif method == 'XGB':
        n_jobs = 3
        param_grid = {
            'n_estimators': [50, 100, 250, 500],
            'max_depth': [1],
            'min_child_weight': [1, 3],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        model = xgboost.XGBClassifier(random_state=random_state)

    elif method == 'SVM':
        n_jobs = None
        param_grid = {
            'C': [1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf'] # Robust to noise - no need to do RFE
        }
        
        model = SVC(probability=True, random_state=random_state)

    print('n_jobs = ' + str(n_jobs))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    tune_search = TuneGridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=cv, scoring='roc_auc',  n_jobs=n_jobs,
                                   verbose=2)

    tune_search.fit(X.values, y.values, groups)
    return tune_search

def predict(fs, n_lags=None, models=None, n_runs=5, 
            optimize=False, importance=False, additional_fields=None):

    common_fields = {'n_lags': n_lags, 'featureset': fs.name, 'optimized': optimize,
                     'target': fs.target_col}
    
    if additional_fields:
        common_fields.update(additional_fields)
    
    # If no custom models are given, run all defaults. Start building a dictionary.
    if not models:
        models = {
            'LogisticR': None, 
            'RF': None, 
            'SVM': None
        }

    for method, clf in models.items():
        tprs_all = [] # Array of true positive rates
        aucs_all = []# Array of AUC scores
        fpr_mean = np.linspace(0, 1, 100)

        shap_values_all = list() 
        test_indices_all = list()

        # Do repeated runs
        for run in range(0, n_runs):
            
            train_res_all = [] # Array of dataframes of true vs pred labels
            test_res_all = [] # Array of dataframes of true vs pred labels
            
            random_state = run

            # Get the correct classifier if it doesn't exist
            if clf is None:
                if method == 'LogisticR':
                    clf = LogisticRegression(solver='liblinear', random_state=random_state)
                elif method == 'RF':
                    clf = RandomForestClassifier(max_depth=1, random_state=random_state)
                elif method == 'SVM':
                    clf = SVC(probability=True, random_state=random_state)

            # Do training and testing
            print('Run %i of %i for %s model.' % (run + 1, n_runs, method))

            # Split into inputs and labels
            X = fs.df[[col for col in fs.df.columns if col != fs.target_col]]
            y = fs.df[fs.target_col]

            # Get list of indices of nominal columns for SMOTE-NC upsampling, used in train_test
            # Safeguard to ensure we're getting the right indices
            nominal_cols = [col for col in X.columns if col in fs.nominal_cols]
            nominal_idx = sorted([X.columns.get_loc(c) for c in nominal_cols])
            
            # Set up outer CV
            ''' Need to be splitting at the subject level
                Thank you, Koesmahargyo et al.! ''' 
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

            # Do prediction task
            for train_index, test_index in cv.split(X=X, y=y, groups=X[fs.id_col]):
                X_train, y_train = X.loc[train_index, :], y[train_index]
                X_test, y_test = X.loc[test_index, :], y[test_index]

                print('Imputing missing data.')
                # Create combined dataframes for imputation only
                df_train = pd.concat(X_train, y_train, axis=1)
                df_test = pd.concat(X_test, y_test, axis=1)

                # Do imputation                
                df_train = transform.impute(df_train, fs.id_col)
                df_test = transform.impute(df_test, fs.id_col)
    
                # Split back into X and y
                X_train, y_train = df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]]
                X_test, y_test = df_train[df_test.columns[:-1]], df_train[df_test.columns[-1]]

                # Perform upsampling to handle class imbalance
                print('Conducting upsampling with SMOTE.')
                smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx)
                cols = X_train.columns
                X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_train_upsampled, columns=cols, dtype=float)
                
                # Save the upsampled groups array
                upsampled_groups = X_train[fs.id_col]
                
                # Drop this column from the Xs - IMPORTANT!
                X_train.drop(columns=[fs.id_col], inplace=True)
                X_test.drop(columns=[fs.id_col], inplace=True)

                # Format y
                y_train = pd.Series(y_train_upsampled)
                y_test = pd.Series(y_test)
                
                ''' Perform Scaling
                    Thank you for your guidance, @Miriam Farber
                    https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
                '''
                print('Performing MinMax scaling.')
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                X_train_scaled = scaler.fit_transform(X_train)
                index = X_train.index
                cols = X_train.columns
                X_train = pd.DataFrame(X_train_scaled, index=index, columns=cols)
                
                X_test_scaled = scaler.fit_transform(X_test)
                index = X_test.index
                cols = X_test.columns
                X_test = pd.DataFrame(X_test_scaled, index=index, columns=cols)

                print('Training and testing.')
                # Replace our default classifier clf with an optimized one
                if optimize:
                    print('Getting optimized classifier using gridsearch.')
                    clf = optimize_params(X=X_train, y=y_train, groups=upsampled_groups, 
                                          method=method, random_state=random_state)
                else:
                    clf.fit(X_train.values, y_train.values)
    
                # Be sure to store the training results so we can check for overfitting later
                y_train_pred = clf.predict(X_train.values)
                y_test_pred = clf.predict(X_test.values)
                y_test_probas = clf.predict_proba(X_test.values)[:, 1]

                # Store TPR and AUC
                # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                fpr, tpr, thresholds = roc_curve(y_test, y_test_probas)
                tprs_all.append(interp(fpr_mean, fpr, tpr))
                tprs_all[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs_all.append(roc_auc)
            
                # Store predicted and actual target values in dataframe
                train_res_all.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
                test_res_all.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

                # Calculate feature importance while we're here, using SHAP
                if importance:
                    print('Calculating feature importance for this fold.')
                    shap_values = calc_shap(X_train=X_train, X_test=X_test,
                                            model=clf, method=method, random_state=random_state)
                    shap_values_all.append(shap_values)
                    test_indices_all.append(test_index)
                
            # Save all relevant stats
            print('Calculating predictive performance for this run.')

            # Get train and test results as separate dictionaries
            train_res = pd.concat(train_res_all, copy=True)
            test_res = pd.concat(test_res_all, copy=True)

            train_perf_metrics = get_performance_metrics(train_res)
            test_perf_metrics = get_performance_metrics(test_res)

            all_res = {
                **{'train_accuracy': train_perf_metrics['accuracy']},
                **{'test_' + str(k): v for k, v in test_perf_metrics.items()}
            }

            common_fields.update({'method': method, 'run': run,
                                  'n_features': X.shape[1], 'n_samples': X.shape[0]})
            all_res.update(common_fields)            
  
            # Results are saved for each run
            pd.DataFrame([all_res]).to_csv('results/pred_results.csv', mode='a', index=False)
 
        # Get and save all the shap values
        if importance:
            print('Gathering feature importance across all runs and folds.')

            ''' Don't forget to drop the groups col and unselected feats.
                Otherwise, we'll have issues with alignment.'''
            X_test, shap_values = gather_shap(
                X=X.drop(columns=[fs.id_col]), method=method, 
                shap_values=shap_values_all, test_indices=test_indices_all
            )

            filename = '%s_%s_%d_lags'.format(fs.name, method, n_lags)
            if optimize:
                filename += '_optimized'
            filename += '.ob'
            
            with open('feature_importance/X_test_' + filename, 'wb') as fp:
                pickle.dump(X_test, fp)

            with open('feature_importance/shap_' + filename, 'wb') as fp:
                pickle.dump(shap_values, fp)
        
        # Calculate and save AUC Metrics
        print('Getting mean ROC curve and AUC mean/std across all runs and folds.')
        test_roc_res, test_auc_res = get_mean_roc_auc(tprs_all, aucs_all, fpr_mean)
        
        common_fields.update({'run': -1}) # Indicates these are aggregated results
        
        test_roc_res.update(common_fields)
        pd.DataFrame().from_dict(test_roc_res).to_csv('results/roc_curves.csv', mode='a', index=False)
        
        test_auc_res.update(common_fields)
        pd.DataFrame([test_auc_res]).to_csv('results/auc_results.csv', mode='a', index=False)
        
def tune_lags(fs):
    
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

        # Tune the tree depth - will help us with gridsearch later on
        for max_depth in range(1, 6):
            print('Using tree with max_depth of %i.' % (max_depth))
            models = {
                'RF': RandomForestClassifier(max_depth=max_depth, random_state=max_depth)
            }
            
            predict(all_feats, n_lags, models=models, 
                    optimize=False, importance=False, n_runs=5, 
                    additional_fields={'max_depth': max_depth})
