import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
import pickle
import xgboost
from scipy import interp
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, roc_curve, auc, confusion_matrix
from sklearn.pipeline import Pipeline
from tune_sklearn import TuneGridSearchCV
import matplotlib.pyplot as plt


def save_res_auc(res, mean_tpr, mean_fpr):
    filename = res['featureset'] + '_' + str(res['n_lags']) + '_lags.csv'
    
    # Save the auc metrics while we're here                    
    auc_df = pd.DataFrame({'test_mean_tpr': mean_tpr, 'test_mean_fpr': mean_fpr})
    auc_df['method'] = res['method']
    auc_df['optimized'] = res['optimized']
    auc_df.to_csv('results/final_auc_results_' + filename, mode='a', index=False)
                  
    pd.DataFrame([res]).to_csv('results/final_pred_results_' + filename, mode='a', index=False)


def get_performance_metrics(df, tprs=None, aucs=None, mean_fpr=None, actual='actual', pred='pred'):
    stats = {}

    stats['accuracy'] = accuracy_score(y_true=df[actual], y_pred=df[pred])
    
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true=df[actual], y_pred=df[pred], average='macro'
    )
    
    stats.update({'precision': precision, 'recall': recall, 
                  'f1_score': f1_score, 'support': support
                 })
    
    if tprs is not None and aucs is not None and mean_fpr is not None:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        stats.update({'mean_auc': mean_auc, 'std_auc': std_auc})
        
        return stats, mean_tpr

    else:
        return stats

# TODO - make shap work for non-test sets too


def calc_shap(X_train, X_test, model, method):
    shap_values = None
    
    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    else:
        X_train_summary = shap.kmeans(X_train, 10)
        shap_values = shap.KernelExplainer(model.predict, X_train_summary).shap_values(X_test)

    return shap_values


def gather_shap(X, method, shap_values, test_indices):
    print('Gathering SHAP stats...')

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    all_test_indices = test_indices[0]
    all_shap_values = np.array(shap_values[0])

    for i in range(1, len(test_indices)):
        all_test_indices = np.concatenate((all_test_indices, test_indices[i]), axis=0)
        
        if method == 'RF': # Random forest has multiple outputs
            all_shap_values = np.concatenate(
                (all_shap_values, np.array(shap_values[i])), axis=1)
        else:
            all_shap_values = np.concatenate((all_shap_values, shap_values[i]), axis=0)

    # Bring back variable names
    X_test = pd.DataFrame(X.iloc[all_test_indices], columns=X.columns)

    return X_test, all_shap_values

# credit to Lee Cai, who bootstrapped the original function in a diff project
# Some modifications have been made to suit this project.


def optimize_params(X, y, method):
    n_jobs = -1
    if method == 'LogisticR':
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l2'],
            'max_iter': [3000, 6000, 9000]
        }
        model = LogisticRegression(random_state=1008)

    elif method == 'RF':
        param_grid = {
            'n_estimators': [50, 100, 250, 500],
            'max_depth': [2, 5, 10, 25],
        }
        model = RandomForestClassifier(oob_score=True, random_state=1008)

    elif method == 'XGB':
#         n_jobs = None # Known bug with multiprocessing when using XGBoost necessitates this...
        param_grid = {
            'n_estimators': [50, 100, 250, 500],
            'max_depth': [3, 6, 9, 12],
            'min_child_weight': [1, 3],
            'learning_rate': [0.01, 0.1, 0.3, 0.5]
        }
        model = xgboost.XGBClassifier(random_state=1008)

    elif method == 'SVM':

        ''' Kernel MUST be linear if we are going to use tune_sklearn, since we need either
        coefficients or feature importances in order to select the best model. '''
        param_grid = {
            'C': [1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear']
        }
        
        model = SVC(random_state=1008)

    print('n_jobs = ' + str(n_jobs))

    # Get RFE and gridsearch objects
    # n_feats = 1.0 # select all features by default, in small datasets
    step = 5
    if X.shape[1] > 30:
        step = 20

    # Ensure we customize our CV to shuffle, to avoid outlier folds that perform worse
    cv = KFold(n_splits=5, shuffle=True, random_state=2)   
    
    estimator = model
    final_param_grid = param_grid
    
    # For methods without intrinsic feature selection, use RFE
    if method != 'XGB' and method != 'RF':
        print('Using RFE')
        estimator = RFE(model, step=step, verbose=3)
        final_param_grid = {'estimator__' + k: v for k, v in param_grid.items()}
        
#     grid = GridSearchCV(estimator=rfe, param_grid=final_param_grid,
#                         cv=cv, scoring='accuracy', n_jobs=n_jobs, verbose=3)

    tune_search = TuneGridSearchCV(estimator=estimator, param_grid=final_param_grid,
                                   cv=cv, scoring='roc_auc',  n_jobs=n_jobs,
                                   verbose=2)

#     grid.fit(X, y)
    tune_search.fit(X, y)
    
    # This will return the best RFE instance
#     return grid.best_estimator_
    return tune_search.best_estimator_


def train_test(X, y, ids, fs_name, method, n_lags, optimize, importance):

    # Get cross validator
    cv = None
    groups = None
    
    if X.shape[0] > 500:
        cv = KFold(n_splits=10, shuffle=True, random_state=5)        
    else:
        '''For a small dataset, leave one group out (LOGO) will function as our leave one subject out (LOSO) cross validation.
        Participant IDs act as group labels. 
        So, at each iteration, one "group" (i.e. one participant id)'s samples will be dropped.
        Seems convoluded but works well.
        '''
        cv = LeaveOneGroupOut()
        groups=ids
        
    print(cv)
    
    # Get a baseline classifier. May not be used if we are optimizing instead.
    clf = None
    if not optimize:
        if method == 'LogisticR':
            clf = LogisticRegression(random_state=1000)
        elif method == 'RF':
            clf = RandomForestClassifier(max_depth=5, random_state=1000)
        elif method == 'XGB':
            clf = xgboost.XGBClassifier(random_state=1000)
        elif method == 'SVM':
            clf = SVC(random_state=1000)

    # Begin train/test
    print('Training and testing with ' + method + ' model...')
   
    tprs = [] # Array of true positive rates
    aucs = []# Array of AUC scores
    mean_fpr = np.linspace(0, 1, 100)
    
    train_res = [] # Array of dataframes of true vs pred labels
    test_res = [] # Array of dataframes of true vs pred labels
    all_shap_values = list() 
    all_test_test_indices = list()

    for train_test_indices, test_test_indices in cv.split(X=X, y=y, groups=groups):

        X_train, y_train = X.loc[train_test_indices, :], y[train_test_indices]
        X_test, y_test = X.loc[test_test_indices, :], y[test_test_indices]

        if optimize:
            # Get the best RFE instance
            clf = optimize_params(X_train, y_train, method)

            if method == 'XGB':
                print(clf.estimator_.get_xgb_params())
            
        clf.fit(X_train, y_train)

        # Be sure to store the training results so we can ensure we aren't overfitting later
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        y_test_probas_ = clf.predict_proba(X_test)

        # Store TPR and AUC
        # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        fpr, tpr, thresholds = roc_curve(y_test, y_test_probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Store predicted and actual vals in dataframe
        train_res.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

        # Calculate feature importance while we're here, using SHAP
        if importance:
            
            # Handle models which used RFE in a particular way
            if optimize and method != 'RF' and method !='XGB':
                
                # Pass in just the selected features and underlying model (not the clf, which is an RFE instance)
                shap_values = calc_shap(X_train=X_train.loc[:, clf.get_support()], X_test=X_test.loc[:, clf.get_support()], 
                                        model=clf.estimator_, method=method)
            else:

                # Pass in just the model (which IS the clf, in this case)
                shap_values = calc_shap(X_train=X_train, X_test=X_test,
                                        model=clf, method=method)

            all_shap_values.append(shap_values)
            all_test_test_indices.append(test_test_indices)

    if importance:

        # Get and save all the shap values
        X_test, shap_values = gather_shap(
            X=X, method=method, shap_values=all_shap_values, test_indices=all_test_test_indices)

        filename = fs_name + '_' + method + '_' + str(n_lags) + '_lags'
        if optimize:
            filename += '_optimized'
        filename += '.ob'
        
        with open('feature_importance/X_test_' + filename, 'wb') as fp:
            pickle.dump(X_test, fp)

        with open('feature_importance/shap_' + filename, 'wb') as fp:
            pickle.dump(shap_values, fp)

    # Save all relevant stats
    print('Calculating performance metrics...')

    # Get train and test results as separate dictionaries
    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)

    train_res = get_performance_metrics(train_res)
    test_res, mean_tpr = get_performance_metrics(test_res, tprs, aucs, mean_fpr)

    # Create a combined results dictionary
    all_res = {'test_' + str(k): v for k, v in test_res.items()}

    # Add only the accuracy from the training results
    # Just used to ensure we aren't overfitting
    all_res['train_accuracy'] = train_res['accuracy']

    return all_res, mean_tpr, mean_fpr

# Adapted from engagement study code - credit to Lee Cai, who co-authored the original code
def predict(fs, n_lags=None, classifiers=None, optimize=True, importance=True):

    print('For featureset "' + fs.name + '"...')

    # Split into inputs and labels
    X = fs.df[[col for col in fs.df.columns if col != fs.target_col]]
    y = fs.df[fs.target_col]

    # Sanity check - Test with a random model first
    print('Conducting sanity check using random model...')

    res = pd.DataFrame(y).rename(columns={fs.target_col: 'actual'})
    res['pred'] = np.random.randint(0, 1, size=len(res))
    stats = get_performance_metrics(res, actual='actual', pred='pred')

    # Make sure it's terrible :P
    assert stats['accuracy'] < 0.5, 'The random model did too well. Go back and check for errors in your data and labels.'
    print('Sanity check passed.')

    #  ----- Handle class imbalance -----
    print('Conducting upsampling with SMOTE...')
    smote = SMOTE(random_state=50)

    # Preserve columns
    cols = X.columns

    # Upsample using SMOTE
    X, y = smote.fit_resample(X, y)

    # Convert X back into a dataframe and ensure its id col is properly formatted
    X = pd.DataFrame(X, columns=cols, dtype=float)
    X[fs.id_col] = X[fs.id_col].astype(str)

    # Format y
    y = pd.Series(y)

    # Pull out the id column so we can do LOOCV in the next steps
    ids = X[fs.id_col]
    X = X[[col for col in X.columns if col != fs.id_col]]

    # If no subset of classifiers is specified, run them all
    if not classifiers:
        classifiers = ['LogisticR', 'RF', 'XGB', 'SVM']

    all_results = []
    for method in classifiers:

        # Do baseline predictions first (no hyperparameter tuning)
        print('Starting with baseline classifier...')
        res, mean_tpr, mean_fpr = train_test(X=X, y=y, ids=ids, fs_name=fs.name, method=method,
                                  n_lags=n_lags, optimize=False, importance=False)
        
        print(res)
        res.update({'n_lags': n_lags, 'featureset': fs.name, 'n_features': X.shape[1],
                    'n_samples': X.shape[0], 'method': method, 'optimized': False,
                    'target': fs.target_col})
        
        save_res_auc(res, mean_tpr, mean_fpr)
        all_results.append(res)

        if optimize:
            print('Getting optimized classifier...')
            res, mean_tpr, mean_fpr  = train_test(X=X, y=y, ids=ids, fs_name=fs.name, method=method,
                                                  n_lags=n_lags, optimize=optimize, importance=importance)
            
            print(res)
                
            res.update({'n_lags': n_lags, 'featureset': fs.name, 'n_features': X.shape[1],
                        'n_samples': X.shape[0], 'method': method, 'optimized': True,
                        'target': fs.target_col})
            
            save_res_auc(res, mean_tpr, mean_fpr)
            all_results.append(res)
            res = None

    return pd.DataFrame(all_results)
