import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
import pickle
import xgboost
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def get_performance_metrics(df, actual='actual', pred='pred', labels=[0, 1]):
    stats = {}

    df['accuracy'] = accuracy_score(y_true=df[actual], y_pred=df[pred])
    stats['accuracy'] = df['accuracy'].sum()/df.shape[0]

    df['f1'] = f1_score(y_true=df[actual], y_pred=df[pred])
    stats['f1'] = df['f1'].sum()/df.shape[0]

    df['precision'] = precision_score(y_true=df[actual], y_pred=df[pred])
    stats['precision'] = df['precision'].sum()/df.shape[0]

    df['recall'] = recall_score(y_true=df[actual], y_pred=df[pred])
    stats['recall'] = df['recall'].sum()/df.shape[0]

    tn, fp, fn, tp = confusion_matrix(
        df[actual], df[pred], labels=labels).ravel()
    stats.update({'tpr': tp / (tp + fn), 'fpr': fp / (fp + tn),
                  'tnr': tn / (tn + fp), 'fnr': fn / (tp + fn)
                  })

    return stats

# TODO - make shap work for non-test sets too


def calc_shap(X, model, method):
    if method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X)
    else:
        shap_values = shap.KernelExplainer(model.predict, X).shap_values(X)

    return shap_values


def gather_shap(X, method, shap_values, indices):
    print('Gathering SHAP stats...')

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    all_indices = indices[0]
    all_values = np.array(shap_values[0])
#     print(indices)
#     print(shap_values)

    for i in range(1, len(indices)):
        all_indices = np.concatenate((all_indices, indices[i]), axis=0)
        if method == 'RF':
            all_values = np.concatenate(
                (all_values, np.array(shap_values[i])), axis=1)
        else:
            all_values = np.concatenate((all_values, shap_values[i]), axis=0)


#     print(all_indices)
#     print(type(all_indices))
#     print(X)
    # Bring back variable names
    X_test = pd.DataFrame(X.iloc[all_indices], columns=X.columns)

    return X_test, all_values

# credit to Lee Cai, who bootstrapped the original function in a diff project
# Some modifications have been made to suit this project.


def optimize_params(X, y, method):
    model = None

    if method == 'LogisticR':
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l2'],
            'max_iter': [3000]
        }
        model = LogisticRegression(random_state=1008)

    elif method == 'RF':
        param_grid = {
            'n_estimators': [50, 100, 250],
            'max_depth': [2, 5, 10, 25],
        }
        model = RandomForestClassifier(oob_score=True, random_state=1008)

    elif method == 'XGB':
        param_grid = {
            'n_estimators': [50, 100, 250],
            'max_depth': [3, 5],
            'min_child_weight': [1, 3],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        model = xgboost.XGBClassifier(random_state=1008)

    elif method == 'SVM':
        param_grid = {
            'C': [1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        model = SVC(random_state=1008)

    final_param_grid = {'estimator__' + k: v for k, v in param_grid.items()}

    # Get RFE and gridsearch objects
    # n_feats = 1.0 # select all features by default, in small datasets
    step = 5
    if X.shape[1] > 30:
        step = 20

    rfe = RFE(model, step=step)
    grid = GridSearchCV(estimator=rfe, param_grid=final_param_grid,
                        cv=5, scoring='accuracy')
    grid.fit(X, y)

    # This will return the best RFE instance
    return grid.best_estimator_


def train_test(X, y, ids, method, n_lags, optimize, importance):
    train_res = []
    test_res = []
    all_shap_values = list()
    all_test_indices = list()

    '''Leave one group out (LOGO) will function as our leave one subject out (LOSO) cross validation.
       Participant IDs act as group labels. 
       So, at each iteration, one "group" (i.e. one participant id)'s samples will be dropped.
       Seems convoluded but works well.
       '''
    logo = LeaveOneGroupOut()

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

    print('Training and testing with ' + method + ' model...')
    for train_indices, test_indices in logo.split(X, y, ids):

        X_train, y_train = X.loc[train_indices, :], y[train_indices]
        X_test, y_test = X.loc[test_indices, :], y[test_indices]

        if optimize:
            # Get the best RFE instance
            clf = optimize_params(X_train, y_train, method)
            
            # Retain only the most important features
            X_train = X_train.loc[:, clf.get_support()]
            X_test = X_test.loc[:, clf.get_support()]

        clf.fit(X_train, y_train)

        # Be sure to store the training results so we can ensure we aren't overfitting later
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Store results
        train_res.append(pd.DataFrame(
            {'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

        # Calculate feature importance while we're here, using SHAP
        if importance and method != 'LogisticR':

            if optimize:
                # Pass in just the model (not the clf, which is an RFE instance)
                shap_values = calc_shap(
                    X=X_test, model=clf.estimator_, method=method)
            else:

                # Pass in just the model (which IS the clf, in this case)
                shap_values = calc_shap(X=X_test, model=clf, method=method)

            all_shap_values.append(shap_values)
            all_test_indices.append(test_indices)

    if importance and method != 'LogisticR':

        # Get and save all the shap values
        X_test, shap_values = gather_shap(
            X=X, method=method, shap_values=all_shap_values, indices=all_test_indices)

        filename = 'feature_importance/X_test_' + \
            method + '_' + str(n_lags) + '_lags'
        if optimize:
            filename += '_optimized'
        filename += '.ob'
        with open(filename, 'wb') as fp:
            pickle.dump(X_test, fp)

        filename = 'feature_importance/shap_' + \
            method + '_' + str(n_lags) + '_lags'
        if optimize:
            filename += '_optimized'
        filename += '.ob'
        with open(filename, 'wb') as fp:
            pickle.dump(shap_values, fp)

    # Save all relevant stats
    print('Calculating performance metrics...')

    # Get train and test results as separate dictionaries
    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)

    train_res = get_performance_metrics(train_res)
    test_res = get_performance_metrics(test_res)

    # Create a combined results dictionary
    all_res = {'test_' + str(k): v for k, v in test_res.items()}

    # Add only the accuracy from the training results
    # Just used to ensure we aren't overfitting
    all_res['train_accuracy'] = train_res['accuracy']

    # Add remaining info
    return all_res

# Adapted from engagement study code - credit to Lee Cai, who co-authored the original code
def predict(fs, n_lags=None, classifiers=None, optimize=True, importance=True):
    all_results = []

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

    for method in classifiers:

        # Do baseline predictions first (no hyperparameter tuning)

        # TODO...Anna...you're doing gridsearch on all your data :| fix this so you only do it on your training set!
        # Will need to move the optimization calls into train_test
        print('Starting with baseline classifier...')
        res = train_test(X=X, y=y, ids=ids, method=method,
                         n_lags=n_lags, optimize=False, importance=importance)
        
        res.update({'n_lags': n_lags, 'featureset': fs.name, 'n_features': X.shape[1],
                    'n_samples': X.shape[0], 'method': method, 'optimized': False,
                    'target': fs.target_col})
        
        all_results.append(res)

        if optimize:

            res = train_test(X=X, y=y, ids=ids, method=method,
                             n_lags=n_lags, optimize=True, importance=importance)
            
            res.update({'n_lags': n_lags, 'featureset': fs.name, 'n_features': X.shape[1],
                        'n_samples': X.shape[0], 'method': method, 'optimized': True,
                        'target': fs.target_col})
            
            all_results.append(res)

    return pd.DataFrame(all_results)
