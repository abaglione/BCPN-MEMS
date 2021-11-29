import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import shap
import pickle
import xgboost
from scipy import interp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from tune_sklearn import TuneGridSearchCV
import matplotlib.pyplot as plt

from . import transform
from . import metrics

# Thank you to Lee Cai, who bootstrapped a similar function in a diff project
# Modifications have been made to suit this project.
def tune_hyperparams_params(X, y, groups, method, random_state):
    print('Getting tune_hyperparamsd classifier using gridsearch.')
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
                                   cv=cv, scoring='recall',  n_jobs=n_jobs,
                                   verbose=2)

    tune_search.fit(X.values, y.values, groups)
    return tune_search.best_estimator_

def train_test(X, y, id_col, clf, random_state, nominal_idx, 
               method, select_feats, tune_hyperparams, importance, fpr_mean):
    tprs = [] # Array of true positive rates
    aucs = []# Array of AUC scores

    shap_values = list() 
    test_indices = list()

    train_res = [] # Array of dataframes of true vs pred labels
    test_res = [] # Array of dataframes of true vs pred labels

    # Set up outer CV
    ''' Need to be splitting at the subject level
        Thank you, Koesmahargyo et al.! ''' 
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Do prediction task
    for train_index, test_index in cv.split(X=X, y=y, groups=X[id_col]):
        X_train, y_train = X.loc[train_index, :], y[train_index]
        X_test, y_test = X.loc[test_index, :], y[test_index]

        # Do imputation
        imputer = IterativeImputer(random_state=5)
        X_train = transform.impute(X_train, id_col, imputer)
        X_test = transform.impute(X_test, id_col, imputer)

        try:
            # Perform upsampling to handle class imbalance
            smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx)
            X_train, y_train, upsampled_groups = transform.upsample(X, y, id_col, smote)
        
        except ValueError:       
            # Set n_neighbors = n_samples
            # Not great if we have a really small sample size. Hmm.
            k_neighbors = (y_train == 1).sum() - 1
            print('%d neighbors for SMOTE' % k_neighbors)
            smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx,
                            k_neighbors=k_neighbors)
            X_train, y_train, upsampled_groups = transform.upsample(X, y, id_col, smote)
  
        # Drop the id column from the Xs - IMPORTANT!
        X_train.drop(columns=[id_col], inplace=True)
        X_test.drop(columns=[id_col], inplace=True)

        # Format y
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        if select_feats:
            '''Thank you @davide-nd: 
              https://stackoverflow.com/questions/59292631/how-to-combine-gridsearchcv-and-selectfrommodel-to-reduce-the-number-of-features '''
            selector = SelectFromModel(estimator=RandomForestClassifier(max_depth=1, random_state=random_state))
            selector.fit(X_train, y_train)
            print('Columns before: ')
            print(X_train.columns)
           
            X_train = X_train.iloc[:,selector.get_support()]
             
            print('Columns after:')
            print(X_train.columns)
            X_test = X_test.iloc[:,selector.get_support()]

        if method == 'LogisticR':
            
            ''' Perform Scaling
                Thank you @miriam-farber
                https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
            '''
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = transform.scale(X_train, scaler)
            X_test = transform.scale(X_test, scaler)

        # Replace our default classifier clf with an tune_hyperparamsd one
        if tune_hyperparams:
            clf = tune_hyperparams_params(X=X_train, y=y_train, groups=upsampled_groups, 
                                  method=method, random_state=random_state)
        else:
            clf.fit(X_train.values, y_train.values)
    
        print('Training and testing.')

        # Be sure to store the training results so we can check for overfitting later
        y_train_pred = clf.predict(X_train.values)
        y_test_pred = clf.predict(X_test.values)
        y_test_probas = clf.predict_proba(X_test.values)[:, 1]

        # Store TPR and AUC
        # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        fpr, tpr, thresholds = roc_curve(y_test, y_test_probas)
        tprs.append(interp(fpr_mean, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
        # Store predicted and actual target values in dataframe
        train_res.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

        # Calculate feature importance while we're here, using SHAP
        if importance:
            shap_values_fold = metrics.calc_shap(X_train=X_train, X_test=X_test,
                                            model=clf, method=method)
            shap_values.append(shap_values_fold)
            test_indices.append(test_index)
    
    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)

    return {'tprs': tprs, 'aucs': aucs,  
            'shap_values': shap_values, 'test_indices': test_indices, 
            'train_res': train_res, 'test_res': test_res}

def predict(fs, n_lags=None, models=None, n_runs=2, select_feats=False,
            tune_hyperparams=False, importance=False, additional_fields=None):

    common_fields = {'n_lags': n_lags, 'featureset': fs.name, 'features_selected': select_feats, 
                     'tuned': tune_hyperparams, 'target': fs.target_col}
    
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
        tprs = [] # Array of true positive rates
        aucs = []# Array of AUC scores
        fpr_mean = np.linspace(0, 1, 100)
        
        shap_values = list() 
        test_indices = list()

        # Do repeated runs
        for run in range(0, n_runs):
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
            
            # Do training and testing
            res = train_test(X=X, y=y, id_col=fs.id_col, clf=clf, random_state=random_state, 
                             nominal_idx=nominal_idx, method=method, select_feats=select_feats,
                             tune_hyperparams=tune_hyperparams, importance=importance, fpr_mean=fpr_mean)
                
            # Save all relevant stats
            print('Calculating predictive performance for this run.')

            # Get train and test results as separate dictionaries
            train_perf_metrics = metrics.calc_performance_metrics(res['train_res'])
            test_perf_metrics = metrics.calc_performance_metrics(res['test_res'])

            all_res = {
                **{'train_accuracy': train_perf_metrics['accuracy']},
                **{'test_' + str(k): v for k, v in test_perf_metrics.items()}
            }

            common_fields.update({'method': method, 'run': run,
                                  'n_features': X.shape[1], 'n_samples': X.shape[0]})
            
            all_res.update(common_fields)            
  
            # Results are saved for each run
            print('Saving performance metrics for this run.')
            pd.DataFrame([all_res]).to_csv('results/pred_results.csv', mode='a', index=False)
            
            ''' Note the need to extend vs append here, since we're pulling in values
                from a nested fuction call '''
            shap_values.extend(res['shap_values'])
            test_indices.extend(res['test_indices'])
            tprs.extend(res['tprs'])
            aucs.extend(res['aucs'])
 
        # Get and save all the shap values
        if importance:

            ''' Don't forget to drop the groups col and unselected feats.
                Otherwise, we'll have issues with alignment.'''
            X_test, shap_values = metrics.gather_shap(
                X=X.drop(columns=[fs.id_col]), method=method, 
                shap_values=shap_values, test_indices=test_indices)

            filename = '%s_%s_%d_lags'.format(fs.name, method, n_lags)
            if tune_hyperparams:
                filename += '_tune_hyperparamsd'
            filename += '.ob'
            
            with open('feature_importance/X_test_' + filename, 'wb') as fp:
                pickle.dump(X_test, fp)

            with open('feature_importance/shap_' + filename, 'wb') as fp:
                pickle.dump(shap_values, fp)
        
        # Calculate and save AUC Metrics
        test_roc_res, test_auc_res = metrics.get_mean_roc_auc(tprs, aucs, fpr_mean)
        
        common_fields.update({'run': -1}) # Indicates these are aggregated results
        
        test_roc_res.update(common_fields)
        pd.DataFrame().from_dict(test_roc_res).to_csv('results/roc_curves.csv', mode='a', index=False)
        
        test_auc_res.update(common_fields)
        pd.DataFrame([test_auc_res]).to_csv('results/auc_results.csv', mode='a', index=False)

        # Kind of dumb, but lets us quickly use shap values to preselect features in the select_feats() function
        # Will change this later, I'm sure
        if importance:
            return X_test, shap_values 
        else:
            return None
    

    