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

def save_res_auc(res, tpr, fpr):
    auc_df = pd.DataFrame({'test_tpr': tpr, 'test_fpr': fpr})
    auc_df['method'] = res['method']
    auc_df['optimized'] = res['optimized']
    auc_df['n_lags'] = res['n_lags']
    auc_df['featureset'] = res['featureset']
    auc_df.to_csv('results/auc_results.csv', mode='a', index=False)
    pd.DataFrame([res]).to_csv('results/pred_results.csv', mode='a', index=False)

def get_mean_auc(tprs, aucs, mean_fpr):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return {'mean_auc': mean_auc, 'std_auc': std_auc}, mean_tpr, mean_fpr

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
        X_train_summary = shap.kmeans(X_train, 10)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train).shap_values(X_test)

    return shap_values


def gather_shap(X, method, shap_values, test_indices):
    print('Gathering SHAP stats.')

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
        n_jobs = 1
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
    return tune_search.best_estimator_


def train_test(fs, method, clf, n_lags, random_state, 
               optimize, importance, common_fields):

    ''' Set up cross validation and AUC metrics
        Need to be splitting at the subject level
        Thank you, Koesmahargyo et al.! ''' 
    
    # Get list of indices of nominal columns for SMOTE-NC upsampling, used in train_test
    nominal_idx = sorted([fs.df.columns.get_loc(c) for c in fs.nominal_cols])
#     print('Nominal Indices')
#     print(nominal_idx)

    # Split into inputs and labels
    X = fs.df[[col for col in fs.df.columns if col != fs.target_col]]
    y = fs.df[fs.target_col]
    
    # Set up outer CV and AUC metric storage objects
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    auc_type = 'mean'
    tprs = [] # Array of true positive rates
    aucs = []# Array of AUC scores
    mean_fpr = np.linspace(0, 1, 100)

    # Begin train/test
    
    train_res = [] # Array of dataframes of true vs pred labels
    test_res = [] # Array of dataframes of true vs pred labels
    all_shap_values = list() 
    all_test_index = list()

    for train_index, test_index in cv.split(X=X, y=y, groups=X[fs.id_col]):
        X_train, y_train = X.loc[train_index, :], y[train_index]
        X_test, y_test = X.loc[test_index, :], y[test_index]

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

        # Replace our default classifier clf with an optimized one
        if optimize:
            print('Getting optimized classifier, using gridsearch.')
            clf = optimize_params(X=X_train, y=y_train, groups=upsampled_groups, 
                                  method=method, random_state=random_state)

        print('Training and testing.')
        clf.fit(X_train.values, y_train.values)

        # Be sure to store the training results so we can check for overfitting later
        y_train_pred = clf.predict(X_train.values)
        y_test_pred = clf.predict(X_test.values)
        y_test_probas = clf.predict_proba(X_test.values)[:, 1]

        # Store the AUC metrics according to the type of AUC we need (aggregate or mean)
        if auc_type == 'agg':
            y_test_all.append(y_test)
            y_test_probas_all.append(y_test_probas)
            
        elif auc_type == 'mean':
            # Store TPR and AUC
            # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
            fpr, tpr, thresholds = roc_curve(y_test, y_test_probas)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        # Store predicted and actual target values in dataframe
        train_res.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

        # Calculate feature importance while we're here, using SHAP
        print('Calculating feature importance.')
        if importance:
            shap_values = calc_shap(X_train=X_train, X_test=X_test,
                                    model=clf, method=method)
            all_shap_values.append(shap_values)
            all_test_index.append(test_index)

    # Get and save all the shap values
    if importance:
        ''' Don't forget to drop the groups col and unselected feats.
            Otherwise, we'll have issues with alignment.'''
        
        X_test, shap_values = gather_shap(
            X=X.drop(columns=[fs.id_col]), method=method, 
            shap_values=all_shap_values, test_indices=all_test_index
        )

        filename = fs.name + '_' + method + '_' + str(n_lags) + '_lags'
        if optimize:
            filename += '_optimized'
        filename += '.ob'
        
        with open('feature_importance/X_test_' + filename, 'wb') as fp:
            pickle.dump(X_test, fp)

        with open('feature_importance/shap_' + filename, 'wb') as fp:
            pickle.dump(shap_values, fp)

    # Save all relevant stats
    print('Calculating predictive performance metrics.')

    # Get train and test results as separate dictionaries
    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)

    train_perf_metrics = get_performance_metrics(train_res)
    test_perf_metrics = get_performance_metrics(test_res)
    
    # Calculate AUC Metrics
    print('Calculating AUC metrics.')
    if auc_type == 'agg':
        test_auc_metrics, test_tpr, test_fpr = get_agg_auc(y_test_all, y_test_probas_all)

    elif auc_type == 'mean':
        test_auc_metrics, test_tpr, test_fpr = get_mean_auc(tprs, aucs, mean_fpr)
    
    ''' Create a combined results and auc dictionary
        Add only the accuracy from the training results
          just used to ensure we aren't overfitting'''
    
    all_res = {
        **{'train_accuracy': train_perf_metrics['accuracy']},
        **{'test_' + str(k): v for k, v in test_perf_metrics.items()},
        **{'test_' + str(k): v for k, v in test_auc_metrics.items()},
    }

    all_res.update(common_fields)
    all_res.update({'n_features': X.shape[1], 'n_samples': X.shape[0]})
    save_res_auc(all_res, test_tpr, test_fpr)
#     return all_res

def predict(fs, n_lags=None, classifiers=None, n_runs=5, 
            optimize=False, importance=False, additional_fields=None):

    all_results = []
    common_fields = {'n_lags': n_lags, 'featureset': fs.name, 'target': fs.target_col}
    
    ''' Do at least one non-optimized run '''
    opt_opts = [False] 
    opt_opts = opt_opts + [True] if optimize else opt_opts
    
    if additional_fields:
        common_fields.update(additional_fields)
        
    # Do repeated runs
    for run in range(0, n_runs):

        # If no subset of classifiers is specified, start with all default classifiers
        if not classifiers:
            classifiers = {
                'LogisticR': LogisticRegression(solver='liblinear', random_state=run), 
                'RF': RandomForestClassifier(max_depth=1, random_state=run), 
                'XGB': xgboost.XGBClassifier(random_state=run), 
                'SVM': SVC(probability=True, random_state=run)
            }

        for method, clf in classifiers.items():
            
            # Do at least a set of non-optimized runs
            for opt in opt_opts:
                print('Run %i of %i for %s model.' % (run + 1, n_runs, method))
                
                common_fields.update({'method': method, 'optimized': opt, 'run': run})
                train_test(fs, method=method, clf=clf, n_lags=n_lags, 
                           random_state=run, optimize=opt, importance=importance,
                           common_fields = common_fields)
#             all_results.append(res)

#     return pd.DataFrame(all_results)

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
            classifiers = {
                'RF': RandomForestClassifier(max_depth=max_depth, random_state=max_depth)
            }
            
            predict(all_feats, n_lags, classifiers=classifiers,
                    optimize=False, importance=False, n_runs=5,
                    additional_fields={'max_depth': max_depth})
