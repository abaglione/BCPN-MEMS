import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import shap
import pickle
from scipy import interp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, recall_score, make_scorer
from tune_sklearn import TuneGridSearchCV

from . import transform
from . import metrics

# Thank you to Lee Cai, who bootstrapped a similar function in a diff project
# Modifications have been made to suit this project.
def tune_hyperparams(X, y, groups, method, random_state, pos_label):
    print('Getting tuned classifier using gridsearch.')
    # n_jobs = -1
    n_jobs = 1
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
            'max_depth': [1, 2, 3],
            'min_samples_leaf': [1, 2, 3]
        }
        model = RandomForestClassifier(oob_score=True, random_state=random_state)

    elif method == 'SVM':
        # n_jobs = None
        param_grid = {
            'C': [1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf'] # Robust to noise - no need to do RFE
        }
        
        model = SVC(probability=True, random_state=random_state)

    print('n_jobs = ' + str(n_jobs))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Create custom scorer
    nonadherent_scorer = make_scorer(recall_score, pos_label=pos_label)

    tune_search = TuneGridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=cv, scoring=nonadherent_scorer,  n_jobs=n_jobs,
                                   verbose=2)

    tune_search.fit(X.values, y.values, groups)
    return tune_search.best_estimator_

def train_test(X, y, id_col, clf, random_state, nominal_idx, 
               method, select_feats, tune, importance, fpr_mean, pos_label=0): # We care more about negative labels - those who don't adhere. Pos label is 0, for us!
    tprs = [] # Array of true positive rates
    aucs = []# Array of AUC scores

    shap_values = list() 
    feats = list()

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
        imputer.fit(X_train)
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

            X_train = X_train.iloc[:,selector.get_support()]
            X_test = X_test.iloc[:,selector.get_support()]

        if method == 'LogisticR':
            
            ''' Perform Scaling
                Thank you @miriam-farber
                https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
            '''
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = transform.scale(X_train, scaler)
            X_test = transform.scale(X_test, scaler)

        # Replace our default classifier clf with an tuned one
        if tune:
            clf = tune_hyperparams(X=X_train, y=y_train, groups=upsampled_groups, 
                                   method=method, random_state=random_state, pos_label=pos_label)
        else:
            clf.fit(X_train.values, y_train.values)
    
        print('Training and testing.')

        # Be sure to store the training results so we can check for overfitting later
        y_train_pred = clf.predict(X_train.values)
        y_test_pred = clf.predict(X_test.values)
        y_test_probas = clf.predict_proba(X_test.values)[:, 1]

        # Store TPR and AUC
        # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        fpr, tpr, thresholds = roc_curve(y_test, y_test_probas, pos_label=pos_label)
        tprs.append(interp(fpr_mean, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
        # Store predicted and actual target values in dataframe
        train_res.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual': y_test}))

        # Calculate feature importance while we're here, using SHAP
        if importance:
            shap_values_fold = metrics.calc_shap(X_train=X_train, X_test=X_test, model=clf, method=method, pos_label=pos_label)
            shap_values.append(shap_values_fold)
            feats.append(list(X_test.columns))
    
    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)

    return {'tprs': tprs, 'aucs': aucs,  
            'shap_values': shap_values, 'features': feats, 
            'train_res': train_res, 'test_res': test_res}

def predict(fs, output_path, write_header, n_lags=None, models=None, n_runs=5, select_feats=False,
            tune=False, importance=False, additional_fields=None):

    mode = 'w' if write_header else 'a' # Set up mode

    common_fields = {'n_lags': n_lags, 'featureset': fs.name, 'features_selected': select_feats, 
                     'tuned': tune, 'target': fs.target_col}
    
    if additional_fields:
        common_fields.update(additional_fields)
    
    ''' Hacky - added just for predict_from_mems. 
    Would need to remove if re-tuning lags!
    Need to change pipeline later to accomodate passing in max_depth'''
    if n_lags == 2: # Study day and study week
        max_depth = 1
    else: # Study month
        max_depth = 5 
    
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
        
        shap_values = [] 
        feats = []
        
        all_res = []

        # Do repeated runs
        for run in range(0, n_runs):
            random_state = run

            # Get the correct classifier if it doesn't exist
            if clf is None:
                if method == 'LogisticR':
                    clf = LogisticRegression(solver='liblinear', random_state=random_state)
                elif method == 'RF':
                    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
                elif method == 'SVM':
                    clf = SVC(probability=True, random_state=random_state)

            # Do training and testing
            print('Run %i of %i for %s model.' % (run + 1, n_runs, method))

            # Split into inputs and labels
            X = fs.df.drop(columns=[fs.target_col])
            y = fs.df[fs.target_col]

            # Get list of indices of nominal columns for SMOTE-NC upsampling, used in train_test
            # Safeguard to ensure we're getting the right indices
            nominal_cols = [col for col in X.columns if col in fs.nominal_cols]
            nominal_idx = sorted([X.columns.get_loc(c) for c in nominal_cols])
            
            # Do training and testing
            res = train_test(X=X, y=y, id_col=fs.id_col, clf=clf, random_state=random_state, 
                             nominal_idx=nominal_idx, method=method, select_feats=select_feats,
                             tune=tune, importance=importance, fpr_mean=fpr_mean)
                
            # Save all relevant stats
            print('Calculating predictive performance for this run.')

            # Get train and test results as separate dictionaries
            train_perf_metrics = metrics.calc_performance_metrics(res['train_res'])
            test_perf_metrics = metrics.calc_performance_metrics(res['test_res'])

            train_perf_metrics.update({'type': 'train'})
            test_perf_metrics.update({'type': 'test'})
            
            common_fields.update({'method': method, 'run': run,
                                  'n_features': X.shape[1], 'n_samples': X.shape[0]})
            
            for d in [train_perf_metrics, test_perf_metrics]:
                d.update(common_fields)
                all_res.append(pd.DataFrame([d]))
            
            shap_values.extend(res['shap_values'])
            feats.extend(res['features'])
            tprs.extend(res['tprs'])
            aucs.extend(res['aucs'])
        
        print('Saving performance metrics for all runs.')
        
        # Save individual run results
        all_res = pd.concat(all_res)
        all_res.to_csv(output_path + 'pred.csv', encoding='utf-8', mode=mode, header=write_header,index=False)
        
        # Calculate and save AUC Metrics
        test_roc_res, test_auc_res = metrics.get_mean_roc_auc(tprs, aucs, fpr_mean)
        
        common_fields.update({'run': -1}) # Indicates these are aggregated results
        
        test_roc_res.update(common_fields)
        pd.DataFrame().from_dict(test_roc_res).to_csv(output_path + 'roc.csv', encoding='utf-8', mode=mode, header=write_header,index=False)
        
        test_auc_res.update(common_fields)
        pd.DataFrame([test_auc_res]).to_csv(output_path + 'auc.csv', encoding='utf-8', mode=mode, header=write_header,index=False)

        # Get and save all the shap values
        if importance:
            shap_df = metrics.gather_shap(shap_values, feats)

            filename = fs.name + '_' + method + '_' + str(fs.n_lags)
            if tune:
                filename += '_tuned'
            filename += '.ob'
            
            with open(output_path + 'shap_df_' + filename, 'wb') as fp:
                pickle.dump(shap_df, fp)
        
        print('Prediction task complete!')
    

    