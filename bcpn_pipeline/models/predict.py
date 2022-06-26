import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import pickle
from scipy import interp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier

from . import optimize
from . import transform
from . import metrics

def train_test(X, y, id_col, clf, random_state, nominal_idx, 
               method, select_feats, tune, importance, fpr_mean): # We care more about negative labels - those who don't adhere. Pos label is 0, for us!

    tprs = [] # Array of true positive rates
    aucs = []# Array of AUC scores

    shap_tuples = []

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
        imputer = IterativeImputer(random_state=random_state)
        imputer.fit(X_train)
        X_train = transform.impute(X_train, imputer)
        X_test = transform.impute(X_test, imputer)

        try:
            # Perform upsampling to handle class imbalance
            smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx)
            X_train, y_train, upsampled_groups = transform.upsample(X_train, y_train, id_col, smote)
        
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

        # Replace our default classifier clf with a tuned one
        if tune:
            clf = optimize.tune_hyperparams(X=X_train, y=y_train, groups=upsampled_groups, 
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
        # Note we do not change pos_label here. Re-read Gu et al for explanation - focus is specificity for scoring, but not for curves
        fpr, tpr, thresholds = roc_curve(y_test, y_test_probas) 
        tprs.append(interp(fpr_mean, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
        # Store predicted and y_true target values in dataframe
        train_res.append(pd.DataFrame({'y_pred': y_train_pred, 'y_true': y_train}))
        test_res.append(pd.DataFrame({'y_pred': y_test_pred, 'y_true': y_test}))

        if importance:
            feats = list(X_test.columns)
            explainer, shap_values = metrics.calc_shap(X_train, X_test, clf, method, random_state)
            shap_tuples.append((feats, explainer, shap_values))

    train_res = pd.concat(train_res, copy=True)
    test_res = pd.concat(test_res, copy=True)
    
    ret = {'tprs': tprs, 'aucs': aucs, 'train_res': train_res, 'test_res': test_res}
    if importance:
        ret.update({'shap_tuples': shap_tuples})

    return ret

def predict(fs, output_path, n_runs=5, select_feats=False,
            tune=False, importance=False, **kwargs):

    common_fields = {'n_lags': fs.n_lags, 'featureset': fs.name, 'features_selected': select_feats, 
                     'tuned': tune, 'target': fs.target_col}
    
    max_depth = None

    if kwargs:
        common_fields.update(kwargs)
        max_depth = kwargs.get('max_depth')

        if kwargs.get('models'):
            common_fields.pop('models') # This is a dictionary - don't include it

    models = kwargs.get('models')
    models = dict.fromkeys(['LogisticR', 'RF', 'XGB', 'SVM']) if not models else models

    for method, clf in models.items():
        tprs = [] # Array of true positive rates
        aucs = []# Array of AUC scores
        fpr_mean = np.linspace(0, 1, 100)
        
        all_res = []

        # Do repeated runs
        for run in range(0, n_runs):
            random_state = run

            if clf is None:
                
                # Chose to initialize methods here so that random_state could be controlled by the run number
                if method == 'RF' or method == 'XGB':
                    common_fields.update({'max_depth': max_depth})
                    
                    if method == 'RF':
                        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
                    else:
                        clf = XGBClassifier(
                            max_depth=max_depth, 
                            objective='binary:logistic', 
                            eval_metric='logloss',
                            use_label_encoder=False, 
                            random_state=random_state
                            )
                else:
                    common_fields.update({'max_depth': 'NA'})

                    if method == 'LogisticR':
                        clf = LogisticRegression(solver='liblinear', random_state=random_state)

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

            # Get and save all the shap values
            if importance:
                # Save shap values
                print('Saving shap values for each fold of this run...')
                fold = 0                
                for (feats, explainer, shap_values) in res['shap_tuples']:

                    filename = f'{fs.name}_{method}_{fs.n_lags}_lags'
                    
                    if max_depth:
                        filename += f'_max_depth_{max_depth}'

                    if tune:
                        filename += '_tuned'
                    
                    filename = f'{filename}_run_{run}_fold_{fold}'

                    with open(f'{output_path}feats_{filename}.pkl', 'wb') as fp:
                        pickle.dump(feats, fp)

                    with open(f'{output_path}shap_explainer_{filename}.pkl', 'wb') as fp:
                        pickle.dump(explainer, fp)
                        
                    with open(f'{output_path}shap_values_{filename}.pkl', 'wb') as fp:
                        pickle.dump(shap_values, fp)

                    fold += 1
                
            # Save all relevant stats
            print('Calculating predictive performance for this run.')

            # Get train and test results as separate dictionaries
            train_perf_metrics = metrics.calc_performance_metrics(
                y_true=res['train_res']['y_true'], y_pred=res['train_res']['y_pred']
            )
            test_perf_metrics = metrics.calc_performance_metrics(
                y_true=res['test_res']['y_true'], y_pred=res['test_res']['y_pred']
            )

            train_perf_metrics.update({'type': 'train'})
            test_perf_metrics.update({'type': 'test'})
            
            common_fields.update({'method': method, 'run': run, 'random_state': random_state,
                                  'n_features': X.shape[1], 'n_samples': X.shape[0]})
            
            for d in [train_perf_metrics, test_perf_metrics]:
                d.update(common_fields)
                all_res.append(pd.DataFrame([d]))
            
            # TPR and AUC will be calculated across all runs and folds at the very end
            tprs.extend(res['tprs'])
            aucs.extend(res['aucs'])

            print('Prediction task complete!')

        print('Saving performance metrics for all runs.')

        # Combine individual run results
        filename = f'{fs.name}_{method}_{fs.n_lags}_lags'
        
        if max_depth:
            filename += f'_max_depth_{max_depth}'

        if tune:
            filename += '_tuned'

        pd.concat(all_res).to_csv(f'{output_path}{filename}_pred.csv')

        # Calculate aggregate AUC and ROC
        test_roc_res, test_auc_res = metrics.get_mean_roc_auc(tprs, aucs, fpr_mean)
        common_fields.update({'run': -1}) # Indicates these are aggregated results
        
        # Save AUC and ROC
        test_roc_res.update(common_fields)
        test_auc_res.update(common_fields)

        pd.DataFrame.from_dict(test_roc_res).to_csv(f'{output_path}{filename}_roc.csv')
        pd.DataFrame([test_auc_res]).to_csv(f'{output_path}{filename}_auc.csv')

    