import numpy as np
from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTENC
import pickle
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from ..consts import FPR_MEAN
from . import optimize
from . import transform
from .metrics import calc_performance_metrics, get_mean_roc_auc, calc_shap


def train_test(X_train, y_train, X_test, y_test, id_col, clf, random_state, nominal_idx,
               method, select_feats, tune, importance):

    # Do imputation
    imputer = IterativeImputer(random_state=random_state)
    imputer.fit(X_train)
    X_train = transform.impute(X_train, imputer)
    X_test = transform.impute(X_test, imputer)

    try:
        # Perform upsampling to handle class imbalance
        smote = SMOTENC(random_state=random_state,
                        categorical_features=nominal_idx)
        X_train, y_train, upsampled_groups = transform.upsample(
            X_train, y_train, id_col, smote)

    except ValueError:
        # Set n_neighbors = n_samples
        # Not great if we have a really small sample size. Hmm.
        k_neighbors = (y_train == 1).sum() - 1
        print('%d neighbors for SMOTE' % k_neighbors)
        smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx,
                        k_neighbors=k_neighbors)

        X_train, y_train, upsampled_groups = transform.upsample(
            X_train, y_train, id_col, smote)

    # Drop the id column from the Xs - IMPORTANT!
    X_train.drop(columns=[id_col], inplace=True)
    X_test.drop(columns=[id_col], inplace=True)

    # Format y
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    if select_feats:
        '''Thank you @davide-nd:
            https://stackoverflow.com/questions/59292631/how-to-combine-gridsearchcv-and-selectfrommodel-to-reduce-the-number-of-features '''
        selector = SelectFromModel(estimator=RandomForestClassifier(
            max_depth=1, random_state=random_state))
        selector.fit(X_train, y_train)

        X_train = X_train.iloc[:, selector.get_support()]
        X_test = X_test.iloc[:, selector.get_support()]

    if method == 'LogisticR' or method == 'SVM':

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

    print('Getting predictions...')

    # Be sure to store the training results so we can check for overfitting later
    y_train_pred = clf.predict(X_train.values)
    y_test_pred = clf.predict(X_test.values)
    y_test_probas = clf.predict_proba(X_test.values)[:, 1]

    # Store TPR and AUC
    # Thank you sklearn documentation https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    # Note we do not change pos_label here. Re-read Gu et al for explanation - focus is specificity for scoring, but not for curves
    fpr, tpr, thresholds = roc_curve(y_test, y_test_probas)
    roc_auc = auc(fpr, tpr)
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc})

    tpr = interp(FPR_MEAN, fpr, tpr)
    tpr[0] = 0.0

    # Store predicted and y_true target values in dataframe
    train_res = pd.DataFrame({'y_pred': y_train_pred, 'y_true': y_train})
    test_res = pd.DataFrame({'y_pred': y_test_pred, 'y_true': y_test})

    res = {'train_res': train_res, 'test_res': test_res,
           'auc': roc_auc, 'tpr': tpr, 'df_roc': df_roc}
    
    if importance:
        feats = list(X_test.columns)
        explainer, shap_values = calc_shap(
            X_train, X_test, clf, method, random_state)
        res['shap_tuple'] = (feats, explainer, shap_values)

    return res, clf


def cross_validate(X, y, id_col, clf, random_state, nominal_idx, method, select_feats,
                   tune):

    res_all = {
        'tpr': [],  # Array of true positive rates
        'auc': [],  # Array of AUC scores
        'train_res': [],  # Array of dataframes of true vs pred labels
        'test_res': [],  # Array of dataframes of true vs pred labels
    }

    # Set up outer CV
    ''' Need to be splitting at the subject level
        Thank you, Koesmahargyo et al.! '''

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=random_state)

    # Do prediction task
    for train_index, test_index in cv.split(X=X, y=y, groups=X[id_col]):
        X_train, y_train = X.loc[train_index, :], y[train_index]
        X_test, y_test = X.loc[test_index, :], y[test_index]

        # Do training and testing
        res, _ = train_test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                            id_col=id_col, clf=clf, random_state=random_state,
                            nominal_idx=nominal_idx, method=method, select_feats=select_feats,
                            tune=tune, importance=False)

        for k, v in res.items():
            res_all[k].append(v)

    train_res = pd.concat(res_all.pop('train_res'), copy=True)
    test_res = pd.concat(res_all.pop('test_res'), copy=True)

    train_perf_metrics = calc_performance_metrics(
        y_true=train_res['y_true'], y_pred=train_res['y_pred']
    )
    test_perf_metrics = calc_performance_metrics(
        y_true=test_res['y_true'], y_pred=test_res['y_pred']
    )

    train_perf_metrics.update({'type': 'train'})
    test_perf_metrics.update({'type': 'test'})

    res_all.update({
        'train_perf_metrics': train_perf_metrics,
        'test_perf_metrics': test_perf_metrics
    })

    return res_all


def repeated_cross_validation(X, y, id_col, clf, nominal_idx, method, select_feats, tune,
                              common_fields, output_path, filename,
                              run_repeats=5):

    tpr = []  # Array of true positive rates
    auc = []  # Array of AUC scores

    all_res = []

    # Do repeated runs
    for run in range(0, run_repeats):
        print('Run %i of %i for %s model.' %
              (run + 1, run_repeats, method))
        random_state = run

        res = cross_validate(X, y, id_col, clf, random_state, nominal_idx, method,
                             select_feats, tune)

        # Get train and test results as separate dictionaries
        for d in [res['train_perf_metrics'], res['test_perf_metrics']]:
            d.update({'method': method, 'run': run, 'random_state': random_state,
                      'n_features': X.shape[1], 'n_samples': X.shape[0]})
            d.update(common_fields)
            all_res.append(pd.DataFrame([d]))

        # TPR and AUC will be calculated across all runs and folds at the very end
        tpr.extend(res['tpr'])
        auc.extend(res['auc'])

        print('Prediction task complete!')

    print('Saving performance metrics for all runs.')

    pd.concat(all_res).to_csv(Path.joinpath(
        output_path, f'{filename}_pred.csv'))

    # Calculate aggregate AUC and ROC
    test_roc_res, test_auc_res = get_mean_roc_auc(tpr, auc, FPR_MEAN)

    # Indicates these are aggregated results
    common_fields.update({'run': -1})

    # Save AUC and ROC
    test_roc_res.update(common_fields)
    test_auc_res.update(common_fields)

    pd.DataFrame.from_dict(test_roc_res).to_csv(
        Path.joinpath(output_path, f'{filename}_roc.csv'))

    pd.DataFrame([test_auc_res]).to_csv(
        Path.joinpath(output_path, f'{filename}_auc.csv'))
