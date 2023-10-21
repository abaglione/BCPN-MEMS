import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import pandas as pd
from ..consts import OUTPUT_PATH_LAGS, OUTPUT_PATH_PRED, OUTPUT_PATH_LMM
from .predict import repeated_cross_validation, train_test
from .shap_only import predict as shap_only
from .transform import impute
from .metrics import calc_performance_metrics, get_mean_roc_auc
# from .helpers import to_csv_async
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import joblib
import json

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

            kwargs = {'models': models, 'max_depth': max_depth}

            # Pass in max_depth so it gets recorded...dont' ask me why I designed it this way.
            predict_from_mems(fs=all_feats, tune=False, output_path=OUTPUT_PATH_LAGS,
                              select_feats=False, importance=False, repeated_cv=True, **kwargs)


def get_default_clf(method, common_fields, max_depth, random_state):

    # Chose to initialize methods here so that random_state could be controlled by the run number
    if method == 'RF' or method == 'XGB':
        common_fields.update({'max_depth': max_depth})

        if method == 'RF':
            clf = RandomForestClassifier(
                max_depth=max_depth, random_state=random_state)
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
            clf = LogisticRegression(
                solver='liblinear', random_state=random_state)

        elif method == 'SVM':
            clf = SVC(probability=True, random_state=random_state)

    return clf, common_fields


def predict_from_mems(fs, tune, select_feats, output_path=OUTPUT_PATH_PRED, importance=True, repeated_cv=True, **kwargs):

    common_fields = {'n_lags': fs.n_lags, 'featureset': fs.name, 'features_selected': select_feats,
                     'tuned': tune, 'target': fs.target_col}

    max_depth = None

    if kwargs:
        common_fields.update(kwargs)
        max_depth = kwargs.get('max_depth')

        if kwargs.get('models'):
            # This is a dictionary - don't include it
            common_fields.pop('models')

    models = kwargs.get('models')
    models = dict.fromkeys(
        ['LogisticR', 'RF', 'XGB', 'SVM']) if not models else models

    for method, clf in models.items():
        if clf is None:
            clf, common_fields = get_default_clf(
                method, common_fields, max_depth, 42)

        # Split into inputs and labels
        X = fs.df.drop(columns=[fs.target_col])
        y = fs.df[fs.target_col]

        # Get list of indices of nominal columns for SMOTE-NC upsampling, used in train_test
        # Safeguard to ensure we're getting the right indices
        nominal_cols = [col for col in X.columns if col in fs.nominal_cols]
        nominal_idx = sorted([X.columns.get_loc(c) for c in nominal_cols])

        filename = f'{fs.name}_{method}_{fs.n_lags}_lags'

        if max_depth:
            filename += f'_max_depth_{max_depth}'

        if tune:
            filename += '_tuned'

        if repeated_cv:
            repeated_cross_validation(X, y, fs.id_col, clf, nominal_idx,
                                    method, select_feats, tune, common_fields, output_path, filename)

        filename = f'{filename}_final_clf'

        # Build final model
        splitter = StratifiedGroupKFold(n_splits=2, random_state=42, shuffle=True)

        train_idx, test_idx = next(splitter.split(X, y, groups=X[fs.id_col]))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        res, best_estimator = train_test(X_train, y_train, X_test, y_test, fs.id_col, clf,
                                         42, nominal_idx, method, select_feats, tune, importance=importance)
        
        joblib.dump(best_estimator, f'{filename}.joblib', compress=1)
        with open(f'{filename}_params.json', 'w') as f:
            json.dump(best_estimator.get_params(), f)

        train_perf_metrics = calc_performance_metrics(
            y_true=res['train_res']['y_true'], y_pred=res['train_res']['y_pred']
        )
        test_perf_metrics = calc_performance_metrics(
            y_true=res['test_res']['y_true'], y_pred=res['test_res']['y_pred']
        )

        train_perf_metrics.update({'type': 'train'})
        test_perf_metrics.update({'type': 'test'})

        all_res = []

        for d in [train_perf_metrics, test_perf_metrics]:
            d.update({'method': method, 'random_state': 42,
                      'n_features': X.shape[1], 'n_samples': X.shape[0]})
            d.update(common_fields)
            all_res.append(pd.DataFrame([d]))

        pd.concat(all_res).to_csv(Path.joinpath(
            output_path, f'{filename}_pred.csv'))

        res['df_roc'].to_csv(
            Path.joinpath(output_path, f'{filename}_roc.csv'))

        if not importance:
            continue
        
        (feats, explainer, shap_values) = res['shap_tuple']

        with open(Path.joinpath(output_path, f'feats_{filename}.pkl'), 'wb') as fp:
            pickle.dump(feats, fp)

            with open(Path.joinpath(output_path, f'shap_explainer_{filename}.pkl'), 'wb') as fp:
                pickle.dump(explainer, fp)

            with open(Path.joinpath(output_path, f'shap_values_{filename}.pkl'), 'wb') as fp:
                pickle.dump(shap_values, fp)
