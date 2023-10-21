from pathlib import Path
import shap
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
               method, select_feats, tune, importance, fpr_mean):  # We care more about negative labels - those who don't adhere. Pos label is 0, for us!

    X_test_all = []
    # Set up outer CV
    ''' Need to be splitting at the subject level
        Thank you, Koesmahargyo et al.! '''
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=random_state)

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
                X, y, id_col, smote)

        # Drop the id column from the Xs - IMPORTANT!
        X_train.drop(columns=[id_col], inplace=True)
        X_test.drop(columns=[id_col], inplace=True)

        if method == 'LogisticR' or method == 'SVM':

            ''' Perform Scaling
                Thank you @miriam-farber
                https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
            '''
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = transform.scale(X_train, scaler)
            X_test = transform.scale(X_test, scaler)

        X_test_all.append(X_test)

    return X_test_all


def predict(fs, output_path, n_runs=5, select_feats=False,
            tune=False, importance=False, **kwargs):

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
        ['RF', 'XGB']) if not models else models

    for method, clf in models.items():
        tprs = []  # Array of true positive rates
        aucs = []  # Array of AUC scores
        fpr_mean = np.linspace(0, 1, 100)

        all_res = []

        # Do repeated runs
        for run in range(0, n_runs):
            random_state = run

            # Split into inputs and labels
            X = fs.df.drop(columns=[fs.target_col])
            y = fs.df[fs.target_col]

            # Get list of indices of nominal columns for SMOTE-NC upsampling, used in train_test
            # Safeguard to ensure we're getting the right indices
            nominal_cols = [col for col in X.columns if col in fs.nominal_cols]
            nominal_idx = sorted([X.columns.get_loc(c) for c in nominal_cols])

            # Do training and testing
            X_test_all = train_test(X=X, y=y, id_col=fs.id_col, clf=clf, random_state=random_state,
                                    nominal_idx=nominal_idx, method=method, select_feats=select_feats,
                                    tune=tune, importance=importance, fpr_mean=fpr_mean)

            # Get and save all the shap values
            if importance:
                # Save shap values
                print('Saving shap values for each fold of this run...')
                fold = 0
                for X_test in X_test_all:

                    filename = f'{fs.name}_{method}_{fs.n_lags}_lags'

                    if max_depth:
                        filename += f'_max_depth_{max_depth}'

                    if tune:
                        filename += '_tuned'

                    filename = f'{filename}_run_{run}_fold_{fold}'

                    with open(Path.joinpath(output_path, f'feats_{filename}.pkl'), 'rb') as fp:
                        feats = pickle.load(fp)

                    with open(Path.joinpath(output_path, f'shap_explainer_{filename}.pkl'), 'rb') as fp:
                        explainer = pickle.load(fp)

                    X_test_new = X_test[feats]

                    if X_test_new.shape[0] > 100:
                        X_test_new = shap.utils.sample(X_test_new, nsamples = 100, random_state=random_state)

                    shap_values = explainer(X_test_new)

                    with open(Path.joinpath(output_path, f'shap_values_{filename}.pkl'), 'wb') as fp:
                        pickle.dump(shap_values, fp)

                    fold += 1

            print('Shap task complete!')
