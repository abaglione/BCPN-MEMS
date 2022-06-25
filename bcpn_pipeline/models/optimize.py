from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, recall_score, make_scorer
from tune_sklearn import TuneGridSearchCV

# Thank you to Lee Cai, who bootstrapped a similar function in a diff project
# Modifications have been made to suit this project.
def tune_hyperparams(X, y, groups, method, random_state):
    print('Getting tuned classifier using gridsearch.')
    n_jobs = None
    if method == 'LogisticR':
        n_jobs = 1 # Ray local mode - LogisticR doesn't play well when paralellized, for my package versions
        C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        param_grid = [
            {
                'C': C,
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            # {
            #     'C': C,
            #     'penalty': ['elasticnet'], 
            #     'solver': ['saga']
            # }
            # 'max_iter': [3000, 6000, 9000]
        ]
        model = LogisticRegression(random_state=random_state)

    elif method == 'RF':
        n_jobs = 1 # Ray local mode
        param_grid = {
            'n_estimators': [50, 100, 250, 500],
            'max_depth': [1, 2, 3],
            'min_samples_leaf': [1, 2, 3]
        }
        model = RandomForestClassifier(oob_score=True, random_state=random_state)

    elif method == 'XGB':
        n_jobs = 2
        param_grid = {
            'gamma': [0.5, 1, 3],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 6],
            'min_child_weight': [1, 5, 10],
            'n_estimators': [100, 250, 500],
            'objective': ['binary:logistic'],
            'eval_metric': ['logloss'] 
        }
        model = XGBClassifier(use_label_encoder=False, random_state=random_state)

    elif method == 'SVM':
        n_jobs = 2
        param_grid = {
            'C': [1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf'] # Robust to noise - no need to do RFE
        }
        
        model = SVC(probability=True, random_state=random_state)

    print('n_jobs = ' + str(n_jobs))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Create custom scorer for specificity
    scorer = make_scorer(recall_score, pos_label=0)

    tune_search = TuneGridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=cv, scoring=scorer,  n_jobs=n_jobs,
                                   verbose=2)

    tune_search.fit(X.values, y.values, groups)
    return tune_search.best_estimator_