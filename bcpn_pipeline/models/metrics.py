import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, recall_score, roc_curve, auc, confusion_matrix
from shap.explainers import Linear, Tree, Sampling

def get_mean_roc_auc(tprs, aucs, fpr_mean):
    print('Getting mean ROC AUC stats.')
    tpr_mean = np.mean(tprs, axis=0)
    tpr_mean[-1] = 1.0
    auc_mean = auc(fpr_mean, tpr_mean)
    auc_std = np.std(aucs)

    return {'tpr_mean': tpr_mean, 'fpr_mean': fpr_mean}, {'auc_mean': auc_mean, 'auc_std': auc_std} 

def get_agg_auc(y_all, y_probas_all):
    print('Getting aggregate ROC AUC stats.')
    y_all = np.concatenate(y_all)
    y_probas_all = np.concatenate(y_probas_all)
    
    # https://stackoverflow.com/questions/57756804/roc-curve-with-leave-one-out-cross-validation-in-sklearn
    fpr, tpr, thresholds = roc_curve(y_all, y_probas_all)
    return {'auc': auc(fpr, tpr)}, tpr, fpr
    
# TODO: Add sensitivity and specificity, ignore support
# Check Gu et al.
def calc_performance_metrics(y_true, y_pred):
    print('Calculating standard performance metrics.')
    stats = {}

    stats['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    precision, sensitivity, f1_score, support = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average='binary'
    )

    specificity = recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)
    
    stats.update({'precision': precision, 'sensitivity': sensitivity, 
                  'specificity': specificity, 'f1_score': f1_score, 'support': support})

    return stats

def calc_shap(X_train, X_test, model, method, random_state, pos_label=1):
    shap_values = None
    explainer = None
    nsamples_max = 100

    print('Calculating SHAP values.')    

    if X_train.shape[0] > nsamples_max:
        X_train = X_train.sample(n = nsamples_max)

    if X_test.shape[0] > nsamples_max:
        X_test = X_test.sample(n = nsamples_max)

    if method == 'LogisticR':
        explainer = Linear(model=model, data=X_train)

    elif method == 'RF' or method == 'XGB':
        explainer = Tree(model=model, data=X_train)
        
    elif method == 'SVM':
        shap_values = Sampling(model=model, data=X_train)

    shap_values = explainer.shap_values(X=X_test)
    return explainer, shap_values