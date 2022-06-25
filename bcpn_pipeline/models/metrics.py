import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, recall_score, roc_curve, auc, confusion_matrix
import shap

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
    nsamples_max = 100

    print('Calculating SHAP values.')    

    if X_train.shape[0] > nsamples_max:
        X_train = shap.sample(X_train, nsamples=nsamples_max, random_state=random_state)

    if X_test.shape[0] > nsamples_max:
        X_test = shap.sample(X_test, nsamples=nsamples_max, random_state=random_state)

    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    elif method == 'SVM':
        shap_values = shap.KernelExplainer(model.predict_proba, X_train).shap_values(X_test)

    return shap_values