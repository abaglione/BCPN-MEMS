import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, roc_curve, auc, confusion_matrix
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
    
def calc_performance_metrics(df, actual='actual', pred='pred'):
    print('Calculating standard performance metrics.')
    stats = {}

    stats['accuracy'] = accuracy_score(y_true=df[actual], y_pred=df[pred])
    
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true=df[actual], y_pred=df[pred], pos_label=0, average='binary'
    )
    
    stats.update({'precision': precision, 'recall': recall, 
                  'f1_score': f1_score, 'support': support
                 })

    return stats

def calc_shap(X_train, X_test, model, method, pos_label):
    print('Calculating SHAP values.')
    shap_values = None
    
    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)[pos_label] # i.e., 0 or 1
    elif method == 'SVM':
        X_train_sampled = shap.sample(X_train, 5)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(X_test)

    return shap_values


def gather_shap(shap_values, feats):
    print('Gathering SHAP values.')

    # Combine results from all iterations
    dfs = []

    for i in range(1, len(feats)):
        df = pd.DataFrame(shap_values[i], columns=feats[i])
        dfs.append(df)
 
    # Get the intersection of features selected in each fold and each run
    df_final = pd.concat(dfs, join='inner', axis=0)
    return df_final