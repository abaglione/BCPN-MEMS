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
        y_true=df[actual], y_pred=df[pred], average='macro'
    )
    
    stats.update({'precision': precision, 'recall': recall, 
                  'f1_score': f1_score, 'support': support
                 })

    return stats

def calc_shap(X_train, X_test, model, method):
    print('Calculating feature importance for this fold.')
    shap_values = None
    
    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    elif method == 'SVM':
        X_train_sampled = shap.sample(X_train, 5)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(X_test)

    return shap_values


def gather_shap(X, method, shap_values, test_indices):
    print('Gathering SHAP stats.')

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    test_indices_all = test_indices[0]
    shap_values_all = np.array(shap_values[0])

    for i in range(1, len(test_indices)):
        test_indices_all = np.concatenate((test_indices_all, test_indices[i]), axis=0)
        
        if method == 'RF' or method == 'SVM': # classifiers with multiple outputs
            shap_values_all = np.concatenate(
                (shap_values_all, np.array(shap_values[i])), axis=1)
        else:
            shap_values_all = np.concatenate((shap_values_all, shap_values[i]), axis=0)

    # Bring back variable names
    X_test = pd.DataFrame(X.iloc[test_indices_all], columns=X.columns)

    return X_test, shap_values_all