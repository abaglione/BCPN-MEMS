import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
import pickle
import xgboost
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def get_stats(df, actual='actual', pred='pred', labels=[0,1]):
    stats = {}
    
    df['accuracy'] = accuracy_score(y_true = df[actual], y_pred=df[pred])
    stats['accuracy'] = df['accuracy'].sum()/df.shape[0]

    df['f1'] = f1_score(y_true = df[actual], y_pred=df[pred])
    stats['f1'] = df['f1'].sum()/df.shape[0]

    df['precision'] = precision_score(y_true = df[actual], y_pred=df[pred])
    stats['precision'] = df['precision'].sum()/df.shape[0]

    df['recall'] = recall_score(y_true = df[actual], y_pred=df[pred])
    stats['recall'] = df['recall'].sum()/df.shape[0]
    
    tn, fp, fn, tp = confusion_matrix(df[actual], df[pred], labels=labels).ravel()
    stats.update({'tpr': tp / (tp + fn), 'fpr': fp / (fp + tn), 
                  'tnr': tn / (tn + fp), 'fnr': fn / (tp + fn)
                 })
    
    return stats

def gather_shap(list_shap_values, list_test_sets, X, method, n_lags, optimize):
    print('Saving SHAP stats...')
    # TODO - bring this back so we can track shap vals
    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    # print(shap_values)
    # print(list_shap_values[1])

    for i in range(1,len(list_test_sets)):
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        if method == 'RF':
            shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
        else:
            shap_values = np.concatenate((shap_values,list_shap_values[i]),axis=0)

    # Bring back variable names    
    X_test = pd.DataFrame(X.iloc[test_set],columns=X.columns)

    # Save the shap info
    filename = 'feature_importance/X_test_' + method + '_' + str(n_lags) + '_lags'
    if optimize:
        filename += '_optimized'
    filename += '.ob'
    with open(filename, 'wb') as fp:
        pickle.dump(X_test, fp)

    filename = 'feature_importance/shap_' + method + '_' + str(n_lags) + '_lags'
    if optimize:
        filename += '_optimized'
    filename += '.ob'
    with open(filename, 'wb') as fp:
        pickle.dump(shap_values, fp)


# credit to Lee Cai, who bootstrapped the original function in a diff project
# Some modifications have been made to suit this project.
def optimize_params(X, y, ids, target_col, method):
    model = None
    
    if method == 'LogisticR':
        param_grid={
            'C': np.logspace(-4, 4, 20), 
            'penalty':['l2'],
            'max_iter': [3000]
        }         
        model = LogisticRegression(random_state=1008)  
    
    elif method == 'RF':
        param_grid = {
            'n_estimators': [50,100,250],
            'max_depth': [2,5,10,25],
            'max_features': ['auto', 'sqrt', 'log2'] 
        }
        model = RandomForestClassifier(oob_score=True,random_state=1008)
        
    elif method == 'XGB':
        param_grid={
            'n_estimators': [50,100,250],
            'max_depth': [3,5],
            'min_child_weight': [1,3],
            'learning_rate': [0.05,0.1,0.15]
        }
        model = xgboost.XGBClassifier(random_state=1008)
    
    elif method == 'SVM':
        param_grid = {
            'C': [1, 10], 
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        } 
        model = SVC(random_state=1008)
    
    # Get RFE and gridsearch objects
    
    rfe = RFECV(model, step=1, cv=5)
    grid = GridSearchCV(estimator=model,param_grid=param_grid, cv=5, scoring='accuracy')
    
    return Pipeline([('feature_selection',rfe),
                     ('clf',grid)])
        
def train_test(X, y, ids, pipeline, method, importance):
    train_res = []
    test_res = []
    list_shap_values = list()
    list_test_sets = list()
    
    '''Leave one group out (LOGO) will function as our leave one subject out (LOSO) cross validation.
       Participant IDs act as group labels. 
       So, at each iteration, one "group" (i.e. one participant id)'s samples will be dropped.
       Seems convoluded but works well.
       '''
    logo = LeaveOneGroupOut()

    print('Training and testing with ' + method + ' model...')
    for train_indices, test_indices in logo.split(X, y, ids):
        
        X_train, y_train = X.loc[train_indices, :], y[train_indices]
        X_test, y_test = X.loc[test_indices, :], y[test_indices]

        pipeline.fit(X_train,y_train)
        
        # Be sure to store the training results so we can ensure we aren't overfitting later
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Store results
        train_res.append(pd.DataFrame({'pred': y_train_pred, 'actual': y_train}))
        test_res.append(pd.DataFrame({'pred': y_test_pred, 'actual':y_test}))

        # Calculate feature importance while we're here
        if importance and method != 'LogisticR':
            if method == 'RF' or method == 'XGB':
                shap_values = shap.TreeExplainer(pipeline['clf']).shap_values(X_test)
            elif method == 'SVM':
                shap_values = shap.KernelExplainer(pipeline['clf'].predict, X_test).shap_values(X_test)

            list_shap_values.append(shap_values)
            list_test_sets.append(test_indices)
    
    return train_res, test_res, list_shap_values, list_test_sets
     
# Adapted from engagement study code - credit to Lee Cai, who co-authored the original code
def predict(fs, n_lags, classifiers=None, optimize=True, importance=True):
    all_results = []
    
    # Split into inputs and labels
    X = fs.df[[col for col in fs.df.columns if col != fs.target_col]]
    y = fs.df[fs.target_col]

    # Sanity check - Test with a random model first
    print('Conducting sanity check using random model...')
    
    res = pd.DataFrame(y).rename(columns={fs.target_col: 'actual'})
    res['pred'] = np.random.randint(0,1, size=len(res))
    stats = get_stats(res, actual='actual', pred='pred')
  
    # Make sure it's terrible :P 
    if stats['accuracy'] > 0.5:
        print('Hmm...the random model did too well. Go back and check for errors in your data and labels.')  
        return None
    else:
        print('Sanity check passed.')
    
        #  ----- Handle class imbalance -----
        print('Conducting upsampling with SMOTE...')
        smote = SMOTE(random_state=50)

        # Preserve columns
        cols = X.columns

        # Upsample using SMOTE
        X, y = smote.fit_resample(X, y)

        # Convert X back into a dataframe and ensure its id col is properly formatted
        X = pd.DataFrame(X,columns=cols,dtype=float)
        X[fs.id_col] = X[fs.id_col].astype(str)

        # Format y
        y = pd.Series(y)

        # Pull out the id column so we can do LOOCV in the next steps
        ids = X[fs.id_col]
        X = X[[col for col in X.columns if col != fs.id_col]]
        
        # If no subset of classifiers is specified, run them all
        if not classifiers:
            classifiers = ['LogisticR', 'RF', 'XGB', 'SVM']
        for method in classifiers:       
            pipeline = None
            if optimize:
                # optimize parameters
                pipeline = optimize_params(X, y, ids, fs.target_col, method)
            else:
                print('Using default params...')
                if method == 'LogisticR':
                    model = LogisticRegression()
                elif method == 'RF':
                    model = RandomForestClassifier(max_depth=5)
                elif method == 'XGB':
                    model = xgboost.XGBClassifier()
                elif method == 'SVM':
                    model = SVC()
                
                pipeline = Pipeline(['clf', model])
  
            train_res, test_res, list_shap_values, list_test_sets = train_test(X, y, ids, pipeline, method, importance)
        
            if importance and method != 'LogisticR':
                gather_shap(list_shap_values, list_test_sets, X, method, n_lags, optimize)

            # Save all relevant stats       
            print('Calculating and saving performance metrics...')
     
            # Get train and test results as separate dictionaries
            train_res = pd.concat(train_res,copy=True)
            test_res = pd.concat(test_res,copy=True)       

            train_res = get_stats(train_res)
            test_res = get_stats(test_res)
            
            # Create a combined results dictionary
            train_test_res = {'test_' + str(k): v for k, v in test_res.items()}
            
            # Add only the accuracy from the training results
            # Just used to ensure we aren't overfitting
            train_test_res['train_accuracy'] = train_res['accuracy']
        
            # Add remaining info 
            train_test_res.update({'n_lags': n_lags, 'featureset': fs.name, 'n_samples': X.shape[0],
                                   'method': method, 'optimized': optimize, 'target': fs.target_col})
            
            all_results.append(train_test_res)
        
        return pd.DataFrame(all_results)

    
    