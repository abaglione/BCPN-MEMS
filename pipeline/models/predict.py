import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
import pickle
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error

def get_stats(res_df, actual='actual', pred='pred'):
    res_df['acc'] = accuracy_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    acc = res_df['acc'].sum()/res_df.shape[0]

    res_df['f1'] = f1_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    f1 = res_df['f1'].sum()/res_df.shape[0]

    res_df['precision'] = precision_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    precision = res_df['precision'].sum()/res_df.shape[0]

    res_df['recall'] = recall_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    recall = res_df['recall'].sum()/res_df.shape[0]
    
    return acc, f1, precision, recall

# credit to Lee Cai
def tune_params(X, y, target_col, method):
    model = None
    
    if method == 'LogisticR':
        param_grid={
            'C': np.logspace(-4, 4, 20), 
            'penalty':['l1','l2'], 
            'max_iter': [3000]
        }         
        model = LogisticRegression()  
    
    elif method == 'RF':
        print(len(X.columns) / 2)
        param_grid = {
            'n_estimators': [50,100,200,500],
            'max_depth': [2,5,10],
            'max_features': [len(X.columns) / 2, len(X.columns)] 
        }
        model = RandomForestClassifier(oob_score=True,random_state=1008)
        
    elif method == 'XGB':
        param_grid={
            'n_estimators': [50,100,200,500],
            'max_depth': [3,6,9],
            'min_child_weight': [1,3,6],
            'learning_rate': [0.05,0.1,0.3,0.5]
        }
        model = xgboost.XGBClassifier()
      
    grid = GridSearchCV(estimator=model,param_grid=param_grid, cv= 5, scoring='accuracy', n_jobs=4)
    grid_result = grid.fit(X,y)
    best_params = grid_result.best_params_        
    
    # Return the final model
    if method == 'LogisticR':
        return LogisticRegression(**best_params)
    elif method == 'RF':
        return RandomForestClassifier(**best_params)
    elif method == 'XGB':
        return xgboost.XGBClassifier(**best_params)
     
# Adapted from engagement study code - credit to Lee Cai, who co-authored the original code
def predict(df, id_col, target_col, sanity_check=False):
    all_results = []
    
    # Split into inputs and labels
    X = df[[col for col in df.columns if col != target_col]]
    y = df[target_col]

    # Sanity check - Test with a random model first
    print('Conducting sanity check using random model...')
    
    res = pd.DataFrame(y).rename(columns={target_col: 'actual'})
    res['pred'] = np.random.randint(0,1, size=len(res))
    acc, f1, precision, recall = get_stats(res, actual='actual', pred='pred')
  
    # Make sure it's terrible :P 
    if acc > 0.5:
        print('Hmm...the random model did too well. Go back and check for errors in your data and labels.')  
        return None
    else:
        print('Sanity check passed. Starting prediction tasks...')
    
        # Need to tune smote samples?
        print('Conducting upsampling with SMOTE...')
        smote = SMOTE(random_state=50)

        # Preserve columns
        cols = X.columns

        # Upsample using SMOTE
        X, y = smote.fit_resample(X, y)

        # Convert X back into a dataframe and ensure its id col is properly formatted
        X = pd.DataFrame(X,columns=cols,dtype=float)
        X[id_col] = X[id_col].astype(str)

        # Format y
        y = pd.Series(y)

        # Pull out the id column so we can do LOOCV in the next steps
        ids = X[id_col]
        X = X[[col for col in X.columns if col != id_col]]


        for method in ['LogisticR', 'RF', 'XGB']:       
            res = []
            list_shap_values = list()
            list_test_sets = list()

            # Tune parameters
            print('Tuning params with gridsearch...')
            model = tune_params(X, y, target_col, method)

            print('Training and testing with ' + method + ' classifier')
            
            # Get LOOCV train-test splits
            '''Leave one group out (LOGO) will function as our leave one subject out (LOSO) cross validation.
               Participant IDs act as group labels. '''
            logo = LeaveOneGroupOut()
            
            fold = 0
            for train_indices, test_indices in logo.split(X, y, ids):
                list_test_sets.append(test_indices)

                X_train, y_train = X.loc[train_indices, :], y[train_indices]
                X_test, y_test = X.loc[test_indices, :], y[test_indices]

                model.fit(X_train,y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Store results
                df = pd.DataFrame({'pred':y_test_pred, 'actual':y_test})
                res.append(df)

#                 shap_values = shap.TreeExplainer(model).shap_values(X_test)
                # print(shap_values)

#                 list_shap_values.append(shap_values)
                
                fold += 1

            # TODO - bring this back so we can track shap vals
            # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

            # Combine results from all iterations
    #         test_set = list_test_sets[0]
    #         shap_values = np.array(list_shap_values[0])
    #         # print(shap_values)
    #         # print(list_shap_values[1])

    #         for i in range(1,len(list_test_sets)):
    #             test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
    #             if method == 'RF':
    #                 shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
    #             else:
    #                 shap_values = np.concatenate((shap_values,list_shap_values[i]),axis=0)

    #         # Bring back variable names    
    #         X_test = pd.DataFrame(X.iloc[test_set],columns=X.columns)

    #         # Save the shap info
    #         filename = method + '_' + target_col
    #         with open('feature_importance/X_test_' + filename + '.ob', 'wb') as fp:
    #             pickle.dump(X_test, fp)

    #         with open('feature_importance/shap_' + filename + '.ob', 'wb') as fp:
    #             pickle.dump(shap_values, fp)

            # Save all relevant stats
            res_df = pd.concat(res,copy=True)
            acc, f1, precision, recall = get_stats(res_df, actual='actual', pred='pred')
        
            all_results.append(
                {
                    'method':method,'target':target_col,
                    'accuracy':acc, 'f1_score': f1, 'precision': precision, 'recall': recall
                }
            )

        return pd.DataFrame(all_results)

    
    