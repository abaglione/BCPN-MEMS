import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import shap
import pickle
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# credit to Lee Cai
def tune_params(X, y, target_col, method):
    
    model = None
    
    # TODO - adjust n_estimators?
    if method == 'XGB':
        n_estimators = [50,100,200,500]
        max_depth = [3,6,9]
        min_child_weight = [1,3,6]
        learning_rate = [0.05,0.1,0.3,0.5]
        param_grid = dict(n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight,
                          learning_rate=learning_rate)        
        model = xgboost.XGBClassifier()

    elif method == 'RF':
        n_estimators = [50,100,200,500]
        max_depth = [2,5,10]
        max_features = [10,12,15]
        param_grid = dict(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features)
        model = RandomForestClassifier(oob_score=True,random_state=1008)
            
    grid = GridSearchCV(estimator=model,param_grid=param_grid, cv= 5, scoring='accuracy', n_jobs=4)
    grid_result = grid.fit(X,y)
    best_params = grid_result.best_params_        
    
    # Return the final model
    if method == 'XGB':
        return RandomForestClassifier(**best_params)
    elif method == 'RF':     
        return xgboost.XGBClassifier(**best_params)

# Adapted from engagement study code - credit to Lee Cai, who co-authored the original code
def predict(df, id_col, target_col):
    
    # Split into inputs and labels
    X = df[[col for col in df.columns if col != target_col]]
    y = df[target_col]
    
    # Need to tune smote samples?
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

    class_outputs = []
    
    for method in ['RF', 'XGB']:       
        res = []
        list_shap_values = list()
        list_test_sets = list()
        
        # Tune parameters
        model = tune_params(X, y, target_col, method)

        # Get LOOCV train-test splits
        '''Leave one group out (LOGO) will function as our leave one subject out (LOSO) cross validation.
           Participant IDs act as group labels. '''
        logo = LeaveOneGroupOut()

        for train_indices, test_indices in logo.split(X, y, groups):
            list_test_sets.append(X_test_indices)
            
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            
            model.fit(X_train,y_train)
            pred = model_final.predict(X_test)
            
            df = pd.DataFrame({'pred':pred, 'actual':y_test})
            res.append(df)
            
            shap_values = shap.TreeExplainer(model_final).shap_values(X_test)
            # print(shap_values)

            list_shap_values.append(shap_values)
            
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
        filename = method + '_' + target_col
        with open('feature_importance/X_test_' + filename + '.ob', 'wb') as fp:
            pickle.dump(X_test, fp)

        with open('feature_importance/shap_' + filename + '.ob', 'wb') as fp:
            pickle.dump(shap_values, fp)

        # Save all relevant stats
        res_df = pd.concat(res,copy=True)

        res_df['acc'] = accuracy_score(y_true = res_df['actual'], y_pred=res_df['pred'])
        acc = res_df['acc'].sum()/res_df.shape[0]

        res_df['f1'] = f1_score(y_true = res_df['actual'], y_pred=res_df['pred'])
        f1 = res_df['f1'].sum()/res_df.shape[0]

        res_df['precision'] = precision_score(y_true = res_df['actual'], y_pred=res_df['pred'])
        precision = res_df['precision'].sum()/res_df.shape[0]

        res_df['recall'] = recall_score(y_true = res_df['actual'], y_pred=res_df['pred'])
        recall = res_df['recall'].sum()/res_df.shape[0]
        
        class_outputs.append(
            {
                'method':method,'target':target_col,
                'accuracy':acc, 'f1_score': f1, 'precision': precision, 'recall': recall
            }
        )

        class_outputs = pd.DataFrame(class_outputs)
        class_outputs.to_csv('results/class_outcomes.csv',index=False)
    


    
    