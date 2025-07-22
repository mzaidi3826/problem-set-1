'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def train_model():
    # Load train and test sets from CSV
    df_arrests_train = pd.read_csv('./data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('./data/df_arrests_test.csv')

    # Define features and target
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]

    # Create parameter grid for max_depth
    param_grid_dt = {'max_depth': [2, 5, 10]}

    # Initialize Decision Tree model
    dt_model = DTC(random_state=42)

    # GridSearchCV
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(X_train, y_train)

    # Best max_depth
    best_depth = gs_cv_dt.best_params_['max_depth']
    print("What was the optimal value for max_depth?")
    print(best_depth)

    if best_depth == min(param_grid_dt['max_depth']):
        reg_strength = "most regularization"
    elif best_depth == max(param_grid_dt['max_depth']):
        reg_strength = "least regularization"
    else:
        reg_strength = "middle regularization"

    print("Did it have the most or least regularization? Answer:", reg_strength)

    # Predict for test set
    df_arrests_test['pred_dt'] = gs_cv_dt.predict_proba(X_test)[:, 1]

    # Save updated test set to CSV for Part 5
    df_arrests_test.to_csv('./data/df_arrests_test.csv', index = False)

    return df_arrests_test, gs_cv_dt.best_estimator_