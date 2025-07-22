'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
def train_model():
    # Load the dataset
    df = pd.read_csv('./data/df_arrests.csv')

    # Split into train/test
    df_arrests_train, df_arrests_test = train_test_split(
        df, 
        test_size = 0.3, 
        shuffle = True, 
        stratify = df['y'], 
        random_state = 42
    )

    # Define features and labels
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']

    # Create param grid
    param_grid = {'C': [0.01, 1, 100]}

    # Initialize logistic regression model
    lr_model = lr(solver='liblinear')

    # Grid search with 5-fold cross-validation
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
    gs_cv.fit(X_train, y_train)

    # Results
    best_C = gs_cv.best_params_['C']
    print("What was the optimal value for C?")
    print(best_C)

    if best_C == min(param_grid['C']):
        regularization_strength = "most regularization"
    elif best_C == max(param_grid['C']):
        regularization_strength = "least regularization"
    else:
        regularization_strength = "middle regularization"

    print(f"Did it have the most or least regularization? Answer: {regularization_strength}")

    # Predict on test set
    df_arrests_test['pred_lr'] = gs_cv.predict_proba(X_test)[:, 1]

    # Optionally save to CSV for use in next steps
    df_arrests_train.to_csv('./data/df_arrests_train.csv', index = False)
    df_arrests_test.to_csv('./data/df_arrests_test.csv', index = False)

    return df_arrests_train, df_arrests_test, gs_cv.best_estimator_

