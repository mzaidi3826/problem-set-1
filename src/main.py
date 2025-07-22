'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in ./data/
    print("Running ETL (Part 1)...")
    etl.load_and_save_data()

    # PART 2: Call functions/instanciate objects from preprocessing
    print("\nRunning Preprocessing (Part 2)...")
    df_arrests = preprocessing.clean_data()
    df_arrests.to_csv('./data/df_arrests.csv', index = False)  # Save for future parts

    # PART 3: Call functions/instanciate objects from logistic_regression
    print("\nRunning Logistic Regression (Part 3)...")
    df_train, df_test, lr_model = logistic_regression.train_model()

    # PART 4: Call functions/instanciate objects from decision_tree
    print("\nRunning Decision Tree (Part 4)...")
    df_test_updated, dt_model = decision_tree.train_model()

    # PART 5: Call functions/instanciate objects from calibration_plot
    print("\nRunning Calibration and Evaluation (Part 5)...")
    calibration_plot.evaluate_models()


if __name__ == "__main__":
    main()