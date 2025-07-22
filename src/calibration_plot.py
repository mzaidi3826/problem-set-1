'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins = 10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins = n_bins)
    
    #Create the Seaborn plot
    sns.set(style = "whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker = 'o', label = "Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc = "best")
    plt.show()
    
def evaluate_models():
    # Load test dataset with predictions
    df = pd.read_csv('./data/df_arrests_test.csv')

    y_true = df['y']
    y_pred_lr = df['pred_lr']
    y_pred_dt = df['pred_dt']

    # Calibration curves
    calibration_plot(y_true, y_pred_lr, n_bins = 5)
    calibration_plot(y_true, y_pred_dt, n_bins = 5)

    print("Which model is more calibrated?")
    print("Answer: Inspect the plots. The model with points closer to the diagonal line is better calibrated.")

    # Extra Credit: PPV for top 50
    top50_lr = df.sort_values(by = 'pred_lr', ascending = False).head(50)
    top50_dt = df.sort_values(by = 'pred_dt', ascending = False).head(50)

    ppv_lr = top50_lr['y'].mean()
    ppv_dt = top50_dt['y'].mean()

    print(f"PPV for top 50 (Logistic Regression): {ppv_lr:.2f}")
    print(f"PPV for top 50 (Decision Tree): {ppv_dt:.2f}")

    # AUC Scores
    auc_lr = roc_auc_score(y_true, y_pred_lr)
    auc_dt = roc_auc_score(y_true, y_pred_dt)

    print(f"AUC (Logistic Regression): {auc_lr:.3f}")
    print(f"AUC (Decision Tree): {auc_dt:.3f}")

    print("Do both metrics agree that one model is more accurate than the other?")
    if (auc_lr > auc_dt and ppv_lr > ppv_dt):
        print("Yes, both AUC and PPV suggest Logistic Regression is more accurate.")
    elif (auc_lr < auc_dt and ppv_lr < ppv_dt):
        print("Yes, both AUC and PPV suggest Decision Tree is more accurate.")
    else:
        print("No, AUC and PPV do not agree. One model may have higher ranking precision, the other better ranking overall.")