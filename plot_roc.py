import os
import pandas as pd
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def plot_roc_sklearn(y_true, probs, model_name, split):
    plt.figure(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        probs,
        name=f"{model_name} ({split})"
    )

    fname = f"roc_{model_name.lower().replace(' ','_')}_{split}.png"
    plt.title(f"ROC Curve - {model_name} ({split})")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {fname}")

def try_plot(file_path, prob_col, model_name):
    if not os.path.exists(file_path):
        print(f"{file_path} not found; skipping.")
        return

    df = pd.read_csv(file_path)

    if 'Loan_Status' not in df.columns:
        print(f"{file_path} has no Loan_Status column; skipping.")
        return

    if prob_col not in df.columns:
        print(f"{prob_col} missing in {file_path}; skipping.")
        return

    y_true = df['Loan_Status'].map(
        lambda x: 1 if str(x).strip().upper() in ['Y', 'YES', '1', 'TRUE'] else 0
    )

    probs = df[prob_col].astype(float).fillna(0)

    if y_true.nunique() < 2:
        print(f"Skipping {file_path}: contains only one class.")
        return

    split = 'train' if 'train' in file_path.lower() else 'test'

    plot_roc_sklearn(y_true, probs, model_name, split)

def main():

    try_plot('logistic_train_output.csv', 'Logistic_Prob', 'Logistic')
    try_plot('logistic_test_output.csv', 'Logistic_Prob', 'Logistic')

    try_plot('comparison_with_rf.csv', 'RF_Prob', 'RandomForest')
    try_plot('comparison_test_with_rf.csv', 'RF_Prob', 'RandomForest')

if __name__ == "__main__":
    main()