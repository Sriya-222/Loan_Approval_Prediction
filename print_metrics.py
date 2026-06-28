import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

CANDIDATES = {
    'Naive Bayes': ['trained_naive_bayes.csv', 'trained_naivebayes.csv', 'nb_trained.csv'],
    'Decision Tree': ['trained_decision_tree.csv', 'trained_decisiontree.csv', 'dt_trained.csv'],
    'Logistic Regression': ['logistic_train_output.csv', 'logistic_train.csv', 'trained_logistic.csv'],
    'Random Forest': ['comparison_with_rf.csv', 'trained_random_forest.csv', 'rf_trained.csv', 'comparison_with_rf.csv']
}

PRED_COLS = ['Loan_Status_Predicted', 'NB_Predicted', 'DT_Predicted',
             'Logistic_Predicted', 'RF_Predicted', 'Prediction', 'pred', 'Loan_Status']

PROB_KEYWORDS = ['prob', 'proba', 'score', 'confidence']

def find_file(patterns):
    for p in patterns:
        if os.path.exists(p):
            return p
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None

def read_df(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

def detect_pred_column(df):
    for c in PRED_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.strip().str.upper().unique()[:10]
        if len(vals) == 0:
            continue
        allowed = set(['Y','N','YES','NO','TRUE','FALSE','1','0','1.0','0.0','T','F'])
        if set(vals).issubset(allowed):
            return c
    return None

def detect_prob_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in PROB_KEYWORDS) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            vals = df[c].dropna()
            if len(vals) == 0:
                continue
            if vals.between(0,1).all():
                return c
    return None

def normalize_pred_series(s):
    s2 = s.fillna('').astype(str).str.strip().str.upper()
    def conv(x):
        if x in ('Y','YES','T','TRUE'): return 1
        if x in ('N','NO','F','FALSE'): return 0
        try:
            fv = float(x)
            return 1 if fv >= 0.5 else 0
        except:
            return 0
    return s2.map(conv).astype(int).to_numpy()

def align_predictions_to_train(model_df, pred_col, train_df):
    prob_col = detect_prob_column(model_df)
    prob_scores = None

    if 'Loan_ID' in model_df.columns and 'Loan_ID' in train_df.columns:
        m_idx = model_df.set_index('Loan_ID')
        t_idx = train_df.set_index('Loan_ID')
        joined = t_idx.join(m_idx[[pred_col]] if pred_col in m_idx.columns else pd.DataFrame(), how='left')
        y_pred = normalize_pred_series(joined[pred_col].fillna(0))
        if prob_col and prob_col in m_idx.columns:
            joined_prob = t_idx.join(m_idx[[prob_col]], how='left')
            prob_scores = joined_prob[prob_col].fillna(0).to_numpy()
        y_true = t_idx['Loan_Status'].to_numpy()
        return y_true, y_pred, prob_scores

    preds = model_df[pred_col].reset_index(drop=True).tolist()
    n = len(train_df)
    if len(preds) < n:
        preds = preds + [0] * (n - len(preds))
    else:
        preds = preds[:n]
    y_pred = normalize_pred_series(pd.Series(preds))
    if prob_col:
        probs = model_df[prob_col].reset_index(drop=True).tolist()
        if len(probs) < n:
            probs = probs + [0] * (n - len(probs))
        else:
            probs = probs[:n]
        prob_scores = np.array(probs)
    y_true = train_df['Loan_Status'].to_numpy()
    return y_true, y_pred, prob_scores

def print_metrics_block(model_name, y_true, y_pred, prob_scores=None):
    print("\n" + "="*60)
    print(f"MODEL: {model_name} (TRAIN)")
    print("-"*60)
    if y_true is None:
        print("No ground-truth available in train_cleaned.csv (expected Loan_Status).")
        return
    def to_bin(arr):
        out = []
        for v in arr:
            if pd.isna(v):
                out.append(0)
                continue
            sv = str(v).strip().upper()
            if sv in ('1','1.0','Y','YES','TRUE','T'):
                out.append(1)
            elif sv in ('0','0.0','N','NO','FALSE','F'):
                out.append(0)
            else:
                try:
                    fv = float(sv)
                    out.append(1 if fv >= 0.5 else 0)
                except:
                    out.append(0)
        return np.array(out, dtype=int)

    yt = to_bin(y_true)
    yp = np.array(y_pred, dtype=int)

    if len(yt) != len(yp):
        print(f"[ERROR] Length mismatch: y_true={len(yt)} vs y_pred={len(yp)}. Skipping.")
        return

    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0,1])

    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    if prob_scores is not None:
        try:
            auc = roc_auc_score(yt, prob_scores)
            print(f"ROC AUC  : {auc:.4f}")
        except Exception as e:
            print("ROC AUC  : could not compute (reason):", e)
    else:
        print("ROC AUC  : Not available (no probability/score column detected).")

    print("MAE / MSE / RMSE : Not applicable for classification (N/A)")
    print("="*60 + "\n")


def main():
    if not os.path.exists('train_cleaned.csv'):
        print("[ERROR] train_cleaned.csv not found in current folder. Place it here and re-run.")
        return
    train_df = pd.read_csv('train_cleaned.csv')
    if 'Loan_Status' not in train_df.columns:
        print("[ERROR] train_cleaned.csv does not contain 'Loan_Status' column. Can't compute train metrics.")
        return

    any_found = False
    for model_name, pats in CANDIDATES.items():
        path = find_file(pats)
        if path is None:
            print(f"[INFO] {model_name} train file not found. Skipping.")
            continue
        any_found = True
        df_model = read_df(path)
        if df_model is None:
            print(f"[WARN] Could not read {path}. Skipping {model_name}.")
            continue
        pred_col = detect_pred_column(df_model)
        if pred_col is None:
            print(f"[WARN] Could not detect prediction column in {path}. Columns: {df_model.columns.tolist()}")
            print(f"[WARN] Skipping {model_name}.")
            continue
        y_true, y_pred, prob_scores = align_predictions_to_train(df_model, pred_col, train_df)
        print_metrics_block(model_name, y_true, y_pred, prob_scores)

    if not any_found:
        print("[WARN] No model train files found. Place model outputs in folder and try again.")

if __name__ == "__main__":
    main()