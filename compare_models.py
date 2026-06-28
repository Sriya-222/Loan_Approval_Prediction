import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TRAIN_FILES = {
    'naive_bayes': 'trained_naive_bayes.csv',
    'decision_tree': 'trained_decision_tree.csv',
    'logistic': 'logistic_train_output.csv',
    'random_forest': 'comparison_with_rf.csv'
}

TEST_FILES = {
    'naive_bayes': 'test_naive_bayes.csv',
    'decision_tree': 'test_decision_tree.csv',
    'logistic': 'logistic_test_output.csv',
    'random_forest': 'comparison_test_with_rf.csv'
}

PRED_COL_CANDIDATES = [
    'Loan_Status_Predicted', 'NB_Predicted', 'DT_Predicted',
    'Logistic_Predicted', 'RF_Predicted', 'Loan_Status', 'Prediction', 'pred', 'Predicted'
]

def read_csv_try(path):
    try:
        return pd.read_csv(path)
    except:
        return None

def detect_prediction_column(df):
    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        sample = df[c].dropna().astype(str).map(lambda x: x.strip().upper()).unique()[:10]
        if len(sample) == 0:
            continue
        if set(sample).issubset({'Y','N','YES','NO','TRUE','FALSE','1','0','1.0','0.0','T','F'}):
            return c
    return None

def normalize_pred_series(s):
    def conv(v):
        if pd.isna(v):
            return None
        sv = str(v).strip().upper()
        if sv in ('Y','YES','T','TRUE'):
            return 1
        if sv in ('N','NO','F','FALSE'):
            return 0
        try:
            fv = float(sv)
            return 1 if fv >= 0.5 else 0
        except:
            return None
    return s.map(conv)

def ensure_mergeable_index(df, prefer_key='Loan_ID'):
    if prefer_key in df.columns:
        return df.set_index(prefer_key), prefer_key, False
    temp = df.copy()
    temp['row_index'] = np.arange(len(temp))
    return temp.set_index('row_index'), 'row_index', True

def compute_metrics_safe(y_true, y_pred):
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

    y_true_arr = to_bin(y_true)
    y_pred_arr = to_bin(y_pred)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    support = int(np.sum(y_true_arr == 1))
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'support': support, 'confusion': cm}

def main():
    if not os.path.exists('train_cleaned.csv'):
        print("train_cleaned.csv not found")
        return
    train_clean = pd.read_csv('train_cleaned.csv')
    test_clean = pd.read_csv('test_cleaned.csv') if os.path.exists('test_cleaned.csv') else None

    train_idxed, _, _ = ensure_mergeable_index(train_clean)
    if test_clean is not None:
        test_idxed, _, _ = ensure_mergeable_index(test_clean)
    else:
        test_idxed = None

    found_train = {m: f for m, f in TRAIN_FILES.items() if os.path.exists(f)}
    found_test = {m: f for m, f in TEST_FILES.items() if os.path.exists(f)}

    print("Found train:", found_train)
    print("Found test:", found_test)

    train_preds = {}
    test_preds = {}

    def align_positional(series_pos, target_index):
        arr = normalize_pred_series(series_pos.reset_index(drop=True))
        arr_list = arr.tolist()
        if len(arr_list) < len(target_index):
            arr_list += [0] * (len(target_index) - len(arr_list))
        else:
            arr_list = arr_list[:len(target_index)]
        return pd.Series(arr_list, index=target_index).astype(int)

    for model, path in found_train.items():
        df = read_csv_try(path)
        if df is None: 
            continue
        pred_col = detect_prediction_column(df)
        if pred_col is None:
            continue
        s = align_positional(df[pred_col], train_idxed.index)
        train_preds[model] = s

    for model, path in found_test.items():
        df = read_csv_try(path)
        if df is None:
            continue
        pred_col = detect_prediction_column(df)
        if pred_col is None:
            continue
        if test_idxed is not None:
            s = align_positional(df[pred_col], test_idxed.index)
        else:
            s = align_positional(df[pred_col], pd.RangeIndex(len(df)))
        test_preds[model] = s

    comp_train = train_idxed.copy()
    for model, series in train_preds.items():
        comp_train[f'{model}_pred'] = series.values
    comp_train['Actual_bin'] = comp_train['Loan_Status'].map(lambda x: 1 if str(x).strip().upper() in ['Y','1','YES','TRUE'] else 0)
    comp_train_reset = comp_train.reset_index()
    comp_train_reset.to_csv('comparison_all_train.csv', index=False)

    rows = []
    for model, series in train_preds.items():
        y_true = comp_train['Actual_bin'].to_numpy()
        y_pred = comp_train[f'{model}_pred'].to_numpy()
        m = compute_metrics_safe(y_true, y_pred)
        rows.append({'model': model, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'], 'support': m['support']})
    dfm = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    dfm.to_csv('metrics_summary_train.csv', index=False)
    print(dfm)

    if not dfm['accuracy'].isna().all():
        plt.figure(figsize=(7,4))
        names = dfm['model'].tolist()
        accs = dfm['accuracy'].fillna(0).tolist()
        plt.bar(names, [a*100 for a in accs])
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Comparison')
        plt.ylim(0,100)
        plt.savefig('accuracy_bar_train.png')
        plt.close()

    for model in train_preds.keys():
        y_true = comp_train['Actual_bin'].to_numpy()
        y_pred = comp_train[f'{model}_pred'].to_numpy()
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, cmap='Blues')
        plt.title(f'Confusion Matrix - {model}')
        plt.xticks([0,1],['Pred N','Pred Y'])
        plt.yticks([0,1],['Act N','Act Y'])
        for (i,j), val in np.ndenumerate(cm):
            plt.text(j,i,int(val),ha='center',va='center')
        plt.savefig(f'confusion_{model}_train.png')
        plt.close()

    if len(test_preds) > 0:
        if test_idxed is not None:
            comp_test = test_idxed.copy()
        else:
            first_model = list(test_preds.keys())[0]
            comp_test = pd.DataFrame(index=test_preds[first_model].index)

        for model, series in test_preds.items():
            comp_test[f'{model}_pred'] = series.values

        comp_test_reset = comp_test.reset_index()
        comp_test_reset.to_csv('comparison_all_test.csv', index=False)

        if test_idxed is not None and 'Loan_Status' in test_idxed.columns:
            actual_test_bin = test_idxed['Loan_Status'].map(lambda x: 1 if str(x).strip().upper() in ['Y','1','YES','TRUE'] else 0).to_numpy()
            rows = []
            for model, series in test_preds.items():
                y_pred = series.to_numpy()
                m = compute_metrics_safe(actual_test_bin, y_pred)
                rows.append({'model': model, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'], 'support': m['support']})
            dfmt = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
            dfmt.to_csv('metrics_summary_test.csv', index=False)
            print(dfmt)

            for model in test_preds.keys():
                y_pred = test_preds[model].to_numpy()
                cm = confusion_matrix(actual_test_bin, y_pred)
                plt.imshow(cm, cmap='Blues')
                plt.title(f'Confusion Matrix - {model} (test)')
                plt.xticks([0,1],['Pred N','Pred Y'])
                plt.yticks([0,1],['Act N','Act Y'])
                for (i,j), val in np.ndenumerate(cm):
                    plt.text(j,i,int(val),ha='center',va='center')
                plt.savefig(f'confusion_{model}_test.png')
                plt.close()

    print("Done.")

if __name__ == "__main__":
    main()