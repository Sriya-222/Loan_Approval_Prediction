import os
import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
except Exception as e:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

def safe_prepare(train_df, test_df=None):
    for df in (train_df, test_df):
        if df is None:
            continue
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace({'3+': 3, '3 +': 3})
    maps = {
        'Gender': {'Male':1,'Female':0},
        'Married': {'Yes':1,'No':0},
        'Education': {'Graduate':1,'Not Graduate':0},
        'Self_Employed': {'Yes':1,'No':0},
        'Property_Area': {'Rural':0,'Semiurban':1,'Urban':2},
        'Loan_Status': {'Y':1,'N':0}
    }
    for col, mp in maps.items():
        if col in train_df.columns:
            train_df[col] = train_df[col].map(mp).where(train_df[col].isin(mp.keys()), train_df[col])
        if test_df is not None and col in test_df.columns:
            test_df[col] = test_df[col].map(mp).where(test_df[col].isin(mp.keys()), test_df[col])
    feature_cols = [c for c in train_df.columns if c not in ('Loan_ID','Loan_Status')]
    for c in feature_cols:
        col_ser = pd.to_numeric(train_df[c], errors='coerce')
        med = col_ser.median()
        if pd.isna(med):
            med = 0.0
        train_df[c] = col_ser.fillna(med)
        if test_df is not None and c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors='coerce').fillna(med)
    return train_df, test_df, feature_cols

def main():
    train_path = "train_cleaned.csv"
    test_path = "test_cleaned.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError("train_cleaned.csv not found")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    train_p, test_p, features = safe_prepare(train.copy(), test.copy() if test is not None else None)

    if 'Loan_Status' not in train_p.columns:
        raise KeyError("Loan_Status not found in train_cleaned.csv")

    X_train = train_p[features].astype(float).values
    y_train = train_p['Loan_Status'].astype(int).values

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=8, random_state=42)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    acc = accuracy_score(y_train, train_pred)
    print(f"Decision Tree training accuracy: {acc*100:.2f}%")

    out_train = train.copy()
    out_train['Loan_Status_Predicted'] = ['Y' if v==1 else 'N' for v in train_pred]
    out_train.to_csv("trained_decision_tree.csv", index=False)
    print("Saved trained_decision_tree.csv")

    if test_p is not None:
        for c in features:
            if c not in test_p.columns:
                test_p[c] = 0.0
        X_test = test_p[features].astype(float).values
        test_pred = clf.predict(X_test)
        out_test = test.copy()
        out_test['Loan_Status_Predicted'] = ['Y' if v==1 else 'N' for v in test_pred]
        out_test.to_csv("test_decision_tree.csv", index=False)
        print("Saved test_decision_tree.csv")
    else:
        print("No test_cleaned.csv found")

if __name__ == "__main__":
    main()