import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def safe_prepare(train_df, test_df):
    if 'Dependents' in train_df.columns:
        train_df['Dependents'] = train_df['Dependents'].replace({'3+':3, '3 +':3})
    if test_df is not None and 'Dependents' in test_df.columns:
        test_df['Dependents'] = test_df['Dependents'].replace({'3+':3, '3 +':3})

    maps = [
        ('Gender', {'Male':1,'Female':0}),
        ('Married', {'Yes':1,'No':0}),
        ('Education', {'Graduate':1,'Not Graduate':0}),
        ('Self_Employed', {'Yes':1,'No':0}),
        ('Property_Area', {'Rural':0,'Semiurban':1,'Urban':2}),
    ]

    for col, mp in maps:
        if col in train_df.columns:
            train_df[col] = train_df[col].map(mp).where(train_df[col].isin(mp.keys()), train_df[col])
        if test_df is not None and col in test_df.columns:
            test_df[col] = test_df[col].map(mp).where(test_df[col].isin(mp.keys()), test_df[col])

    feature_cols = [c for c in train_df.columns if c not in ('Loan_ID','Loan_Status')]
    medians = {}

    for c in feature_cols:
        series = pd.to_numeric(train_df[c], errors='coerce')
        median = series.median() if not pd.isna(series.median()) else 0.0

        train_df[c] = series.fillna(median)
        medians[c] = median

        if test_df is not None and c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors='coerce').fillna(median)

    return train_df, test_df, feature_cols

def main():
    if not os.path.exists('train_cleaned.csv'):
        raise FileNotFoundError("train_cleaned.csv missing.")

    train = pd.read_csv('train_cleaned.csv')
    test = pd.read_csv('test_cleaned.csv') if os.path.exists('test_cleaned.csv') else None

    train_p, test_p, feature_cols = safe_prepare(train.copy(), test.copy() if test is not None else None)

    X = train_p[feature_cols].values.astype(float)
    y = train_p['Loan_Status'].astype(int).values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=22, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_probs = rf.predict_proba(X)[:,1]
    train_preds = (train_probs >= 0.5).astype(int)

    out_train = train.copy()
    out_train['RF_Prob'] = train_probs
    out_train['RF_Predicted'] = ['Y' if p == 1 else 'N' for p in train_preds]
    out_train.to_csv('comparison_with_rf.csv', index=False)

    print("Saved comparison_with_rf.csv")
    if test_p is not None:
        X_test = test_p[feature_cols].values.astype(float)
        test_probs = rf.predict_proba(X_test)[:,1]
        test_preds = (test_probs >= 0.5).astype(int)

        out_test = test.copy()
        out_test['RF_Prob'] = test_probs
        out_test['RF_Predicted'] = ['Y' if p == 1 else 'N' for p in test_preds]
        out_test.to_csv('comparison_test_with_rf.csv', index=False)

        print("Saved comparison_test_with_rf.csv")
    else:
        print("No test_cleaned.csv found — skipped RF test predictions.")

if __name__ == "__main__":
    main()