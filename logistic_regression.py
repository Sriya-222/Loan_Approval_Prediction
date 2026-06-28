import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def safe_prepare(train_df, test_df):
    if 'Dependents' in train_df.columns:
        train_df['Dependents'] = train_df['Dependents'].replace({'3+': 3, '3 +': 3})
    if test_df is not None and 'Dependents' in test_df.columns:
        test_df['Dependents'] = test_df['Dependents'].replace({'3+': 3, '3 +': 3})

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
        s = pd.to_numeric(train_df[c], errors='coerce')
        med = s.median()
        if pd.isna(med):
            med = 0.0
        train_df[c] = s.fillna(med)
        medians[c] = med
        if test_df is not None and c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors='coerce').fillna(med)

    return train_df, test_df, feature_cols


def main():
    if not os.path.exists('train_cleaned.csv'):
        raise FileNotFoundError("train_cleaned.csv missing.")

    train = pd.read_csv('train_cleaned.csv')
    test = pd.read_csv('test_cleaned.csv') if os.path.exists('test_cleaned.csv') else None

    train_p, test_p, feature_cols = safe_prepare(train.copy(), test.copy() if test is not None else None)

    X_train = train_p[feature_cols].values.astype(float)
    y_train = train_p['Loan_Status'].astype(int).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train_scaled, y_train)

    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    train_preds = model.predict(X_train_scaled)

    out_train = train.copy()
    out_train['Logistic_Prob'] = train_probs
    out_train['Logistic_Predicted'] = ['Y' if p == 1 else 'N' for p in train_preds]
    out_train.to_csv('logistic_train_output.csv', index=False)
    print("Saved logistic_train_output.csv")

    if test_p is not None:
        X_test = test_p[feature_cols].values.astype(float)
        X_test_scaled = scaler.transform(X_test)

        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        test_preds = model.predict(X_test_scaled)

        out_test = test.copy()
        out_test['Logistic_Prob'] = test_probs
        out_test['Logistic_Predicted'] = ['Y' if p == 1 else 'N' for p in test_preds]
        out_test.to_csv('logistic_test_output.csv', index=False)
        print("Saved logistic_test_output.csv")
    else:
        print("No test_cleaned.csv found — skipped test predictions.")

if __name__ == "__main__":
    main()