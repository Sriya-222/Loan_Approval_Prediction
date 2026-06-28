import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def fill_mode(df, column):
    freq = df[column].value_counts().idxmax()
    df[column] = df[column].fillna(freq)

def fill_median(df, column):
    sorted_vals = sorted(df[column].dropna())
    n = len(sorted_vals)
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]
    df[column] = df[column].fillna(median)

cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols:
    fill_mode(train, col)
    fill_mode(test, col)

num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    fill_median(train, col)
    fill_median(test, col)


train['Gender'] = train['Gender'].replace({'Male': 1, 'Female': 0})
test['Gender'] = test['Gender'].replace({'Male': 1, 'Female': 0})

train['Married'] = train['Married'].replace({'Yes': 1, 'No': 0})
test['Married'] = test['Married'].replace({'Yes': 1, 'No': 0})

train['Education'] = train['Education'].replace({'Graduate': 1, 'Not Graduate': 0})
test['Education'] = test['Education'].replace({'Graduate': 1, 'Not Graduate': 0})

train['Self_Employed'] = train['Self_Employed'].replace({'Yes': 1, 'No': 0})
test['Self_Employed'] = test['Self_Employed'].replace({'Yes': 1, 'No': 0})

train['Property_Area'] = train['Property_Area'].replace({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
test['Property_Area'] = test['Property_Area'].replace({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

train['Loan_Status'] = train['Loan_Status'].replace({'Y': 1, 'N': 0})

train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']

import math
def log_transform(val):
    if val <= 0:
        return 0
    else:
        return math.log(val)

train['LoanAmount_log'] = [log_transform(x) for x in train['LoanAmount']]
test['LoanAmount_log'] = [log_transform(x) for x in test['LoanAmount']]

train.drop(['Loan_ID'], axis=1, inplace=True)
test.drop(['Loan_ID'], axis=1, inplace=True)

train.to_csv("train_cleaned.csv", index=False)
test.to_csv("test_cleaned.csv", index=False)
