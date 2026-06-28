import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import shorten

folder = "preprocessing_plots"
os.makedirs(folder, exist_ok=True)

df_path = "train_cleaned.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"{df_path} not found in current directory.")
df = pd.read_csv(df_path)

print("\nLoaded", df_path)
print(df.head())

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap (After Cleaning)")
plt.tight_layout()
plt.savefig(f"{folder}/missing_values_heatmap.png", dpi=300)
plt.close()

df.isnull().sum().to_csv(f"{folder}/missing_values_table.csv")
print("Saved missing values heatmap & table")

num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if len(num_cols) > 0:
    df[num_cols].hist(figsize=(15,10), bins=20)
    plt.suptitle("Histograms of Numerical Features", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{folder}/histograms.png", dpi=300)
    plt.close()
    print("Saved histogram plot")
else:
    print("No numeric columns found to plot histograms.")

boxplot_cols = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount","TotalIncome"] if c in df.columns]
if len(boxplot_cols) > 0:
    plt.figure(figsize=(15,5 * ((len(boxplot_cols)+1)//2)))
    for i, col in enumerate(boxplot_cols, 1):
        plt.subplot((len(boxplot_cols)+1)//2, 2, i)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(f"{folder}/boxplots.png", dpi=300)
    plt.close()
    print("Saved boxplot image")
else:
    print("No requested columns for boxplots found.")

if len(num_cols) > 1:
    plt.figure(figsize=(12,10))
    sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(f"{folder}/correlation_heatmap.png", dpi=300)
    plt.close()
    print("Saved correlation heatmap")
else:
    print("Not enough numeric columns for correlation heatmap.")

cat_cols = [c for c in ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"] if c in df.columns]
combined_preview = []
for col in cat_cols:
    vc = df[col].fillna("NA").astype(str).value_counts(dropna=False)
    vc_df = vc.reset_index()
    vc_df.columns = [col, "count"]
    fname = f"{folder}/categorical_{col}_value_counts.csv"
    vc_df.to_csv(fname, index=False)
    print(f"Saved value counts for {col} -> {fname}")

    unique_vals = df[col].dropna().astype(str).unique().tolist()
    short_uniques = ", ".join([str(x) for x in unique_vals[:6]])
    short_uniques = shorten(short_uniques, width=120, placeholder="...")
    combined_preview.append({"column": col, "sample_uniques": short_uniques})

if combined_preview:
    pd.DataFrame(combined_preview).to_csv(f"{folder}/categorical_encoding_preview.csv", index=False)
    print("Saved combined categorical preview CSV")
else:
    print("No categorical columns found for encoding preview.")

demo_cols = [c for c in ["ApplicantIncome","CoapplicantIncome","TotalIncome","LoanAmount","LoanAmount_log"] if c in df.columns]
if len(demo_cols) > 0:
    df[demo_cols].head(10).to_csv(f"{folder}/feature_engineering_preview.csv", index=False)
    print("Saved feature engineering preview")
else:
    print("No feature engineering columns found to preview.")

for col in cat_cols:
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x=df[col].astype(str))
    ax.set_title(f"Countplot: {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{folder}/cat_count_{col}.png", dpi=300)
    plt.close()
    print(f"Saved categorical countplot for {col}")

print("\nAll preprocessing visuals and CSVs created in folder:", folder)