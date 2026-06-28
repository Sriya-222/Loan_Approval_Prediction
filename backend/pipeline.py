import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, silhouette_score

# Try to import mlflow, if unavailable we fall back to local logging
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Ensure model directory exists
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Helper function to log locally
def log_experiment_local(params, metrics, run_name):
    logs_file = os.path.join(os.path.dirname(__file__), "experiment_history.json")
    history = []
    if os.path.exists(logs_file):
        try:
            with open(logs_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append({
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "metrics": metrics
    })
    
    with open(logs_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Logged run '{run_name}' to local history.")

def prepare_data():
    train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train.csv")
    if not os.path.exists(train_path):
        train_path = "train.csv"
        
    df = pd.read_csv(train_path)
    
    # Handle dependents
    df['Dependents'] = df['Dependents'].replace({'3+': 3, '3 +': 3})
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
    df['Dependents'] = df['Dependents'].fillna(0)
    
    # Fill categorical missing values with mode
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
    # Fill numerical missing values with median
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
        
    # Feature Engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
    df['Debt_to_Income'] = (df['LoanAmount'] * 1000) / (df['TotalIncome'] * df['Loan_Amount_Term'].replace(0, 360) + 1)
    
    # Maps for categorical encoding
    gender_map = {'Male': 1, 'Female': 0, 'Unknown': 1}
    married_map = {'Yes': 1, 'No': 0, 'Unknown': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0, 'Unknown': 0}
    self_employed_map = {'Yes': 1, 'No': 0, 'Unknown': 0}
    property_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2, 'Unknown': 1}
    
    df['Gender'] = df['Gender'].map(gender_map)
    df['Married'] = df['Married'].map(married_map)
    df['Education'] = df['Education'].map(education_map)
    df['Self_Employed'] = df['Self_Employed'].map(self_employed_map)
    df['Property_Area'] = df['Property_Area'].map(property_map)
    
    # Targets
    df['Loan_Status_encoded'] = df['Loan_Status'].replace({'Y': 1, 'N': 0}).astype(int)
    # Default is the opposite of approval in this historical dataset
    df['Default'] = 1 - df['Loan_Status_encoded']
    
    # Synthesize a realistic interest rate column based on risk factors (for the regression model)
    # Higher rate for poor credit history, high loan amount, high risk.
    np.random.seed(42)
    base_rate = 8.0
    risk_premium = 5.0 * (1 - df['Credit_History']) + 4.0 * (df['LoanAmount'] / (df['TotalIncome'] / 10 + 1))
    noise = np.random.normal(0, 0.2, len(df))
    df['Interest_Rate'] = base_rate + risk_premium + noise
    df['Interest_Rate'] = df['Interest_Rate'].clip(6.5, 18.0) # bound realistic interest rates
    
    return df

def train_models():
    df = prepare_data()
    
    # Define features
    feature_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'TotalIncome', 'LoanAmount_log', 'Debt_to_Income'
    ]
    
    X = df[feature_cols].values
    y_approval = df['Loan_Status_encoded'].values
    y_default = df['Default'].values
    y_interest = df['Interest_Rate'].values
    
    # Train test splits
    X_train, X_test, y_train_app, y_test_app = train_test_split(X, y_approval, test_size=0.2, random_state=42, stratify=y_approval)
    _, _, y_train_def, y_test_def = train_test_split(X, y_default, test_size=0.2, random_state=42, stratify=y_default)
    _, _, y_train_int, y_test_int = train_test_split(X, y_interest, test_size=0.2, random_state=42)
    
    print("Training models...")
    
    # 1. Approval Model
    clf_app = RandomForestClassifier(n_estimators=100, random_state=22, n_jobs=-1)
    clf_app.fit(X_train, y_train_app)
    app_preds = clf_app.predict(X_test)
    app_acc = accuracy_score(y_test_app, app_preds)
    
    # 2. Probability of Default (PD) Model
    clf_def = RandomForestClassifier(n_estimators=100, random_state=22, n_jobs=-1)
    clf_def.fit(X_train, y_train_def)
    def_probs = clf_def.predict_proba(X_test)[:, 1]
    def_auc = roc_auc_score(y_test_def, def_probs)
    
    # 3. Fraud Detection Module (Isolation Forest)
    # Fit on training data. Output is normal (1) or anomaly (-1).
    # Map score to a probability scale: higher score means higher anomaly likelihood
    detector_fraud = IsolationForest(contamination=0.05, random_state=42)
    detector_fraud.fit(X_train)
    
    # 4. Interest Rate Regressor
    reg_int = RandomForestRegressor(n_estimators=100, random_state=22, n_jobs=-1)
    reg_int.fit(X_train, y_train_int)
    int_preds = reg_int.predict(X_test)
    int_rmse = np.sqrt(mean_squared_error(y_test_int, int_preds))
    
    # 5. Customer Segmentation (Unsupervised KMeans Clustering)
    # We segment based on: TotalIncome, LoanAmount, and Credit_History
    segment_features = df[['TotalIncome', 'LoanAmount', 'Credit_History']].copy()
    scaler_segment = StandardScaler()
    scaled_segments = scaler_segment.fit_transform(segment_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_segments)
    
    sil_score = silhouette_score(scaled_segments, clusters)
    
    # Save all models
    models = {
        "approval_model.pkl": clf_app,
        "pd_model.pkl": clf_def,
        "fraud_model.pkl": detector_fraud,
        "interest_model.pkl": reg_int,
        "segment_model.pkl": kmeans,
        "scaler_segment.pkl": scaler_segment,
        "feature_names.pkl": feature_cols,
        "X_train.pkl": X_train  # saved for SHAP background explainer
    }
    
    for filename, model_obj in models.items():
        filepath = os.path.join(MODELS_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(model_obj, f)
    print("Saved all 5 models and scales to disk.")
    
    # Logs and metrics
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    params = {
        "rf_estimators": 100,
        "kmeans_clusters": 3,
        "fraud_contamination": 0.05
    }
    metrics = {
        "approval_accuracy": float(app_acc),
        "default_auc": float(def_auc),
        "interest_rate_rmse": float(int_rmse),
        "segmentation_silhouette": float(sil_score)
    }
    
    print(f"Metrics: {metrics}")
    
    # MLflow tracking
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("Loan_Intelligence_System")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                # Save artifacts to mlflow
                mlflow.sklearn.log_model(clf_app, "approval_model")
                mlflow.sklearn.log_model(clf_def, "pd_model")
                mlflow.sklearn.log_model(reg_int, "interest_model")
                print("MLflow run logged successfully.")
        except Exception as e:
            print(f"Failed to log run to MLflow: {e}")
            
    # Always log locally as fallback/record
    log_experiment_local(params, metrics, run_name)

if __name__ == "__main__":
    train_models()
