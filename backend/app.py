import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import LoanApplicationRequest, LoanPredictionResponse, ChatRequest, ChatResponse, UserAuth, UserResponse
from backend.utils import calculate_loan_recommendation, generate_recommendations, query_ai_financial_assistant
from backend.database import db_signup, db_login, db_log_prediction, db_get_logs, db_get_all_logs

# Try importing shap for model explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = FastAPI(title="Credit Risk Intelligence API", version="1.0.0")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
MODELS = {}
MODEL_FILES = [
    "approval_model.pkl",
    "pd_model.pkl",
    "fraud_model.pkl",
    "interest_model.pkl",
    "segment_model.pkl",
    "scaler_segment.pkl",
    "feature_names.pkl",
    "X_train.pkl"
]

@app.on_event("startup")
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory {models_dir} does not exist yet. Please run pipeline.py first.")
        return
        
    for file in MODEL_FILES:
        filepath = os.path.join(models_dir, file)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    MODELS[file.replace(".pkl", "")] = pickle.load(f)
                print(f"Successfully loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"Warning: Model file {file} is missing.")

def map_categorical_inputs(req: LoanApplicationRequest):
    # Mapping logic identical to pipeline
    gender_map = {'Male': 1, 'Female': 0, 'Unknown': 1}
    married_map = {'Yes': 1, 'No': 0, 'Unknown': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0, 'Unknown': 0}
    self_employed_map = {'Yes': 1, 'No': 0, 'Unknown': 0}
    property_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2, 'Unknown': 1}
    
    # Dependents numeric conversion
    dep_val = 0
    if req.Dependents == "3+":
        dep_val = 3
    else:
        try:
            dep_val = int(req.Dependents)
        except ValueError:
            dep_val = 0
            
    total_income = req.ApplicantIncome + req.CoapplicantIncome
    loan_amount_log = np.log1p(req.LoanAmount)
    
    term_val = req.Loan_Amount_Term if req.Loan_Amount_Term > 0 else 360
    debt_to_income = (req.LoanAmount * 1000) / (total_income * term_val + 1)
    
    features = [
        gender_map.get(req.Gender, 1),
        married_map.get(req.Married, 0),
        dep_val,
        education_map.get(req.Education, 0),
        self_employed_map.get(req.Self_Employed, 0),
        req.ApplicantIncome,
        req.CoapplicantIncome,
        req.LoanAmount,
        req.Loan_Amount_Term,
        req.Credit_History,
        property_map.get(req.Property_Area, 1),
        total_income,
        loan_amount_log,
        debt_to_income
    ]
    return np.array(features).reshape(1, -1), total_income

@app.post("/predict", response_model=LoanPredictionResponse)
def predict_loan(req: LoanApplicationRequest):
    if not MODELS:
        # Try loading models dynamically in case server was started before training completed
        load_models()
        if not MODELS:
            raise HTTPException(status_code=503, detail="Models are not loaded. Please train the models first by running backend/pipeline.py.")
            
    try:
        # Preprocess features
        features_arr, total_income = map_categorical_inputs(req)
        feature_names = MODELS["feature_names"]
        
        # 1. Fraud Detection Check (Isolation Forest)
        fraud_model = MODELS["fraud_model"]
        # raw score ranges from ~0.3 (outlier) to ~0.8 (normal) in scikit-learn
        raw_score = fraud_model.score_samples(features_arr)[0]
        # map to 0-1 fraud score (1 is anomalous, 0 is normal)
        # IsolationForest score_samples returns negative values between -1 and 0 (lower is more anomalous)
        # Typically normal is > -0.45, anomalous is < -0.58
        fraud_score = float(np.clip((-0.45 - raw_score) / 0.13, 0.0, 1.0))
        fraud_detected = bool(fraud_score > 0.65)
        
        # 2. Probability of Default (PD) & Risk Score (Random Forest)
        pd_model = MODELS["pd_model"]
        default_prob = float(pd_model.predict_proba(features_arr)[0][1])
        
        # Adjust risk score upward if credit history is poor or fraud is flagged
        risk_score = default_prob * 100.0
        if req.Credit_History == 0.0:
            risk_score = max(risk_score, 85.0) # Bad credit history is heavily penalized in finance
        if fraud_detected:
            risk_score = max(risk_score, 90.0)
            
        risk_score = float(np.clip(risk_score, 0.0, 100.0))
        
        if risk_score < 30:
            risk_level = "Low Risk"
        elif risk_score <= 65:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        # 3. Loan Status Approval Model (Random Forest)
        approval_model = MODELS["approval_model"]
        approval_prediction = int(approval_model.predict(features_arr)[0])
        
        # Final decision logic combining ML model and business rules:
        # Reject if fraud is detected, credit history is bad (0.0), or risk score is too high.
        approved = bool(approval_prediction == 1 and not fraud_detected and risk_score < 65.0)
        
        # 4. Loan Amount Recommendation System
        recommended_loan = calculate_loan_recommendation(
            req.ApplicantIncome,
            req.CoapplicantIncome,
            req.LoanAmount,
            req.Loan_Amount_Term,
            req.Credit_History,
            req.Dependents
        )
        
        # 5. Dynamic Interest Rate Regressor
        interest_model = MODELS["interest_model"]
        predicted_interest = float(interest_model.predict(features_arr)[0])
        
        # Adjust interest rate based on risk level
        if risk_level == "Low Risk":
            predicted_interest = min(predicted_interest, 9.0)
        elif risk_level == "Medium Risk":
            predicted_interest = max(predicted_interest, 9.0)
        else:
            predicted_interest = max(predicted_interest, 14.0)
            
        predicted_interest = float(np.round(np.clip(predicted_interest, 6.5, 18.0), 2))
        
        # 6. Personalized Recommendations
        recommendations = generate_recommendations(
            req.dict(),
            risk_score,
            recommended_loan
        )
        
        # 7. Model Explainability (SHAP / Local Contribution Approximator)
        shap_contribs = {}
        use_shap = SHAP_AVAILABLE
        if use_shap:
            try:
                # We can construct tree explainer
                # To avoid re-initializing, we can store it or calculate on the fly (it's fast for single prediction)
                explainer = shap.TreeExplainer(approval_model)
                shap_values = explainer.shap_values(features_arr)
                
                # Check shape: shap_values can be [2, 1, 14] (list of two classes)
                if isinstance(shap_values, list):
                    # class 1 (Approved class contributions)
                    class_shap = shap_values[1][0]
                elif len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, 1]
                else:
                    class_shap = shap_values[0]
                    
                for name, val in zip(feature_names, class_shap):
                    shap_contribs[name] = float(val)
            except Exception as e:
                print(f"SHAP error: {e}. Falling back to rule-based explanation.")
                use_shap = False
                
        if not use_shap:
            # Fallback local feature contribution approximation:
            # Based on standard logic: Credit History is most important, followed by Income, LoanAmount, etc.
            # positive value pushes to approval, negative to rejection
            shap_contribs = {name: 0.0 for name in feature_names}
            shap_contribs["Credit_History"] = 0.45 if req.Credit_History == 1.0 else -0.55
            shap_contribs["ApplicantIncome"] = 0.15 if req.ApplicantIncome > 4000 else -0.10
            shap_contribs["LoanAmount"] = -0.15 if req.LoanAmount > recommended_loan else 0.10
            shap_contribs["Debt_to_Income"] = -0.20 if (req.LoanAmount / (total_income + 1)) > 0.03 else 0.05
            shap_contribs["CoapplicantIncome"] = 0.10 if req.CoapplicantIncome > 0 else 0.0
            
        # 8. Customer Segmentation
        segment_model = MODELS["segment_model"]
        scaler_segment = MODELS["scaler_segment"]
        seg_feats = np.array([[total_income, req.LoanAmount, req.Credit_History]])
        scaled_seg = scaler_segment.transform(seg_feats)
        cluster = int(segment_model.predict(scaled_seg)[0])
        
        # Map clusters dynamically to Human-readable Profiles based on Centroids
        centers = segment_model.cluster_centers_
        ch_centers = centers[:, 2]
        inc_centers = centers[:, 0]
        
        high_risk_idx = int(np.argmin(ch_centers))
        remaining_idxs = [i for i in range(3) if i != high_risk_idx]
        if inc_centers[remaining_idxs[0]] > inc_centers[remaining_idxs[1]]:
            premium_idx = remaining_idxs[0]
            regular_idx = remaining_idxs[1]
        else:
            premium_idx = remaining_idxs[1]
            regular_idx = remaining_idxs[0]
            
        cluster_names = {
            high_risk_idx: "High Risk / Subprime Customer",
            regular_idx: "Regular Customer",
            premium_idx: "Premium Customer"
        }
        segment_name = cluster_names.get(cluster, "Regular Customer")
            
        response_obj = LoanPredictionResponse(
            approved=approved,
            risk_score=risk_score,
            risk_level=risk_level,
            default_probability=default_prob,
            fraud_score=fraud_score,
            fraud_detected=fraud_detected,
            recommended_loan_amount=recommended_loan,
            predicted_interest_rate=predicted_interest,
            recommendations=recommendations,
            shap_values=shap_contribs,
            segment_cluster=cluster,
            segment_name=segment_name
        )
        
        # Log prediction to DB if user_id is provided
        if req.user_id:
            inputs_dict = {k: v for k, v in req.dict().items() if k != "user_id"}
            outputs_dict = response_obj.dict()
            db_log_prediction(req.user_id, inputs_dict, outputs_dict)
            
        return response_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def chat_assistant(req: ChatRequest):
    app_dict = req.application_data.dict() if req.application_data else {}
    pred_dict = req.prediction_result or {}
    
    response_text = query_ai_financial_assistant(
        req.message,
        app_dict,
        pred_dict,
        req.chat_history or []
    )
    return ChatResponse(response=response_text)

@app.get("/runs")
def get_mlops_runs():
    logs_file = os.path.join(os.path.dirname(__file__), "experiment_history.json")
    if not os.path.exists(logs_file):
        return []
    try:
        with open(logs_file, "r") as f:
            return json.load(f)
    except Exception:
        return []

@app.post("/auth/signup", response_model=UserResponse)
def signup(auth: UserAuth):
    res = db_signup(auth.username, auth.password)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return UserResponse(**res)

@app.post("/auth/login", response_model=UserResponse)
def login(auth: UserAuth):
    res = db_login(auth.username, auth.password)
    if not res["success"]:
        raise HTTPException(status_code=401, detail=res["message"])
    return UserResponse(**res)

@app.get("/logs/{user_id}")
def get_user_logs(user_id: str):
    return db_get_logs(user_id)

@app.get("/logs")
def get_all_logs():
    return db_get_all_logs()
