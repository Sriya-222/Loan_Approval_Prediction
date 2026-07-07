from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class LoanApplicationRequest(BaseModel):
    Gender: str = Field(..., example="Male")
    Married: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="0") # Keep as string because dataset contains "3+"
    Education: str = Field(..., example="Graduate")
    Self_Employed: str = Field(..., example="No")
    ApplicantIncome: float = Field(..., example=5000.0)
    CoapplicantIncome: float = Field(..., example=0.0)
    LoanAmount: float = Field(..., example=150.0) # in thousands, e.g. 150000
    Loan_Amount_Term: float = Field(..., example=360.0) # in days or months, standard is 360
    Credit_History: float = Field(..., example=1.0) # 1.0 = Good, 0.0 = Poor
    Property_Area: str = Field(..., example="Urban")
    user_id: Optional[str] = Field(default=None, example="user123")

class UserAuth(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="johndoe")
    password: str = Field(..., min_length=6, example="password123")

class UserResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    username: Optional[str] = None

class LoanPredictionResponse(BaseModel):
    approved: bool
    risk_score: float
    risk_level: str
    default_probability: float
    fraud_score: float
    fraud_detected: bool
    recommended_loan_amount: float
    predicted_interest_rate: float
    recommendations: List[str]
    shap_values: Dict[str, float]
    segment_cluster: int
    segment_name: str

class ChatRequest(BaseModel):
    message: str
    application_data: Optional[LoanApplicationRequest] = None
    prediction_result: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
