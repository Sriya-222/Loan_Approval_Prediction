import os
import google.generativeai as genai
from typing import List, Dict, Any

# Configure Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CONFIGURED = False

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        GEMINI_CONFIGURED = True
    except Exception as e:
        print(f"Warning: Failed to configure Gemini: {e}")

def calculate_loan_recommendation(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    term: float,
    credit_history: float,
    dependents: str
) -> float:
    """
    Recommends a safe maximum loan amount based on income, credit history, and dependents.
    """
    total_income = applicant_income + coapplicant_income
    monthly_income = total_income / 12.0
    
    # Base EMI capacity: 40% of income if credit history is good, 15% if bad
    capacity_ratio = 0.40 if credit_history == 1.0 else 0.15
    
    # Adjust for dependents: subtract 5% for each dependent, up to 15% max
    dep_count = 0
    if dependents == "3+":
        dep_count = 3
    else:
        try:
            dep_count = int(dependents)
        except ValueError:
            dep_count = 0
            
    capacity_ratio -= min(0.05 * dep_count, 0.15)
    capacity_ratio = max(0.10, capacity_ratio) # Ensure at least 10% capacity
    
    # Max allowed monthly EMI
    max_emi = monthly_income * capacity_ratio
    
    # Recommended loan amount = max_emi * term
    # Since loan_amount is in thousands, divide by 1000
    recommended_loan = (max_emi * term) / 1000.0
    
    # Round to nearest thousand
    recommended_loan = round(recommended_loan, 1)
    
    return recommended_loan

def generate_recommendations(
    app_data: Dict[str, Any],
    risk_score: float,
    recommended_loan: float
) -> List[str]:
    """
    Generates actionable financial recommendations based on input features.
    """
    recommendations = []
    credit_history = app_data.get("Credit_History", 1.0)
    loan_amount = app_data.get("LoanAmount", 0.0)
    applicant_income = app_data.get("ApplicantIncome", 0.0)
    coapplicant_income = app_data.get("CoapplicantIncome", 0.0)
    total_income = applicant_income + coapplicant_income
    
    # 1. Credit History Recommendation
    if credit_history == 0.0:
        recommendations.append("Improve credit rating above 700: Pay off existing credit card debt and clear outstanding bills on time.")
    
    # 2. Debt to Income / Loan Amount Recommendation
    if loan_amount > recommended_loan:
        diff_amount = round((loan_amount - recommended_loan) * 1000)
        recommendations.append(f"Reduce loan request: Decrease your loan amount by ₹{diff_amount:,} to bring it under your safe limit of ₹{round(recommended_loan * 1000):,}.")
        
    # 3. Income Recommendations
    debt_ratio = loan_amount / (total_income + 1)
    if debt_ratio > 0.03: # Arbitrary high ratio for standard monthly capacity
        needed_income = (loan_amount * 1000) / 0.40 / 12 - total_income
        if needed_income > 0:
            recommendations.append(f"Increase annual income: Adding an extra ₹{round(needed_income):,} to your yearly earnings will improve approval probability.")

    # 4. Coapplicant Recommendation
    if coapplicant_income == 0.0 and risk_score > 40:
        recommendations.append("Apply with a co-applicant: Adding a family member with active salary and clean credit history will lower the application risk.")
        
    # 5. Term Length Recommendation
    term = app_data.get("Loan_Amount_Term", 360)
    if term < 360 and risk_score > 30:
        recommendations.append("Extend loan term: Extending the loan term (e.g., to 360 months) will decrease monthly EMI capacity constraints.")
        
    # If applicant is low risk and approved, give positive reinforcement
    if not recommendations and risk_score < 30:
        recommendations.append("Maintain credit excellence: Continue paying bills on time to enjoy premium interest rates.")
        
    return recommendations

def query_ai_financial_assistant(
    message: str,
    app_data: Dict[str, Any],
    pred_result: Dict[str, Any],
    chat_history: List[Dict[str, str]]
) -> str:
    """
    Answers user queries about their loan decision using Gemini API or rule-based fallback.
    """
    if not GEMINI_CONFIGURED:
        return get_fallback_ai_response(message, app_data, pred_result)
        
    try:
        # Build prompt context
        context = f"""
You are an expert AI Credit Officer and Financial Advisor. A customer is asking questions about their loan application.
Here are the customer's application details:
- Income: {app_data.get('ApplicantIncome')}
- Co-applicant Income: {app_data.get('CoapplicantIncome')}
- Loan Amount: {app_data.get('LoanAmount')} (in thousands)
- Loan Term: {app_data.get('Loan_Amount_Term')} months
- Credit History Status: {'Good (1.0)' if app_data.get('Credit_History') == 1.0 else 'Poor (0.0)'}
- Employment Status: {'Self Employed' if app_data.get('Self_Employed') == 'Yes' else 'Salaried Employee'}

System Predictions:
- Approved: {pred_result.get('approved')}
- Risk Score: {pred_result.get('risk_score')}/100 (Risk Level: {pred_result.get('risk_level')})
- Default Probability: {pred_result.get('default_probability') * 100:.1f}%
- Recommended Safe Loan Amount: {pred_result.get('recommended_loan_amount')} (in thousands)
- Suggested Interest Rate: {pred_result.get('predicted_interest_rate')}%
- Recommendations Generated: {', '.join(pred_result.get('recommendations', []))}

Respond to the user's question. Be encouraging, professional, and clear.
Analyze their metrics (especially why they were approved/rejected) and offer financial advice to help them improve.
        """
        
        # Initialize Gemini Model
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction="You are an empathetic, highly skilled financial advisor at a bank."
        )
        
        # Prepare chat history format
        contents = []
        for chat in chat_history or []:
            role = "user" if chat["role"] == "user" else "model"
            contents.append({"role": role, "parts": [chat["content"]]})
            
        contents.append({"role": "user", "parts": [f"{context}\n\nUser Question: {message}"]})
        
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        print(f"Gemini generation error: {e}. Falling back to rule-based engine.")
        return get_fallback_ai_response(message, app_data, pred_result)

def get_fallback_ai_response(message: str, app_data: Dict[str, Any], pred_result: Dict[str, Any]) -> str:
    """
    Rule-based response generator when Gemini API is unavailable.
    """
    message_lower = message.lower()
    
    # Setup values
    approved = pred_result.get('approved', False)
    risk_score = pred_result.get('risk_score', 50)
    rec_loan = pred_result.get('recommended_loan_amount', 100)
    interest = pred_result.get('predicted_interest_rate', 10.5)
    credit_hist = app_data.get('Credit_History', 1.0)
    
    rejection_reasons = []
    if credit_hist == 0.0:
        rejection_reasons.append("a poor credit history record (0.0)")
    if app_data.get('LoanAmount', 0.0) > rec_loan:
        rejection_reasons.append("requesting a loan amount higher than recommended based on your monthly income capacity")
    
    reasons_str = " and ".join(rejection_reasons) if rejection_reasons else "a elevated risk profile"

    if "why" in message_lower and ("reject" in message_lower or "denied" in message_lower or "approve" in message_lower):
        if approved:
            return (
                f"Your loan was approved because you have a healthy credit risk profile (Risk Score: {risk_score:.0f}/100) "
                f"and your credit history record is clean. We have offered a premium interest rate of {interest:.1f}%. "
                f"To keep this status, continue maintaining low credit card balances."
            )
        else:
            return (
                f"Your loan application was flagged as high risk (Risk Score: {risk_score:.0f}/100) and was rejected "
                f"primarily because of {reasons_str}.\n\n"
                f"To improve your chances:\n"
                f"1. Clear any outstanding balances to build up credit history.\n"
                f"2. Apply for a lower loan amount (close to ₹{round(rec_loan*1000):,}).\n"
                f"3. Secure a co-applicant to lower the overall default risk."
            )
            
    if "interest" in message_lower or "rate" in message_lower:
        return (
            f"Your dynamically calculated interest rate is {interest:.1f}%. This rate is directly tied to your "
            f"risk score of {risk_score:.0f}/100. Lowering your overall risk score by improving your credit history "
            f"will qualify you for lower rates (starting at 6.5%)."
        )
        
    if "recommend" in message_lower or "how to" in message_lower or "improve" in message_lower:
        return (
            f"Here are the top actions you can take to improve your approval chances:\n"
            + "\n".join([f"- {rec}" for rec in pred_result.get("recommendations", ["No specific recommendations at this time."])])
        )

    # Generic greeting
    return (
        f"Thank you for contacting customer support. Regarding your application: "
        f"Status: {'Approved' if approved else 'Rejected'}, "
        f"Risk Score: {risk_score:.0f}/100, "
        f"Recommended Loan Amount: ₹{round(rec_loan*1000):,}, "
        f"Interest Rate: {interest:.1f}%.\n"
        f"Please let me know if you would like tips on improving your score or understanding your interest rate."
    )
