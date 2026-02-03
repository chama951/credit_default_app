from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
model = joblib.load("credit_default_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

app = FastAPI(title="Credit Default Prediction API")

# Configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Enums for categorical fields - MUST match your HTML dropdown values
class HomeOwnership(str, Enum):
    home_mortgage = "home_mortgage"
    own_home = "own_home"
    rent = "rent"

class YearsInJob(str, Enum):
    less_than_1_year = "less_than_1_year"
    years_2 = "2_years"
    years_3 = "3_years"
    years_4 = "4_years"
    years_5 = "5_years"
    years_6 = "6_years"
    years_7 = "7_years"
    years_8 = "8_years"
    years_9 = "9_years"
    years_10_plus = "10plus_years"

class LoanPurpose(str, Enum):
    buy_a_car = "buy_a_car"
    buy_house = "buy_house"
    debt_consolidation = "debt_consolidation"
    educational_expenses = "educational_expenses"
    home_improvements = "home_improvements"
    major_purchase = "major_purchase"
    medical_bills = "medical_bills"
    moving = "moving"
    other = "other"
    small_business = "small_business"
    take_a_trip = "take_a_trip"
    vacation = "vacation"
    wedding = "wedding"
    renewable_energy = "renewable_energy"

class Term(str, Enum):
    short_term = "short_term"
    long_term = "long_term"

class CreditInput(BaseModel):
    # Numeric fields
    annual_income: float
    credit_score: float
    number_of_open_accounts: float
    years_of_credit_history: float
    maximum_open_credit: float
    current_loan_amount: float
    current_credit_balance: float
    monthly_debt: float
    tax_liens: float
    number_of_credit_problems: float
    bankruptcies: float
    
    # Categorical fields (as dropdown enums)
    home_ownership: HomeOwnership
    years_in_current_job: YearsInJob
    purpose: LoanPurpose
    term: Term

@app.post("/predict")
def predict_credit_default(data: CreditInput):
    """Predict credit default risk"""
    
    # Convert to dict and prepare for one-hot encoding
    input_dict = data.dict()
    
    print(f"Received input: {input_dict}")
    
    # Create a dictionary with all possible boolean features initialized to False
    all_features = {
        # Numeric features
        'annual_income': input_dict['annual_income'],
        'tax_liens': input_dict['tax_liens'],
        'number_of_open_accounts': input_dict['number_of_open_accounts'],
        'years_of_credit_history': input_dict['years_of_credit_history'],
        'maximum_open_credit': input_dict['maximum_open_credit'],
        'number_of_credit_problems': input_dict['number_of_credit_problems'],
        'bankruptcies': input_dict['bankruptcies'],
        'current_loan_amount': input_dict['current_loan_amount'],
        'current_credit_balance': input_dict['current_credit_balance'],
        'monthly_debt': input_dict['monthly_debt'],
        'credit_score': input_dict['credit_score'],
        
        # Home ownership
        'home_ownership_home_mortgage': input_dict['home_ownership'] == 'home_mortgage',
        'home_ownership_own_home': input_dict['home_ownership'] == 'own_home',
        'home_ownership_rent': input_dict['home_ownership'] == 'rent',
        
        # Years in job
        'years_in_current_job_10plus_years': input_dict['years_in_current_job'] == '10plus_years',
        'years_in_current_job_2_years': input_dict['years_in_current_job'] == '2_years',
        'years_in_current_job_3_years': input_dict['years_in_current_job'] == '3_years',
        'years_in_current_job_4_years': input_dict['years_in_current_job'] == '4_years',
        'years_in_current_job_5_years': input_dict['years_in_current_job'] == '5_years',
        'years_in_current_job_6_years': input_dict['years_in_current_job'] == '6_years',
        'years_in_current_job_7_years': input_dict['years_in_current_job'] == '7_years',
        'years_in_current_job_8_years': input_dict['years_in_current_job'] == '8_years',
        'years_in_current_job_9_years': input_dict['years_in_current_job'] == '9_years',
        'years_in_current_job_less_than_1_year': input_dict['years_in_current_job'] == 'less_than_1_year',
        
        # Loan purpose
        'purpose_buy_a_car': input_dict['purpose'] == 'buy_a_car',
        'purpose_buy_house': input_dict['purpose'] == 'buy_house',
        'purpose_debt_consolidation': input_dict['purpose'] == 'debt_consolidation',
        'purpose_educational_expenses': input_dict['purpose'] == 'educational_expenses',
        'purpose_home_improvements': input_dict['purpose'] == 'home_improvements',
        'purpose_major_purchase': input_dict['purpose'] == 'major_purchase',
        'purpose_medical_bills': input_dict['purpose'] == 'medical_bills',
        'purpose_moving': input_dict['purpose'] == 'moving',
        'purpose_other': input_dict['purpose'] == 'other',
        'purpose_small_business': input_dict['purpose'] == 'small_business',
        'purpose_take_a_trip': input_dict['purpose'] == 'take_a_trip',
        'purpose_vacation': input_dict['purpose'] == 'vacation',
        'purpose_wedding': input_dict['purpose'] == 'wedding',
        'purpose_renewable_energy': input_dict['purpose'] == 'renewable_energy',
        
        # Loan term
        'term_short_term': input_dict['term'] == 'short_term'
    }
    
    print(f"Processed features: {list(all_features.keys())}")
    print(f"Home ownership value: {input_dict['home_ownership']}")
    print(f"Years in job value: {input_dict['years_in_current_job']}")
    print(f"Purpose value: {input_dict['purpose']}")
    print(f"Term value: {input_dict['term']}")
    
    # Convert to DataFrame
    input_df = pd.DataFrame([all_features])
    
    # Ensure column order matches training
    input_df = input_df[feature_order]
    
    print(f"DataFrame shape: {input_df.shape}")
    print(f"Expected features: {len(feature_order)}")
    
    # Scale numeric features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Get prediction explanation
    explanation = get_prediction_explanation(prediction, probability, input_dict)
    
    return {
        "credit_default": int(prediction),
        "default_probability": round(float(probability), 4),
        "prediction_label": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": "High" if probability > 0.8 else "Medium" if probability > 0.6 else "Low",
        "explanation": explanation,
        "processed_fields": len(feature_order)
    }

def get_prediction_explanation(prediction, probability, input_data):
    """Generate human-readable explanation for prediction"""
    explanations = []
    
    if prediction == 1:  # High risk
        if input_data['credit_score'] < 600:
            explanations.append("Low credit score")
        if input_data['bankruptcies'] > 0:
            explanations.append("Previous bankruptcies")
        if input_data['number_of_credit_problems'] > 0:
            explanations.append("History of credit problems")
        if input_data['monthly_debt'] > (input_data['annual_income'] / 12 * 0.5):
            explanations.append("High debt-to-income ratio")
        
        if not explanations:
            explanations.append("Multiple risk factors detected")
    else:  # Low risk
        if input_data['credit_score'] > 750:
            explanations.append("Excellent credit score")
        if input_data['years_of_credit_history'] > 10:
            explanations.append("Long credit history")
        if input_data['bankruptcies'] == 0 and input_data['number_of_credit_problems'] == 0:
            explanations.append("Clean credit record")
        
        if not explanations:
            explanations.append("Acceptable risk profile")
    
    return f"Based on analysis: {', '.join(explanations)}."

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "API is running", "service": "Credit Default Prediction", "model_loaded": True}

@app.get("/test-input")
def test_input():
    """Return a valid test input structure"""
    return {
        "annual_income": 72000.0,
        "credit_score": 710.0,
        "number_of_open_accounts": 6.0,
        "years_of_credit_history": 12.0,
        "maximum_open_credit": 30000.0,
        "current_loan_amount": 15000.0,
        "current_credit_balance": 6000.0,
        "monthly_debt": 420.0,
        "tax_liens": 0.0,
        "number_of_credit_problems": 0.0,
        "bankruptcies": 0.0,
        "home_ownership": "home_mortgage",
        "years_in_current_job": "3_years",
        "purpose": "buy_house",
        "term": "short_term"
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "root": "/",
            "health_check": "/health",
            "test_input": "/test-input",
            "prediction": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "input_format": {
            "numeric_fields": [
                "annual_income", "credit_score", "number_of_open_accounts",
                "years_of_credit_history", "maximum_open_credit", "current_loan_amount",
                "current_credit_balance", "monthly_debt", "tax_liens",
                "number_of_credit_problems", "bankruptcies"
            ],
            "categorical_fields": {
                "home_ownership": ["home_mortgage", "own_home", "rent"],
                "years_in_current_job": [
                    "less_than_1_year", "2_years", "3_years", "4_years",
                    "5_years", "6_years", "7_years", "8_years", "9_years",
                    "10plus_years"
                ],
                "purpose": [
                    "buy_a_car", "buy_house", "debt_consolidation", "educational_expenses",
                    "home_improvements", "major_purchase", "medical_bills", "moving",
                    "other", "small_business", "take_a_trip", "vacation", "wedding",
                    "renewable_energy"
                ],
                "term": ["short_term", "long_term"]
            }
        }
    }

@app.options("/{rest_of_path:path}")
async def options_handler(rest_of_path: str):
    """Handle OPTIONS requests for CORS"""
    return {}