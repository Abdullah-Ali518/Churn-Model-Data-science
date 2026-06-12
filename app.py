import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Assets Loading
@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler (5).pkl")
    # Feature names check bypass
    if hasattr(model, 'feature_names_in_'): model.feature_names_in_ = None
    if hasattr(scaler, 'feature_names_in_'): scaler.feature_names_in_ = None
    return model, scaler

model, scaler = load_assets()

# 2. Columns Setup
# Numerical columns jinpar scaler train hua hai (Exactly 3 features)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Baqi 27 columns (8 user inputs - 3 numerical = 5 + 22 hidden = 27)
all_30_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'gender_Female', 'Partner_No', 'Dependents_No', 'PhoneService_No',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes'
]

# --- UI (User Inputs) ---
# (Wohi 8 inputs rakhein jo pehle code mein thay)
st.title("📞 Telco Churn Predictor")
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

if st.button("🚀 Predict Churn"):
    try:
        # STEP 1: Sirf un 3 columns ko scale karein jo scaler mang raha hai
        numeric_input = np.array([[tenure, monthly_charges, total_charges]])
        scaled_numeric = scaler.transform(numeric_input) # Output: 3 features

        # STEP 2: Ab 30 columns ka full dictionary banayein
        input_dict = {feature: 0 for feature in all_30_features}
        
        # Scaled values ko update karein
        input_dict['tenure'] = scaled_numeric[0][0]
        input_dict['MonthlyCharges'] = scaled_numeric[0][1]
        input_dict['TotalCharges'] = scaled_numeric[0][2]
        
        # Categorical values ko update karein (0/1)
        input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
        input_dict['gender_Male'] = 1 if gender == "Male" else 0
        input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
        input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
        input_dict['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0

        # STEP 3: Final DataFrame aur Prediction (Exactly 30 columns in order)
        final_input_df = pd.DataFrame([input_dict])[all_30_features]
        
        prediction = model.predict(final_input_df.values)[0]
        prob = model.predict_proba(final_input_df.values)[0][1]

        if prediction == 1:
            st.error(f"🚨 Risk! (Prob: {prob:.2%})")
        else:
            st.success(f"✅ Safe! (Prob: {prob:.2%})")

    except Exception as e:
        st.error(f"Error: {e}")