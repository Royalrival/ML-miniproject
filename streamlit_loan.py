import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the app
st.title("Loan Approval Prediction")

# Feature Input Section
st.header("Enter Loan Application Details")

# Define input fields based on loan dataset features
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Calculate additional features based on input
total_income = applicant_income + coapplicant_income
emi = loan_amount / loan_amount_term if loan_amount_term != 0 else 0

# Encoding inputs to match training format
input_data = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": 1 if credit_history == "Yes" else 0,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
    "TotalIncome": total_income,
    "EMI": emi
}

# Convert to DataFrame, remove feature names by converting to NumPy array
input_df = pd.DataFrame([input_data]).to_numpy()

# Predict button
if st.button("Predict Loan Approval Status"):
    # Perform prediction
    prediction = model.predict(input_df)[0]
    
    # Log prediction result for debugging
    st.write("Prediction Raw Output:", prediction)

    # Determine the loan status based on prediction result
    if prediction == 1:
        st.success("The loan is likely to be approved.")
    elif prediction == 0:
        st.error("The loan is likely to be rejected.")
    else:
        st.warning("Unexpected prediction result. Please check the model or input data.")
