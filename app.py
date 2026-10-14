import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "models/loan_model.pkl"
    )


model = load_model()


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant details below "
    "to predict the loan approval status."
)


# --------------------------------------------------
# User inputs
# --------------------------------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


married = st.selectbox(
    "Married",
    ["Yes", "No"]
)


dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)


education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)


self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)


applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)


coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)


loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)


loan_term = st.number_input(
    "Loan Amount Term",
    min_value=0,
    value=360
)


credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)


property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame(
        {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        }
    )


    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0][1]


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    if prediction == 1:

        st.success(
            "✅ Loan is likely to be APPROVED"
        )

    else:

        st.error(
            "❌ Loan is likely to be REJECTED"
        )


    st.write(
        f"Approval probability: "
        f"{probability * 100:.2f}%"
    )


    st.info(
        "This prediction is based on the "
        "trained machine-learning model and "
        "should not be considered a final "
        "financial decision."
    )