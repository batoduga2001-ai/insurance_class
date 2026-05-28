import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('insurance_classifier.pkl')

st.title("Insurance Risk Level Predictor")

st.write("Enter patient details to predict insurance risk level.")

# Inputs (NO REGION)
age = st.number_input("Age", 0, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])

# Encoding (must match training)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

# Prediction
if st.button("Predict Risk Level"):
    input_data = np.array([[age, sex, bmi, children, smoker]])

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("Low Risk 🟢")
    elif prediction == 1:
        st.warning("Medium Risk 🟡")
    else:
        st.error("High Risk 🔴")