
import streamlit as st
import joblib
import numpy as np

# Load model and threshold
model = joblib.load("logistic_regression_model.pkl")
threshold = 0.35

# App title
st.title("🩺 Heart Disease Risk Predictor")

st.write("Enter the patient's health data below to predict the risk of heart disease.")

# Input fields for all 21 features used in training
age = st.slider("Age Grp", 0, 15, 5)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
mental_health = st.slider("Mental Health Days (MentHlth)", 0, 30, 0)
physical_health = st.slider("Physical Health Days (PhysHlth)", 0, 30, 0)
gen_hlth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
education = st.selectbox("Education Level (1 = Less than High School, 6 = College Graduate)", [1, 2, 3, 4, 5, 6])
income = st.selectbox("Income Level (1 = <$10k, 8 = $75k+)", [1, 2, 3, 4, 5, 6, 7, 8])

# Binary inputs
def bin_input(label):
    return 1 if st.radio(label, ["No", "Yes"], horizontal=True) == "Yes" else 0

high_bp = bin_input("High Blood Pressure?")
high_chol = bin_input("High Cholesterol?")
chol_check = bin_input("Cholesterol Check in last 5 years?")
smoker = bin_input("Smoker?")
stroke = bin_input("Ever had a Stroke?")
diabetes = bin_input("Have Diabetes?")
phys_activity = bin_input("Physically Active?")
diff_walk = bin_input("Difficulty Walking?")
sex = bin_input("Sex (0 = Female, 1 = Male)?")
no_doc_cost = bin_input("Could not see doctor due to cost?")
any_healthcare = bin_input("Have any kind of health care coverage?")
hvy_alcohol = bin_input("Heavy Alcohol Consumption?")

# Final input vector (matching model's expected feature order)
input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
                        phys_activity, diff_walk, sex, age, education, income, gen_hlth,
                        mental_health, physical_health, no_doc_cost, any_healthcare, hvy_alcohol,
                        physical_health, mental_health]])

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = "At Risk" if prob >= threshold else "Not at Risk"

    st.subheader(f"Prediction: **{prediction}**")
    st.write(f"Predicted probability of heart disease: **{prob:.2%}**")
    if prob >= threshold:
        st.warning("⚠️ High risk detected. Recommend further medical evaluation.")
    else:
        st.success("✅ Low risk detected. Keep up the healthy habits!")
