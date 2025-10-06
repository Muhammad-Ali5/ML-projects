import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("DTR Medical insurance cost prediction.pkl")

st.title("Medical Insurance Cost Prediction")

# User input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# One-hot encoding for categorical variables
sex_male = 1 if sex == "male" else 0
sex_female = 1 if sex == "female" else 0
smoker_yes = 1 if smoker == "yes" else 0
smoker_no = 1 if smoker == "no" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
region_northeast = 1 if region == "northeast" else 0
region_northwest = 1 if region == "northwest" else 0

# Arrange input as per your model's expected order
input_data = np.array([[age, bmi, children, sex_female, sex_male,
                        smoker_no, smoker_yes,
                        region_northeast, region_northwest, region_southeast, region_southwest]])

if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Medical Insurance Cost: ${prediction[0]:.2f}")