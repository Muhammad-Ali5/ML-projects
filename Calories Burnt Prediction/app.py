import streamlit as st
import numpy as np
import joblib

# Load the trained model
model_filename = "XGB_reg Calories Burnt Predition.pkl"
loaded_model = joblib.load(model_filename)

# Streamlit app
st.title("Calories Burnt Prediction")

# Input fields for user data
st.header("Enter the details below:")

gender = st.selectbox("Gender (0 for Female, 1 for Male):", [0, 1])
age = st.number_input("Age (in years):", min_value=0, max_value=120, value=25)
height = st.number_input("Height (in cm):", min_value=50.0, max_value=250.0, value=165.0)
weight = st.number_input("Weight (in kg):", min_value=10.0, max_value=200.0, value=65.0)
duration = st.number_input("Duration of Exercise (in minutes):", min_value=0.0, max_value=300.0, value=30.0)
heart_rate = st.number_input("Heart Rate (in bpm):", min_value=30.0, max_value=200.0, value=85.0)
body_temp = st.number_input("Body Temperature (in °C):", min_value=30.0, max_value=45.0, value=37.0)

# Predict button
if st.button("Predict Calories Burnt"):
    # Prepare input data
    input_data = (gender, age, height, weight, duration, heart_rate, body_temp)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_as_numpy_array)

    # Display result
    st.success(f"Estimated Calories Burnt: {prediction[0]:.2f}")