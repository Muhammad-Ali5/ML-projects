import streamlit as st
import pandas as pd
import joblib

# Load encoder and model
encoder = joblib.load("encoder.joblib")
model = joblib.load("Linear car price prediction.joblib")

st.title("Car Price Prediction")

# Sidebar inputs
st.sidebar.header("Input Car Details")

year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, value=2018)
mileage = st.sidebar.number_input("Mileage", min_value=0, value=30000)
tax = st.sidebar.number_input("Tax", min_value=0, value=150)
mpg = st.sidebar.number_input("MPG", min_value=0.0, value=50.0)
engineSize = st.sidebar.number_input("Engine Size", min_value=0.0, value=1.0)

model_options = encoder.categories_[0]
transmission_options = encoder.categories_[1]
fuelType_options = encoder.categories_[2]

model_input = st.sidebar.selectbox("Model", model_options)
transmission_input = st.sidebar.selectbox("Transmission", transmission_options)
fuelType_input = st.sidebar.selectbox("Fuel Type", fuelType_options)

if st.sidebar.button("Predict Price"):
    # Prepare input
    input_df = pd.DataFrame({
        "model": [model_input],
        "transmission": [transmission_input],
        "fuelType": [fuelType_input],
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    # Encode categorical features
    encoded = encoder.transform(input_df[["model", "transmission", "fuelType"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["model", "transmission", "fuelType"]))
    input_final = pd.concat([input_df[["year", "mileage", "tax", "mpg", "engineSize"]], encoded_df], axis=1)

    # Predict
    price = model.predict(input_final)[0]
    st.success(f"Predicted Car Price: £{price:,.2f}")