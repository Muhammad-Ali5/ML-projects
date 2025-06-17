import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Big Mart Sales Prediction", layout="wide")

# Custom CSS with Tailwind CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .title { font-size: 2.5rem; font-weight: bold; color: #1e40af; text-align: center; margin-bottom: 1rem; }
        .subtitle { font-size: 1.25rem; color: #4b5563; text-align: center; margin-bottom: 2rem; }
        .prediction { font-size: 1.5rem; font-weight: bold; color: #047857; text-align: center; margin-top: 1.5rem; }
        .error { font-size: 1.25rem; color: #dc2626; text-align: center; margin-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">Big Mart Sales Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter product and outlet details to predict sales</div>', unsafe_allow_html=True)

# Load the trained model
model = joblib.load("ali_trained_model.joblib")

# Create input form
with st.form(key="prediction_form"):
    st.subheader("Product Details")
    col1, col2 = st.columns(2)

    with col1:
        item_identifier = st.number_input("Item Identifier (0-1558)", min_value=0, max_value=1558, value=0)
        item_weight = st.number_input("Item Weight (4.5-21.35)", min_value=4.5, max_value=21.35, value=12.79)
        item_fat_content = st.selectbox("Item Fat Content", options=[0, 1], format_func=lambda x: "Low Fat" if x == 0 else "Regular")
        item_visibility = st.number_input("Item Visibility (0-0.33)", min_value=0.0, max_value=0.33, value=0.07)

    with col2:
        item_type = st.selectbox("Item Type", options=list(range(16)), format_func=lambda x: {
            0: "Baking Goods", 1: "Breads", 2: "Breakfast", 3: "Canned", 4: "Dairy", 5: "Frozen Foods",
            6: "Fruits and Vegetables", 7: "Hard Drinks", 8: "Health and Hygiene", 9: "Household", 10: "Meat",
            11: "Others", 12: "Seafood", 13: "Snack Foods", 14: "Soft Drinks", 15: "Starchy Foods"
        }.get(x))
        item_mrp = st.number_input("Item MRP (31.29-266.89)", min_value=31.29, max_value=266.89, value=141.0)

    st.subheader("Outlet Details")
    col3, col4 = st.columns(2)

    with col3:
        outlet_identifier = st.selectbox("Outlet Identifier", options=list(range(10)), format_func=lambda x: f"OUT0{x+10}")
        outlet_establishment_year = st.number_input("Outlet Establishment Year (1985-2009)", min_value=1985, max_value=2009, value=1997)
        outlet_size = st.selectbox("Outlet Size", options=[0, 1, 2], format_func=lambda x: {0: "High", 1: "Medium", 2: "Small"}.get(x))

    with col4:
        outlet_location_type = st.selectbox("Outlet Location Type", options=[0, 1, 2], format_func=lambda x: f"Tier {x+1}")
        outlet_type = st.selectbox("Outlet Type", options=[0, 1, 2, 3], format_func=lambda x: {
            0: "Grocery Store", 1: "Supermarket Type1", 2: "Supermarket Type2", 3: "Supermarket Type3"
        }.get(x))
        source = st.selectbox("Source", options=[0, 1], format_func=lambda x: "Test" if x == 0 else "Train")

    # Submit button
    submit_button = st.form_submit_button(label="Predict Sales", use_container_width=True)

# Process input and make prediction
if submit_button:
    try:
        # Create input array
        input_data = np.array([[
            item_identifier, item_weight, item_fat_content, item_visibility, item_type, item_mrp,
            outlet_identifier, outlet_establishment_year, outlet_size, outlet_location_type, outlet_type, source
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown(f'<div class="prediction">Predicted Sales: ${prediction:.2f}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error">Error: {str(e)}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr class="my-4">
    <p class="text-center text-gray-500">Powered by Streamlit & GradientBoostingRegressor</p>
""", unsafe_allow_html=True)