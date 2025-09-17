import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("card_transdata.csv")

@st.cache_resource
def load_model():
    return joblib.load("my_RFC_model Credit card fraud Prediction.joblib")

data = load_data()
model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

if page == "Dashboard":
    st.header("📊 Data Visualization")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud vs Legit Transactions")
        fig, ax = plt.subplots()
        sns.countplot(x="fraud", data=data, ax=ax, palette="Set2")
        ax.set_xticklabels(["Legit", "Fraud"])
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.heatmap(data.corr(), annot=False, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    st.subheader("Sample Data")
    st.dataframe(data.head(10))

elif page == "Prediction":
    st.header("🔎 Predict Fraudulent Transaction")
    st.write("Enter transaction details:")

    # Get feature names except 'fraud'
    features = [col for col in data.columns if col != "fraud"]
    user_input = []
    for feat in features:
        val = st.number_input(f"{feat}", value=float(data[feat].mean()))
        user_input.append(val)

    if st.button("Predict"):
        input_array = np.array(user_input).reshape(1, -1)
        pred = model.predict(input_array)[0]
        if pred == 1:
            st.error("⚠️ This transaction is predicted to be **FRAUDULENT**!")
        else:
            st.success("✅ This transaction is predicted to be **LEGITIMATE**.")

    st.info("Model used: Random Forest Classifier")

st.markdown(
    """
    <style>
    .stButton>button {background-color: #0099ff; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)