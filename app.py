import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to check if it's fraudulent.")

# List of input features (excluding 'Time' & 'Class')
features = [f'V{i}' for i in range(1, 29)] + ['Amount']

user_input = []

for feature in features:
    value = st.number_input(f"{feature}", value=0.0, format="%.4f")
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ Warning: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")
