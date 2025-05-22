import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('fraud_model.pkl', 'rb'))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below and check if it's fraudulent.")

# Create a form to collect inputs
with st.form(key="input_form"):
    v_features = []
    for i in range(1, 29):  # V1 to V28
        v = st.number_input(f"V{i}", step=0.01)
        v_features.append(v)

    amount = st.number_input("Transaction Amount", step=0.01)
    v_features.append(amount)

    submit = st.form_submit_button("🔍 Predict Fraud")

if submit:
    input_data = np.array([v_features])
    prediction = model.predict(input_data)
    result = "⚠️ Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.success(result)


