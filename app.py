import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Sidebar ---
st.sidebar.title("🧠 About This Project")
st.sidebar.info("""
This project detects fraudulent credit card transactions using the XGBoost algorithm.

Built by **Shaik Omer Farooq** — a passionate learner transitioning from a non-IT to IT background!

📊 Data Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
""")

# --- Title ---
st.title("💳 Credit Card Fraud Detection with XGBoost")

# --- Load Model and Data ---
model = pickle.load(open('fraud_model.pkl', 'rb'))
X_test = joblib.load(open('X_test.pkl', 'rb'))
y_test = joblib.load(open('y_test.pkl', 'rb'))

# --- Section 1: Prediction ---
st.header("🔍 Predict Fraudulent Transactions")

# For demo purposes, we use a random row from X_test
sample_index = st.number_input("Choose a test sample index:", min_value=0, max_value=len(X_test)-1, step=1)
sample = X_test.iloc[sample_index:sample_index+1]

if st.button("🔎 Predict"):
    prediction = model.predict(sample)[0]
    result = "✅ Legit Transaction" if prediction == 0 else "⚠️ Fraudulent Transaction"
    st.subheader("Prediction Result:")
    st.success(result)

# --- Section 2: Model Evaluation ---
st.header("📊 Model Evaluation Metrics")

y_pred = model.predict(X_test)
col1, col2 = st.columns(2)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

with st.expander("📋 Classification Report"):
    st.text(classification_report(y_test, y_pred))

with st.expander("📉 Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

# --- Section 3: Feature Importance ---
st.header("🧠 Feature Importance (Top 10)")

try:
    importances = model.feature_importances_
    features = X_test.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(top_features['Feature'], top_features['Importance'], color="skyblue")
    ax_imp.set_xlabel("Importance Score")
    ax_imp.set_title("Top 10 Important Features")
    st.pyplot(fig_imp)
except AttributeError:
    st.warning("Feature importance is not available for this model.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by [Shaik Omer Farooq](https://github.com/YOUR_GITHUB_USERNAME)")


