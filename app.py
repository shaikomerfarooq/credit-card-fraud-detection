import streamlit as st
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Load model and test data
model = pickle.load(open('fraud_model.pkl', 'rb'))
X_test = joblib.load(open('X_test.pkl', 'rb'))
y_test = joblib.load(open('y_test.pkl', 'rb'))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to check if it's fraudulent.")

# --- Input Form ---
with st.form(key="input_form"):
    v_features = [st.number_input(f"V{i}", step=0.01) for i in range(1, 29)]
    amount = st.number_input("Transaction Amount", step=0.01)
    v_features.append(amount)
    submit = st.form_submit_button("🔍 Predict Fraud")

# --- Prediction ---
if submit:
    input_data = np.array([v_features])
    prediction = model.predict(input_data)
    result = "⚠️ Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.success(result)

# --- Feature Importance ---
st.subheader("🔎 Feature Importance")
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, importances, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# --- Model Performance ---
st.subheader("📊 Model Evaluation on Test Data")

# Confusion Matrix
st.markdown("### Confusion Matrix")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm, cmap='Blues', alpha=0.7)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
st.pyplot(fig)

# ROC Curve
st.markdown("### ROC Curve")
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
st.pyplot(fig)

# Precision-Recall Curve
st.markdown("### Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_probs)
fig, ax = plt.subplots()
ax.plot(recall, precision, color='green')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
st.pyplot(fig)


