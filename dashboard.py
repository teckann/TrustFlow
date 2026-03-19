import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Real-Time Fraud Shield",
    page_icon="🛡️",
    layout="wide"
)

# Load model data
@st.cache_resource
def load_model_data():
    with open('fraud_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open('feature_medians.pkl', 'rb') as f:
        medians = pickle.load(f)
    return model_data, le_dict, medians

model_data, le_dict, medians = load_model_data()
model = model_data['model']
features = model_data['features']

st.title("🛡️ Real-Time Fraud Shield for the Unbanked")
st.markdown("""
This dashboard demonstrates an AI-powered fraud detection system designed to protect digital wallet users in real-time.
It analyzes transaction behavior to detect anomalies and prevent fraudulent activities.
""")

# Sidebar for inputs
st.sidebar.header("Transaction Details")

with st.sidebar:
    transaction_id = st.number_input("Transaction ID", value=1001)
    amount = st.number_input("Transaction Amount ($)", value=st.session_state.get('amount', 50.0), step=1.0)
    card_type = st.selectbox("Card Type", options=['visa', 'mastercard', 'american express', 'discover'])
    card_category = st.selectbox("Card Category", options=['debit', 'credit'])
    email_domain = st.selectbox("Email Domain", options=['gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com', 'unknown'])
    device_type = st.selectbox("Device Type", options=['desktop', 'mobile', 'tablet'])
    
    st.markdown("---")
    st.subheader("Advanced Features (Optional)")
    c8 = st.slider("Frequency Count (C8)", 0, 200, value=st.session_state.get('c8', 1), help="Higher values indicate card frequency/testing.")
    v317 = st.slider("Transaction Vol (V317)", 0, 2000, value=st.session_state.get('v317', 0), help="Cumulative transaction volume in current session.")
    v287 = st.slider("Session Activity (V287)", 0, 10, value=st.session_state.get('v287', 0), help="Number of distinct actions in the current session.")
    
    st.markdown("---")
    if st.button("Load Fraud Example"):
         st.session_state.amount = 150.0
         st.session_state.c8 = 150
         st.session_state.v317 = 1800
         st.session_state.v287 = 8
         st.rerun()

if st.button("Analyze Transaction"):
    with st.spinner("Analyzing..."):
        # Prepare data for prediction
        input_data = {}
        for feat in features:
            if feat in medians:
                input_data[feat] = medians[feat]
            else:
                input_data[feat] = 0 # Default for non-numeric/missing medians
        
        # Mapping inputs to features
        input_data['TransactionID'] = transaction_id
        input_data['TransactionAmt'] = amount
        input_data['card4'] = card_type
        input_data['card6'] = card_category
        input_data['P_emaildomain'] = email_domain
        input_data['DeviceType'] = device_type
        input_data['C8'] = c8
        input_data['V317'] = v317
        input_data['V287'] = v287
        
        # Preprocess using label encoders
        for col, le in le_dict.items():
            if col in input_data:
                val = str(input_data[col])
                if val in le.classes_:
                    input_data[col] = le.transform([val])[0]
                else:
                    input_data[col] = le.transform(['unknown'])[0] if 'unknown' in le.classes_ else 0
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])[features]
        
        # Prediction
        start_time = time.time()
        prob = model.predict_proba(df_input)[0, 1]
        latency = (time.time() - start_time) * 1000
        
        # Display Results
        st.subheader("Risk Analysis Result")
        
        col1, col2, col3 = st.columns(3)
        
        # Determine status and color
        if prob < 0.015:
            status = "APPROVED"
            color = "green"
            action = "Transaction safe. No further action needed."
        elif prob < 0.05:
            status = "FLAGGED"
            color = "orange"
            action = "Suspicious activity detected. Manual review required."
        else:
            status = "BLOCKED"
            color = "red"
            action = "High risk of fraud. Transaction denied automatically."
            
        with col1:
            st.metric("Risk Score", f"{prob:.4f}")
        with col2:
            st.markdown(f"### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        with col3:
            st.metric("Latency", f"{latency:.2f} ms")
            
        st.info(f"**Action:** {action}")
        
        # Visualizing Risk Level
        st.markdown("---")
        st.subheader("Risk Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(["Risk Level"], [prob], color=color)
        ax.set_xlim(0, 0.1) # Adjusted for the new sensitivity
        ax.axvline(0.015, color='orange', linestyle='--', label='Flag Threshold')
        ax.axvline(0.05, color='red', linestyle='--', label='Block Threshold')
        ax.legend()
        st.pyplot(fig)

# Show data insights
st.markdown("---")
st.header("Model Insights")

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.subheader("Top Feature Importance")
    if os.path.exists('feature_importance.csv'):
        importance_df = pd.read_csv('feature_importance.csv').head(10)
        st.bar_chart(importance_df.set_index('feature'))
    else:
        st.write("Run training to see feature importance.")

with col_feat2:
    st.subheader("Fraud Patterns Analyzed")
    st.write("""
    - **Transaction Frequency**: High frequency of small transactions (C8) often indicates card testing.
    - **Behavioral Baselines**: Deviations from normal behavioral metrics (V30) trigger flags.
    - **Device Reputation**: Mobile devices in certain regions may have different risk profiles.
    - **Email Integrity**: Use of anonymous or unknown email domains increases risk scores.
    """)

st.markdown("---")
st.caption("Developed for VHack 2026 - Digital Trust Track. Powered by XGBoost and FastAPI.")
