import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Retention AI", page_icon="🏦", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # We use a simple check to see if files exist to avoid crashing
    if not os.path.exists('churn_model.pkl'):
        return None, None, None
    
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, scaler, columns

model, scaler, model_columns = load_assets()

# --- HEADER ---
st.title("🏦 Customer Churn Predictor")
st.write("Predict if a customer is at risk of leaving based on their profile.")

if model is None:
    st.error("⚠️ Model files not found.")
    st.info("Please upload 'churn_model.pkl', 'scaler.pkl', and 'model_features.pkl' to GitHub.")
else:
    # --- TABS ---
    tab1, tab2 = st.tabs(["⚡ Prediction Tool", "📊 Project Insights"])

    # === TAB 1: PREDICTION ===
    with tab1:
        st.subheader("Enter Customer Details")
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 600)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.slider("Tenure (Years)", 0, 10, 3)
            balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0)
        
        with col2:
            products = st.selectbox("Products", [1, 2, 3, 4])
            salary = st.number_input("Salary ($)", 0.0, 200000.0, 50000.0)
            active = st.selectbox("Active Member?", ["Yes", "No"])
            card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])

        if st.button("Analyze Risk 🚀", type="primary", use_container_width=True):
            # 1. Prepare Data
            input_data = {col: 0 for col in model_columns}
            input_data['CreditScore'] = credit_score
            input_data['Age'] = age
            input_data['Tenure'] = tenure
            input_data['Balance'] = balance
            input_data['NumOfProducts'] = products
            input_data['EstimatedSalary'] = salary
            input_data['IsActiveMember'] = 1 if active == "Yes" else 0
            input_data['HasCrCard'] = 1 if card == "Yes" else 0
            
            # One-Hot Encoding
            if country == "Germany" and 'Geography_Germany' in input_data:
                input_data['Geography_Germany'] = 1
            if country == "Spain" and 'Geography_Spain' in input_data:
                input_data['Geography_Spain'] = 1
            if gender == "Male" and 'Gender_Male' in input_data:
                input_data['Gender_Male'] = 1
            
            # 2. Predict
            df_input = pd.DataFrame([input_data])
            df_input = df_input[model_columns]
            df_scaled = scaler.transform(df_input)
            
            prediction = model.predict(df_scaled)
            probability = model.predict_proba(df_scaled)[0][1]
            
            # 3. Result
            st.divider()
            if prediction[0] == 1:
                st.error(f"🚨 **High Churn Risk!** (Probability: {probability:.1%})")
                st.caption("Key Drivers Detected:")
                if age > 45: st.write("• Age (>45) is a high-risk factor.")
                if balance > 80000: st.write("• High account balance.")
                st.warning("Recommendation: Offer retention bonus immediately.")
            else:
                st.success(f"✅ **Safe.** (Probability: {probability:.1%})")
                st.write("Customer is likely to stay.")

    # === TAB 2: INSIGHTS ===
    with tab2:
        st.header("Project Analysis")
        
        st.subheader("1. Feature Importance")
        # CHECK IF FILE EXISTS (Safe Method)
        if os.path.exists("churn_drivers.png"):
            st.image("churn_drivers.png", caption="Age & Balance are key drivers", use_container_width=True)
        else:
            st.info("ℹ️ Upload 'churn_drivers.png' to see the chart.")

        st.divider()

        st.subheader("2. Model Accuracy")
        # CHECK IF FILE EXISTS (Safe Method)
        if os.path.exists("confusion_matrix_final.png"):
            st.image("confusion_matrix_final.png", caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("ℹ️ Upload 'confusion_matrix_final.png' to see the matrix.")
            
