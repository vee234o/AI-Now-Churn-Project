import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Customer Retention AI", page_icon="🏦", layout="centered")

# --- 1. LOAD ASSETS (The "Brain") ---
@st.cache_resource
def load_assets():
    try:
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_features.pkl', 'rb') as f:
            columns = pickle.load(f)
        return model, scaler, columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_assets()

# --- 2. HEADER SECTION ---
st.title("🏦 Customer Churn Predictor")
st.write("Predict if a customer is at risk of leaving based on their profile.")

if model is None:
    st.error("⚠️ Error: Model files not found.")
    st.info("Please upload 'churn_model.pkl', 'scaler.pkl', and 'model_features.pkl' to your GitHub repository.")
else:
    # --- 3. APP STRUCTURE (Tabs) ---
    tab1, tab2 = st.tabs(["⚡ Prediction Tool", "📊 Project Performance"])

    # === TAB 1: THE PREDICTION INTERFACE ===
    with tab1:
        st.subheader("Enter Customer Data")
        
        # Create a nice 2-column layout for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 600)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.slider("Tenure (Years)", 0, 10, 3)
            balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0)
        
        with col2:
            products = st.selectbox("Number of Products", [1, 2, 3, 4])
            salary = st.number_input("Est. Salary ($)", 0.0, 200000.0, 50000.0)
            active = st.selectbox("Is Active Member?", ["Yes", "No"])
            card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])

        # --- PREDICTION LOGIC ---
        if st.button("Analyze Risk 🚀", type="primary", use_container_width=True):
            # 1. Prepare Input Data (Initialize with 0s)
            input_data = {col: 0 for col in model_columns}
            
            # 2. Fill Numeric Values
            input_data['CreditScore'] = credit_score
            input_data['Age'] = age
            input_data['Tenure'] = tenure
            input_data['Balance'] = balance
            input_data['NumOfProducts'] = products
            input_data['EstimatedSalary'] = salary
            
            # 3. Manual Mappings (Must match Training Data!)
            input_data['IsActiveMember'] = 1 if active == "Yes" else 0
            input_data['HasCrCard'] = 1 if card == "Yes" else 0
            
            # One-Hot Encoding Mappings
            if country == "Germany" and 'Geography_Germany' in input_data:
                input_data['Geography_Germany'] = 1
            if country == "Spain" and 'Geography_Spain' in input_data:
                input_data['Geography_Spain'] = 1
            if gender == "Male" and 'Gender_Male' in input_data:
                input_data['Gender_Male'] = 1
            
            # 4. Create DataFrame & Scale
            df_input = pd.DataFrame([input_data])
            df_input = df_input[model_columns] # Enforce correct column order
            df_scaled = scaler.transform(df_input)
            
            # 5. Predict
            prediction = model.predict(df_scaled)
            probability = model.predict_proba(df_scaled)[0][1]
            
            # 6. Display Result
            st.divider()
            if prediction[0] == 1:
                st.error(f"🚨 **High Churn Risk!** (Probability: {probability:.1%})")
                
                # Smart Advice Logic
                st.caption("Risk Factors Identified:")
                if age > 45: st.write("• Customer age indicates high flight risk.")
                if balance > 80000: st.write("• High balance account (Financial threat).")
                if active == "No": st.write("• Inactive status increases churn probability.")
                
                st.warning("💡 **Recommendation:** Contact immediately with a retention offer.")
            else:
                st.success(f"✅ **Low Risk.** (Probability: {probability:.1%})")
                st.write("Customer is likely to stay.")

    # === TAB 2: PROJECT INSIGHTS ===
    with tab2:
        st.header("How the Model Works")
        
        st.subheader("1. What drives churn?")
        try:
            st.image("churn_drivers.png", caption="Feature Importance Analysis", use_container_width=True)
        except:
            st.info("ℹ️ Upload 'churn_drivers.png' to see the chart.")

        st.divider()

        st.subheader("2. Model Evaluation")
        try:
            st.image("confusion_matrix_final.png", caption="Confusion Matrix (Random Forest)", use_container_width=True)
        except:
            st.info("ℹ️ Upload 'confusion_matrix_final.png' to see the matrix.")
