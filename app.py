import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart AI Pro", layout="wide")

@st.cache_resource
def train_models():
    try:
        df = pd.read_csv("heart_disease.csv")
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]
        df = df.fillna(0)
        target = df.columns[-1]
        X = df.drop(target, axis=1)
        y = df[target]
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        dt = DecisionTreeClassifier(random_state=42).fit(X, y)
        return rf, lr, dt, X.columns.tolist(), df
    except:
        return None, None, None, None, None

rf_model, lr_model, dt_model, feature_cols, raw_df = train_models()

if rf_model is not None:
    st.title("❤️ Heart Disease Diagnostic Center")
    
    # 1. Patient Name Input
    p_name = st.text_input("👤 Enter Patient Full Name", placeholder="e.g. Ramji Pandey")
    
    st.subheader("📋 Clinical Parameters")
    user_inputs = {}
    
    # Dropdown Columns
    binary_cols = ['High Blood Pressure', 'High LDL Cholesterol', 'Smoking', 'Diabetes', 'Exercise Habits', 'Family Heart Disease', 'Alcohol Consumption']

    with st.form("diagnostic_form"):
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            with cols[i % 3]:
                # Gender Dropdown Logic
                if col_name.lower() == 'gender':
                    gender_choice = st.selectbox("Gender", options=["Female", "Male"], key="gender_select")
                    user_inputs[col_name] = 1 if gender_choice == "Male" else 0
                # Yes/No Dropdown Logic
                elif col_name in binary_cols:
                    choice = st.selectbox(f"{col_name}", options=["No", "Yes"], key=f"input_{col_name}")
                    user_inputs[col_name] = 1 if choice == "Yes" else 0
                # Numeric Input
                else:
                    avg_val = float(raw_df[col_name].mean())
                    user_inputs[col_name] = st.number_input(f"{col_name}", value=avg_val, key=f"input_{col_name}")
        
        submitted = st.form_submit_button("Generate Diagnostic Report")

    if submitted:
        if p_name:
            input_data = np.array([user_inputs[c] for c in feature_cols]).reshape(1, -1)
            
            # Predictions
            rf_p = rf_model.predict_proba(input_data)[0][1] * 100
            lr_p = lr_model.predict_proba(input_data)[0][1] * 100
            dt_p = dt_model.predict_proba(input_data)[0][1] * 100
            
            st.markdown(f"## 📝 Diagnostic Report for: {p_name}")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("### 📊 Accuracy-based Risk Scores")
                res_df = pd.DataFrame({
                    "Algorithm": ["Random Forest", "Logistic Regression", "Decision Tree"],
                    "Risk Probability": [f"{rf_p:.1f}%", f"{lr_p:.1f}%", f"{dt_p:.1f}%"]
                })
                st.table(res_df)
            
            with c2:
                st.write("### 📈 Visual Analysis")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["RF", "LR", "DT"], [rf_p, lr_p, dt_p], color=['#FF4B4B', '#1C83E1', '#00C781'])
                ax.set_ylim(0, 100)
                st.pyplot(fig)
            
            # Status Logic
            if rf_p > 30:
                st.error(f"### ⚠️ FINAL VERDICT: HIGH RISK ({rf_p:.1f}%)")
                status = "HIGH RISK"
            else:
                st.success(f"### ✅ FINAL VERDICT: NORMAL ({rf_p:.1f}%)")
                status = "NORMAL"
                
            # Download Data
            report_txt = f"HEART DIAGNOSTIC REPORT\n--------------------\nPatient: {p_name}\nVerdict: {status}\n\nDetailed Scores:\nRandom Forest: {rf_p:.1f}%\nLogistic Regression: {lr_p:.1f}%\nDecision Tree: {dt_p:.1f}%"
            st.download_button("📥 Download Official Report", data=report_txt, file_name=f"{p_name}_Heart_Report.txt")
        else:
            st.warning("Please enter the patient's name before generating the report.")
else:
    st.error("Model training failed. Please check the CSV file.")
