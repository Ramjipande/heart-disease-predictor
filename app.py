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
        X = df.drop(df.columns[-1], axis=1)
        y = df[df.columns[-1]]
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        dt = DecisionTreeClassifier(random_state=42).fit(X, y)
        return rf, lr, dt, X.columns.tolist(), df
    except:
        return None, None, None, None, None

rf_model, lr_model, dt_model, feature_cols, raw_df = train_models()

if rf_model is not None:
    st.title("❤️ Heart Disease Diagnostic Center")
    p_name = st.text_input("👤 Patient Full Name")
    
    st.subheader("📋 Enter Clinical Data")
    user_inputs = {}
    cols = st.columns(3)
    
    # Yes/No Mapping
    binary_cols = ['High Blood Pressure', 'High LDL Cholesterol', 'Smoking', 'Diabetes', 'Exercise Habits', 'Family Heart Disease', 'Alcohol Consumption']

    for i, col_name in enumerate(feature_cols):
        with cols[i % 3]:
            if col_name in binary_cols:
                choice = st.selectbox(f"{col_name}", options=["No", "Yes"], key=col_name)
                user_inputs[col_name] = 1 if choice == "Yes" else 0
            else:
                avg_val = float(raw_df[col_name].mean())
                user_inputs[col_name] = st.number_input(f"{col_name}", value=avg_val, key=col_name)

    if st.button("Generate Diagnostic Report & Graph"):
        if p_name:
            input_data = np.array([user_inputs[c] for c in feature_cols]).reshape(1, -1)
            rf_p = rf_model.predict_proba(input_data)[0][1] * 100
            lr_p = lr_model.predict_proba(input_data)[0][1] * 100
            dt_p = dt_model.predict_proba(input_data)[0][1] * 100

            # 1. Comparison Table
            st.subheader(f"📊 Model Analysis for {p_name}")
            res_df = pd.DataFrame({
                "Algorithm": ["Random Forest", "Logistic Regression", "Decision Tree"],
                "Risk Score (%)": [rf_p, lr_p, dt_p]
            })
            st.table(res_df)

            # 2. Patient Graph
            fig, ax = plt.subplots()
            ax.bar(res_df["Algorithm"], res_df["Risk Score (%)"], color=['red', 'blue', 'green'])
            ax.set_ylabel("Risk Percentage (%)")
            ax.set_title("Patient Heart Risk Comparison")
            st.pyplot(fig)

            # 3. Final Verdict
            if rf_p > 30:
                st.error(f"### ⚠️ HIGH RISK DETECTED: {rf_p:.1f}%")
                report_status = "High Risk"
            else:
                st.success(f"### ✅ NORMAL CONDITION: {rf_p:.1f}%")
                report_status = "Normal"

            # 4. Download Report Feature
            report_text = f"HEART DIAGNOSTIC REPORT\nPatient: {p_name}\nStatus: {report_status}\nRF Risk: {rf_p:.1f}%\nLR Risk: {lr_p:.1f}%\nDT Risk: {dt_p:.1f}%"
            st.download_button("📥 Download Patient Report", data=report_text, file_name=f"{p_name}_Report.txt")

else:
    st.error("Check CSV file on GitHub.")
