import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart AI Pro", layout="wide")

# 1. Dataset Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart_disease.csv")
        df.columns = df.columns.str.strip()
        # Drop rows with missing target
        target_options = ['Target', 'target', 'heart_disease', 'output']
        target_col = next((c for c in target_options if c in df.columns), df.columns[-1])
        df = df.dropna(subset=[target_col])
        
        # Simple Numeric Conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]
        
        # Fill missing values with 0
        df = df.fillna(0)
        return df, target_col
    except Exception as e:
        st.error(f"Dataset Load Error: {e}")
        return None, None

df, target_column = load_data()

if df is not None:
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Train Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    dt = DecisionTreeClassifier().fit(X, y)

    st.title("❤️ Heart Disease Diagnostic Center")
    p_name = st.text_input("Patient Name")
    
    st.subheader("Enter Details")
    user_inputs = []
    cols = st.columns(3)
    for i, col_name in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(f"{col_name}", value=float(X[col_name].mean()))
            user_inputs.append(val)

    if st.button("Predict"):
        if p_name:
            input_data = np.array(user_inputs).reshape(1, -1)
            rf_res = rf.predict_proba(input_data)[0][1]
            lr_res = lr.predict_proba(input_data)[0][1]
            dt_res = dt.predict_proba(input_data)[0][1]
            
            st.write(f"### Report for: {p_name}")
            
            # Comparison Table
            res_df = pd.DataFrame({
                "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
                "Risk Probability": [f"{rf_res*100:.1f}%", f"{lr_res*100:.1f}%", f"{dt_res*100:.1f}%"]
            })
            st.table(res_df)
            
            # Final Verdict (Threshold 0.35 for high sensitivity)
            if rf_res > 0.35:
                st.error(f"Result: HIGH RISK ({rf_res*100:.1f}%)")
            else:
                st.success(f"Result: NORMAL ({rf_res*100:.1f}%)")
        else:
            st.warning("Please enter patient name")
else:
    st.error("Check if 'heart_disease.csv' is in your GitHub folder.")
