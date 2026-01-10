import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# Page Config
st.set_page_config(page_title="Heart AI Pro: 3-Model System", layout="wide")

# 1. Dataset Loading & Preprocessing
@st.cache_data
def load_data():
    # Aapki file ka naam yaha check kar lein
    df = pd.read_csv("heart_disease.csv") 
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    return df

df = load_data()
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Training 3 Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dt_model = DecisionTreeClassifier().fit(X_train, y_train)

# UI Header
st.title("❤️ Heart Disease Diagnostic Center (3-Model Comparison)")
st.write("Professional AI Analysis using Random Forest, Logistic Regression, and Decision Tree")

# 3. Input Fields (2 columns)
col1, col2 = st.columns(2)
inputs = {}
for i, column in enumerate(X.columns):
    with col1 if i % 2 == 0 else col2:
        inputs[column] = st.number_input(f"Enter {column}", value=float(X[column].mean()))

if st.button("Analyze Heart Health"):
    input_df = pd.DataFrame([inputs])
    
    # Predictions
    rf_prob = rf_model.predict_proba(input_df)[0][1]
    lr_prob = lr_model.predict_proba(input_df)[0][1]
    dt_prob = dt_model.predict_proba(input_df)[0][1]

    # Comparison Table
    st.subheader("📊 Algorithm Comparison Results")
    results_df = pd.DataFrame({
        "Algorithm": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Risk Probability": [f"{rf_prob*100:.1f}%", f"{lr_prob*100:.1f}%", f"{dt_prob*100:.1f}%"],
        "Status": ["High Risk" if p > 0.35 else "Normal" for p in [rf_prob, lr_prob, dt_prob]]
    })
    st.table(results_df)

    # Main Analysis (Based on Best Model - Random Forest)
    final_risk = rf_prob * 100
    st.divider()
    if rf_prob > 0.35: # Threshold lowered for better sensitivity
        st.error(f"⚠️ HIGH RISK DETECTED: {final_risk:.1f}%")
    else:
        st.success(f"✅ NORMAL / NO RISK: {final_risk:.1f}%")

    # Risk Graph
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(["Risk Score"], [final_risk], color='red' if final_risk > 35 else 'green')
    ax.set_xlim(0, 100)
    st.pyplot(fig)
