import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart AI Pro", layout="wide")

# 1. Dataset Loading with Error Handling
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("heart_disease.csv")
        df.columns = df.columns.str.strip() # Remove spaces
        
        # Target column dhundhna
        target_options = ['Target', 'target', 'heart_disease', 'output', 'Condition']
        target_col = next((c for c in target_options if c in df.columns), df.columns[-1])
        
        # Data ko numeric banana training ke liye
        df_num = df.copy()
        for col in df_num.columns:
            if df_num[col].dtype == 'object':
                df_num[col] = pd.factorize(df_num[col])[0]
        
        imputer = SimpleImputer(strategy='mean')
        df_clean = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
        return df_clean, target_col, list(df_num.columns)
    except Exception as e:
        return None, None, None

df, target_column, all_columns = load_and_clean_data()

if df is not None:
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training 3 Models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    lr_model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

    st.title("❤️ Heart Disease Diagnostic Center")
    patient_name = st.text_input("👤 Enter Patient Full Name")
    
    st.subheader("📋 Patient Health Parameters")
    col1, col2, col3 = st.columns(3)
    user_inputs = {}

    # Har column ke liye input box banana (Agar column CSV mein hai tabhi)
    cols_in_data = list(X.columns)
    
    for i, col in enumerate(cols_in_data):
        target_col_box = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
        with target_col_box:
            if "Gender" in col:
                val = st.selectbox(col, ["Male", "Female"])
                user_inputs[col] = 1 if val == "Male" else 0
            elif any(x in col for x in ["Smoking", "Alcohol", "Diabetes", "Family", "High"]):
                val = st.selectbox(col, ["No", "Yes"])
                user_inputs[col] = 1 if val == "Yes" else 0
            elif "Stress" in col:
                val = st.selectbox(col, ["Low", "Medium", "High"])
                user_inputs[col] = {"Low":0, "Medium":1, "High":2}[val]
            else:
                avg = float(df[col].mean())
                user_inputs[col] = st.number_input(col, value=avg)

    if st.button("Run Multi-Model Analysis"):
        if not patient_name:
            st.error("Please enter patient name.")
        else:
            # Sahi order mein data lagana
            input_list = [user_inputs[c] for c in cols_in_data]
            input_array = np.array(input_list).reshape(1, -1)
            
            # Predictions
            rf_p = rf_model.predict_proba(input_array)[0][1]
            lr_p = lr_model.predict_proba(input_array)[0][1]
            dt_p = dt_model.predict_proba(input_array)[0][1]

            st.subheader(f"📊 Report: {patient_name}")
            results = pd.DataFrame({
                "Algorithm": ["Random Forest", "Logistic Regression", "Decision Tree"],
                "Risk %": [f"{rf_p*100:.1f}%", f"{lr_p*100:.1f}%", f"{dt_p*100:.1f}%"],
                "Status": ["High Risk" if p > 0.35 else "Normal" for p in [rf_p, lr_p, dt_p]]
            })
            st.table(results)

            if rf_p > 0.35:
                st.error(f"### ⚠️ HIGH RISK: {rf_p*100:.1f}%")
            else:
                st.success(f"### ✅ NORMAL: {rf_p*100:.1f}%")

            fig, ax = plt.subplots(figsize=(10, 1))
            ax.barh(["Risk Meter"], [rf_p*100], color='red' if rf_p > 0.35 else 'green')
            ax.set_xlim(0, 100)
            st.pyplot(fig)
else:
    st.error("CSV file not found or corrupted. Please check heart_disease.csv in GitHub.")
