import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import os
from sklearn.ensemble import RandomForestClassifier

# --- MODEL TRAINING (SAFE MODE) ---
@st.cache_resource
def train_fresh_model():
    if os.path.exists("heart_disease.csv"):
        df = pd.read_csv("heart_disease.csv")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]
        target_col = [c for c in df.columns if c.lower() in ['target', 'outcome']][0] if any(c.lower() in ['target', 'outcome'] for c in df.columns) else df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        model.feature_names_in_ = list(X.columns)
        return model
    return None

model = train_fresh_model()

# --- PROFESSIONAL PDF REPORT FUNCTION ---
def create_advanced_report(name, age, res, prob, advice, diet, meds, graph_buf):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 51, 102) # Dark Blue
    pdf.cell(200, 20, "HEART HEALTH ANALYSIS REPORT", ln=True, align='C')
    
    # Patient Info Table
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient Name: {name}", border=0)
    pdf.cell(100, 10, f"Age: {age}", ln=True, border=0)
    pdf.line(10, 40, 200, 40) # Horizontal line
    pdf.ln(5)

    # Diagnosis Results
    pdf.set_font("Arial", 'B', 14)
    status_color = (200, 0, 0) if "HIGH" in res else (0, 128, 0)
    pdf.set_text_color(*status_color)
    pdf.cell(200, 10, f"Health Status: {res}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Risk Probability: {prob}%", ln=True)

    # Graph Section
    temp_graph = "temp_report_graph.png"
    with open(temp_graph, "wb") as f:
        f.write(graph_buf.getbuffer())
    pdf.image(temp_graph, x=50, y=75, w=110)
    pdf.ln(75) # Space for graph

    # Medical Analysis & Suggestions
    sections = [
        ("Medical Analysis & Suggestions:", advice),
        ("Dietary Recommendations:", diet),
        ("Lifestyle & Precautions:", meds)
    ]

    for title, items in sections:
        pdf.set_font("Arial", 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.set_font("Arial", size=11)
        for item in items:
            pdf.multi_cell(0, 8, f"- {item}")
        pdf.ln(3)

    if os.path.exists(temp_graph): os.remove(temp_graph)
    return pdf.output(dest="S").encode("latin-1")

st.set_page_config(page_title="Heart AI Pro", layout="wide")
st.title("🏥 Professional Heart Diagnostic Center")

p_name = st.text_input("Patient Full Name", "Guest User")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 500, 200)
    exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    family = st.selectbox("Family History", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    hbp = st.selectbox("High BP History", ["No", "Yes"])

with col2:
    lhdl = st.selectbox("Low HDL", ["No", "Yes"])
    hldl = st.selectbox("High LDL", ["No", "Yes"])
    alc = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    sleep = st.number_input("Sleep Hours", 1, 15, 7)
    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    tri = st.number_input("Triglycerides", 50, 500, 150)
    fbs = st.number_input("Fasting Sugar", 50, 300, 100)
    crp = st.number_input("CRP Level", 0.0, 10.0, 1.0)
    homo = st.number_input("Homocysteine", 0, 50, 15)

if st.button("Generate Complete Report"):
    if model is None:
        st.error("CSV file not found!")
    else:
        m = {"Male":1,"Female":0,"Yes":1,"No":0,"Low":0,"Medium":1,"High":2}
        data = [[age, m.get(gender,0), bp, chol, m.get(exercise,0), m.get(smoke,0), m.get(family,0), m.get(diabetes,0), bmi, m.get(hbp,0), m.get(lhdl,0), m.get(hldl,0), m.get(alc,0), m.get(stress,0), sleep, m.get(sugar,0), tri, fbs, crp, homo]]
        df_in = pd.DataFrame(data, columns=model.feature_names_in_)
        
        pred = model.predict(df_in)[0]
        prob = model.predict_proba(df_in)[0][1] * 100
        
        if pred == 1:
            res, color = "HIGH RISK / HEART DISEASE DETECTED", "red"
            advice = ["Possibility of arterial blockage.", "Risk of cardiac arrest or stroke."]
            diet = ["Avoid Trans-fats and high salt.", "Include Omega-3 (Walnuts, Fish)."]
            meds = ["Consult a cardiologist immediately.", "Monitor BP every 4 hours."]
        else:
            res, color = "NORMAL / NO RISK", "green"
            advice = ["Heart condition appears stable.", "No immediate risk detected."]
            diet = ["Maintain fiber-rich diet.", "Continue regular water intake."]
            meds = ["Annual check-up is sufficient.", "Regular morning walks advised."]

        st.markdown(f"### Result: {res}")
        
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(['Risk Score'], [prob], color=color)
        ax.set_xlim(0, 100)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        st.pyplot(fig)
        
        pdf = create_advanced_report(p_name, age, res, round(prob,2), advice, diet, meds, buf)
        st.download_button(f"📥 Download Full Report for {p_name}", pdf, f"{p_name}_Heart_Report.pdf", "application/pdf")
