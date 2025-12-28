import streamlit as st
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import os
from sklearn.ensemble import RandomForestClassifier

# --- DIRECT MODEL TRAINING (No Pickle, No Error) ---
@st.cache_resource
def train_fresh_model():
    if os.path.exists("heart_disease.csv"):
        df = pd.read_csv("heart_disease.csv")
        # Target column pehchanna
        target_col = [c for c in df.columns if c.lower() in ['target', 'outcome']][0] if any(c.lower() in ['target', 'outcome'] for c in df.columns) else df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        model.feature_names_in_ = list(X.columns)
        return model
    return None

model = train_fresh_model()

def create_advanced_report(name, age, res, prob, advice, diet, meds, graph_buf):
    pdf = FPDF()
    pdf.add_page()
    temp_graph = "temp_report_graph.png"
    with open(temp_graph, "wb") as f:
        f.write(graph_buf.getbuffer())
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 20, "HEART HEALTH ANALYSIS REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient: {name} | Age: {age}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Result: {res} ({prob}%)", ln=True)
    pdf.image(temp_graph, x=50, y=70, w=110)
    pdf.ln(80)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, f"Advice: {advice[0]}\nDiet: {diet[0]}\nMeds: {meds[0]}")
    if os.path.exists(temp_graph): os.remove(temp_graph)
    return pdf.output(dest="S").encode("latin-1")

st.set_page_config(page_title="Heart AI", layout="wide")
st.title("🏥 Smart Heart Diagnostic Center")

p_name = st.text_input("Patient Name", "User")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 500, 200)
    exercise = st.selectbox("Exercise", ["Low", "Medium", "High"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    family = st.selectbox("Family History", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    hbp = st.selectbox("High BP History", ["No", "Yes"])

with col2:
    lhdl = st.selectbox("Low HDL", ["No", "Yes"])
    hldl = st.selectbox("High LDL", ["No", "Yes"])
    alc = st.selectbox("Alcohol", ["Low", "Medium", "High"])
    stress = st.selectbox("Stress", ["Low", "Medium", "High"])
    sleep = st.number_input("Sleep Hours", 1, 15, 7)
    sugar = st.selectbox("Sugar Level", ["Low", "Medium", "High"])
    tri = st.number_input("Triglycerides", 50, 500, 150)
    fbs = st.number_input("Fasting Sugar", 50, 300, 100)
    crp = st.number_input("CRP Level", 0.0, 10.0, 1.0)
    homo = st.number_input("Homocysteine", 0, 50, 15)

if st.button("Generate Report"):
    if model is None:
        st.error("CSV file missing on GitHub!")
    else:
        m = {"Male":1,"Female":0,"Yes":1,"No":0,"Low":0,"Medium":1,"High":2}
        data = [[age, m[gender], bp, chol, m[exercise], m[smoke], m[family], m[diabetes], bmi, m[hbp], m[lhdl], m[hldl], m[alc], m[stress], sleep, m[sugar], tri, fbs, crp, homo]]
        df_in = pd.DataFrame(data, columns=model.feature_names_in_)
        
        pred = model.predict(df_in)[0]
        prob = model.predict_proba(df_in)[0][1] * 100
        
        res, color = ("HIGH RISK", "red") if pred == 1 else ("NORMAL", "green")
        st.subheader(f"Status: {res}")
        
        fig, ax = plt.subplots(figsize=(6,2))
        ax.barh(['Risk'], [prob], color=color)
        ax.set_xlim(0, 100)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        st.pyplot(fig)
        
        pdf = create_advanced_report(p_name, age, res, round(prob,2), ["Follow doctor advice"], ["Eat greens"], ["Check regularly"], buf)
        st.download_button("📥 Download Report", pdf, "Heart_Report.pdf", "application/pdf")
