import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import os
from sklearn.ensemble import RandomForestClassifier

# --- MODEL TRAINING LOGIC ---
@st.cache_resource
def load_or_train_model():
    try:
        # Pehle check karein agar purana model kaam kar raha hai
        if os.path.exists("heart_model.pkl"):
            with open("heart_model.pkl", "rb") as f:
                model = pickle.load(f)
            return model
    except:
        pass
    
    # Agar model nahi hai ya fail hua, toh CSV se train karein
    if os.path.exists("heart_disease.csv"):
        df = pd.read_csv("heart_disease.csv")
        
        # Target column dhoondne ka smart tarika (Target, target, ya aakhri column)
        if 'Target' in df.columns:
            target_col = 'Target'
        elif 'target' in df.columns:
            target_col = 'target'
        else:
            target_col = df.columns[-1] # Sabse aakhri column ko target maan lo
            
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    return None

model = load_or_train_model()

def create_advanced_report(name, age, res, prob, advice, diet, meds, graph_buf):
    pdf = FPDF()
    pdf.add_page()
    temp_graph = "temp_graph.png"
    with open(temp_graph, "wb") as f:
        f.write(graph_buf.getbuffer())
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(200, 20, "ADVANCED HEART HEALTH REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Patient Name: {name} | Age: {age}", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Diagnosis: {res} ({prob}%)", ln=True)
    pdf.image(temp_graph, x=55, y=60, w=100)
    pdf.ln(65) 
    sections = [("Future Risks:", advice), ("Dietary Plans:", diet), ("Medical Precautions:", meds)]
    for title, items in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, title, ln=True)
        pdf.set_font("Arial", size=10)
        for item in items:
            pdf.multi_cell(0, 8, f"- {item}")
        pdf.ln(3)
    if os.path.exists(temp_graph): os.remove(temp_graph)
    return pdf.output(dest="S").encode("latin-1")

st.set_page_config(page_title="Heart AI Pro", layout="wide")
st.title("🏥 Professional Heart Diagnostic Center")
p_name = st.text_input("Patient Full Name", "Guest User")

st.subheader("Patient Clinical Data")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Blood Pressure", value=120.0)
    chol = st.number_input("Cholesterol Level", value=200.0)
    exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    family = st.selectbox("Family Heart Disease", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    bmi = st.number_input("BMI", value=25.0)
    hbp = st.selectbox("High Blood Pressure History", ["No", "Yes"])
with col2:
    lhdl = st.selectbox("Low HDL Cholesterol", ["No", "Yes"])
    hldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
    alc = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    sleep = st.number_input("Sleep Hours", value=7.0)
    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    tri = st.number_input("Triglyceride Level", value=150.0)
    fbs = st.number_input("Fasting Blood Sugar", value=100.0)
    crp = st.number_input("CRP Level", value=2.0)
    homo = st.number_input("Homocysteine Level", value=15.0)

if st.button("Generate Complete Report"):
    if model is None:
        st.error("Model train nahi ho paya. Please check heart_disease.csv file.")
    else:
        mapping = {"Male": 1, "Female": 0, "Yes": 1, "No": 0, "Low": 0, "Medium": 1, "High": 2}
        input_data = [[age, mapping[gender], bp, chol, mapping[exercise], mapping[smoke], mapping[family], mapping[diabetes], bmi, mapping[hbp], mapping[lhdl], mapping[hldl], mapping[alc], mapping[stress], sleep, mapping[sugar], tri, fbs, crp, homo]]
        
        # Columns match karna zaroori hai
        input_df = pd.DataFrame(input_data, columns=model.feature_names_in_)
        
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100
        
        res_text, color = ("HIGH RISK DETECTED", "red") if pred == 1 else ("LOW RISK / NORMAL", "green")
        advice = ["Possibility of Arterial Blockage."] if pred == 1 else ["Heart condition stable."]
        diet = ["Avoid Trans-fats."] if pred == 1 else ["Maintain fiber intake."]
        meds = ["Consult doctor."] if pred == 1 else ["Annual check-up."]

        st.markdown(f"### Result: <span style='color:{color}'>{res_text}</span>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(['Risk Score'], [prob], color=color)
        ax.set_xlim(0, 100)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.pyplot(fig)
        pdf_bytes = create_advanced_report(p_name, age, res_text, round(prob, 2), advice, diet, meds, buf)
        st.download_button(f"📥 Download Report", pdf_bytes, f"Report.pdf", "application/pdf")
