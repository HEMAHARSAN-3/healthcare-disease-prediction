import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="CardioAI",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("../models/heart_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ==========================================
# CLEAN WHITE MODERN CSS
# ==========================================
st.markdown("""
<style>

/* Remove default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Global font */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Header Bar */
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 0;
}

.logo {
    font-size:38px;
    font-weight:700;
    color:#87CEEB;
}


.subtitle {
    color:#6B7280;
    font-size:14px;
}

/* Card */
.card {
    background:transparent;
    border:none;
    box-shadow:none;
}

/* Button */
.stButton>button {
    background-color:#111827;
    color:white;
    border-radius:8px;
    height:45px;
    font-weight:600;
    border:none;
}

.stButton>button:hover {
    background-color:#1F2937;
}

/* Risk Text */
.high {
    color:#DC2626;
    font-weight:700;
    font-size:22px;
}

.low {
    color:#059669;
    font-weight:700;
    font-size:22px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TOP BAR
# ==========================================
st.markdown("""
<div class="topbar">
<div class="logo">🩺 CardioAI</div>
<div class="subtitle">Clinical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
"""
### Predict Heart Disease Risk Using AI  
A machine learning system designed to assist in early cardiovascular risk assessment.
"""
)

st.write("")

# ==========================================
# MAIN LAYOUT
# ==========================================
col1, col2 = st.columns([1,1])

# ------------------------------------------
# LEFT — PATIENT FORM
# ------------------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Information")

    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex", [0,1], format_func=lambda x:"Female" if x==0 else "Male")
    cp = st.slider("Chest Pain Type", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 220, 120)
    chol = st.number_input("Cholesterol", 100, 500, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    restecg = st.slider("Rest ECG", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.slider("Slope", 0, 2, 1)
    ca = st.slider("Major Vessels", 0, 4, 0)
    thal = st.slider("Thalassemia", 0, 3, 1)

    predict_btn = st.button("Run Risk Assessment")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# RIGHT — RESULT PANEL
# ------------------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if predict_btn:

        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach,
                                exang, oldpeak, slope,
                                ca, thal]])

        columns = [
            "age","sex","cp","trestbps","chol",
            "fbs","restecg","thalach",
            "exang","oldpeak","slope",
            "ca","thal"
        ]

        df = pd.DataFrame(input_data, columns=columns)
        scaled = scaler.transform(df)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        if prediction == 1:
            st.markdown('<p class="high">High Risk of Heart Disease</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="low">Low Risk of Heart Disease</p>', unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text':"Risk Probability %"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"#111827"},
                'steps':[
                    {'range':[0,40],'color':"#D1FAE5"},
                    {'range':[40,70],'color':"#FEF3C7"},
                    {'range':[70,100],'color':"#FEE2E2"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Enter patient information and click **Run Risk Assessment** to generate prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.caption("This AI system is for screening purposes only and does not replace professional medical advice.")