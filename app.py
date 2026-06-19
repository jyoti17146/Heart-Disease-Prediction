import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 1rem 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #B91C1C;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 14px;
        border-radius: 6px;
        border: none;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background-color: #991B1B;
    }
    .gradient-title {
        text-align: center !important;
        font-weight: 800 !important;
        font-size: 80px !important;
        line-height: 1.1 !important;
        background: linear-gradient(90deg, #F87171 0%, #B91C1C 50%, #7F1D1D 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 10px 0 0 0 !important;
        padding: 0 !important;
    }
    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 18px;
        margin-bottom: 40px;
        margin-top: 0px;
    }
    .result-box {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
    }
    .result-title {
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-subtitle {
        font-weight: 400;
    }
    .card-header {
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 10px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(128,128,128,0.3);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .risk-factor-item {
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 14px;
        border-left: 3px solid;
    }
    .risk-high {
        background-color: rgba(239,68,68,0.08);
        border-left-color: #EF4444;
        color: #FCA5A5;
    }
    .risk-normal {
        background-color: rgba(34,197,94,0.08);
        border-left-color: #22C55E;
        color: #86EFAC;
    }

    .footer-container {
        margin-top: 50px;
        padding: 24px;
        border-radius: 10px;
        background-color: #1A1D24;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.2);
    }
    .footer-label {
        font-size: 13px;
        color: #9CA3AF;
        margin-bottom: 14px;
    }
    .team-links {
        display: flex;
        justify-content: center;
        gap: 24px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    .team-link {
        color: #E5E7EB;
        text-decoration: none;
        font-size: 15px;
        font-weight: 600;
        padding: 8px 18px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.3);
        transition: all 0.2s ease;
    }
    .team-link:hover {
        background-color: #B91C1C;
        color: white;
        border-color: #B91C1C;
    }
    .footer-copyright {
        font-size: 12px;
        color: #6B7280;
    }

    @media (max-width: 768px) {
        .gradient-title {
            font-size: 36px !important;
        }
        .subtitle {
            font-size: 14px !important;
            padding: 0 10px;
        }
        .main {
            padding: 0.5rem 1rem !important;
        }
        div.stButton > button {
            width: 100% !important;
        }
        .card-header {
            font-size: 13px;
        }
        .result-box {
            padding: 18px;
        }
    }
    </style>
""", unsafe_allow_html=True)

model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Session state defaults for form fields
defaults = {
    'age': 50, 'sex': 'Male', 'cp': 0, 'trestbps': 120, 'chol': 200,
    'fbs': 'No', 'restecg': 0, 'thalach': 150, 'exang': 'No',
    'oldpeak': 1.0, 'slope': 0, 'ca': 0, 'thal': 0
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def load_sample(sample_type):
    if sample_type == 'low_risk':
        st.session_state.update({
            'age': 35, 'sex': 'Female', 'cp': 0, 'trestbps': 110, 'chol': 180,
            'fbs': 'No', 'restecg': 0, 'thalach': 175, 'exang': 'No',
            'oldpeak': 0.2, 'slope': 0, 'ca': 0, 'thal': 0
        })
    else:
        st.session_state.update({
            'age': 65, 'sex': 'Male', 'cp': 3, 'trestbps': 165, 'chol': 310,
            'fbs': 'Yes', 'restecg': 2, 'thalach': 105, 'exang': 'Yes',
            'oldpeak': 3.8, 'slope': 2, 'ca': 3, 'thal': 2
        })

st.markdown('<div class="gradient-title">Heart Disease<br>Prediction System</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered tool to assess heart disease risk based on medical parameters</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About This App")
    st.write("""
    This application uses a Machine Learning model (K-Nearest Neighbors) 
    trained on the UCI Heart Disease Dataset to predict 
    the likelihood of heart disease.
    """)
    st.metric("Model Accuracy", "81.97%")
    st.divider()
    st.write("**Tech Stack**")
    st.write("Python · Scikit-learn · Streamlit")
    st.divider()
    st.caption("This tool is for educational purposes only and should not replace professional medical advice.")


col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<p class="card-header">Personal Information</p>', unsafe_allow_html=True)
        age = st.slider('Age', min_value=1, max_value=120, key='age')
        sex = st.radio('Sex', ['Male', 'Female'], horizontal=True, key='sex')
        cp = st.selectbox('Chest Pain Type', 
                           options=[0, 1, 2, 3],
                           format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x],
                           key='cp')
    
    with st.container(border=True):
        st.markdown('<p class="card-header">Vital Signs</p>', unsafe_allow_html=True)
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, key='trestbps')
        chol = st.slider('Cholesterol Level (mg/dl)', min_value=100, max_value=600, key='chol')
        thalach = st.slider('Max Heart Rate Achieved', min_value=60, max_value=220, key='thalach')

with col2:
    with st.container(border=True):
        st.markdown('<p class="card-header">Medical Test Results</p>', unsafe_allow_html=True)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], horizontal=True, key='fbs')
        restecg = st.selectbox('Resting ECG Result', 
                                options=[0, 1, 2],
                                format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x],
                                key='restecg')
        exang = st.radio('Exercise Induced Angina', ['No', 'Yes'], horizontal=True, key='exang')
    
    with st.container(border=True):
        st.markdown('<p class="card-header">Exercise Test Results</p>', unsafe_allow_html=True)
        oldpeak = st.slider('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, step=0.1, key='oldpeak')
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                              options=[0, 1, 2],
                              format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x],
                              key='slope')
        ca = st.selectbox('Number of Major Vessels', options=[0, 1, 2, 3, 4], key='ca')
        thal = st.selectbox('Thalassemia Type', 
                             options=[0, 1, 2, 3],
                             format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x],
                             key='thal')

sex_val = 1 if sex == 'Male' else 0
fbs_val = 1 if fbs == 'Yes' else 0
exang_val = 1 if exang == 'Yes' else 0

# Input validation warnings
warnings_list = []
if trestbps > 180:
    warnings_list.append("Resting blood pressure is unusually high (>180 mm Hg). Please double check this value.")
if chol > 450:
    warnings_list.append("Cholesterol level is unusually high (>450 mg/dl). Please double check this value.")
if thalach < 80 and age < 40:
    warnings_list.append("Max heart rate seems low for the given age. Please double check this value.")

if warnings_list:
    for w in warnings_list:
        st.warning(w)

st.write("")
st.write("")

button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
with button_col2:
    predict_btn = st.button('PREDICT HEART DISEASE RISK')

if predict_btn:
    
    input_data = (age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)
    is_high_risk = prediction[0] == 1
    
    st.write("")
    
    result_col1, result_col2, result_col3 = st.columns([1, 1.4, 1])
    with result_col2:
        if not is_high_risk:
            st.markdown("""
                <div class="result-box" style="background-color: #14291A; border: 2px solid #22C55E; box-shadow: 0 4px 20px rgba(34,197,94,0.15);">
                    <div style="font-size: 36px; margin-bottom: 8px;">&#10003;</div>
                    <div class="result-title" style="color: #4ADE80; font-size: 26px;">LOW RISK</div>
                    <div class="result-subtitle" style="color: #86EFAC; font-size: 15px; margin-top: 4px;">The person does not show signs of heart disease</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box" style="background-color: #2D1414; border: 2px solid #EF4444; box-shadow: 0 4px 20px rgba(239,68,68,0.15);">
                    <div style="font-size: 36px; margin-bottom: 8px;">&#9888;</div>
                    <div class="result-title" style="color: #F87171; font-size: 26px;">HIGH RISK</div>
                    <div class="result-subtitle" style="color: #FCA5A5; font-size: 15px; margin-top: 4px;">The person shows signs of heart disease. Please consult a doctor.</div>
                </div>
            """, unsafe_allow_html=True)

    # Risk Factors Explanation
    st.write("")
    factor_col1, factor_col2, factor_col3 = st.columns([1, 1.4, 1])
    with factor_col2:
        with st.container(border=True):
            st.markdown('<p class="card-header">Risk Factor Breakdown</p>', unsafe_allow_html=True)

            checks = [
                ("Age", age, age >= 55, "Age 55+ is associated with higher heart disease risk", f"{age} years"),
                ("Cholesterol", chol, chol >= 240, "Cholesterol above 240 mg/dl is considered high", f"{chol} mg/dl"),
                ("Resting Blood Pressure", trestbps, trestbps >= 140, "Blood pressure above 140 mm Hg is considered high", f"{trestbps} mm Hg"),
                ("Max Heart Rate", thalach, thalach < 120, "Lower max heart rate during exertion can indicate risk", f"{thalach} bpm"),
                ("Exercise Induced Angina", exang_val, exang_val == 1, "Chest pain during exercise is a risk indicator", exang),
                ("ST Depression (Oldpeak)", oldpeak, oldpeak >= 2.0, "Oldpeak above 2.0 suggests possible heart strain", f"{oldpeak}"),
                ("Major Vessels Affected", ca, ca >= 1, "Presence of blocked vessels increases risk", f"{ca}"),
            ]

            for label, value, is_risky, explanation, display_val in checks:
                if is_risky:
                    st.markdown(f"""
                        <div class="risk-factor-item risk-high">
                            <strong>{label}: {display_val}</strong> — {explanation}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="risk-factor-item risk-normal">
                            <strong>{label}: {display_val}</strong> — within normal range
                        </div>
                    """, unsafe_allow_html=True)

    # Download Report
    report_text = f"""HEART DISEASE PREDICTION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULT: {'HIGH RISK' if is_high_risk else 'LOW RISK'}

PATIENT DETAILS
----------------
Age: {age}
Sex: {sex}
Chest Pain Type: {['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][cp]}
Resting Blood Pressure: {trestbps} mm Hg
Cholesterol Level: {chol} mg/dl
Fasting Blood Sugar > 120 mg/dl: {fbs}
Resting ECG Result: {['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][restecg]}
Max Heart Rate Achieved: {thalach} bpm
Exercise Induced Angina: {exang}
ST Depression (Oldpeak): {oldpeak}
Slope of Peak Exercise ST Segment: {['Upsloping', 'Flat', 'Downsloping'][slope]}
Number of Major Vessels: {ca}
Thalassemia Type: {['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][thal]}

DISCLAIMER
----------
This report is generated by an AI model for educational purposes only.
It should not be used as a substitute for professional medical advice.
Please consult a qualified doctor for an accurate diagnosis.
"""

    dl_col1, dl_col2, dl_col3 = st.columns([1, 1.4, 1])
    with dl_col2:
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Footer
st.markdown("""
    <div class="footer-container">
        <div class="footer-label">Designed and Developed by</div>
        <div class="team-links">
            <a href="https://www.linkedin.com/in/jyoti-pandey-6460a32a7" target="_blank" class="team-link">Jyoti Pandey</a>
            <a href="https://www.linkedin.com/in/riya-kumari-7b152030b" target="_blank" class="team-link">Riya Kumari</a>
            <a href="https://www.linkedin.com/in/tanishk-sidharth-61b811272" target="_blank" class="team-link">Tanishk Sidharth</a>
            <a href="https://www.linkedin.com/in/bhanu-pratap-251976393" target="_blank" class="team-link">Bhanu Pratap</a>
        </div>
        <div class="footer-copyright">Copyright &copy; 2026. All Rights Reserved.</div>
    </div>
""", unsafe_allow_html=True)
