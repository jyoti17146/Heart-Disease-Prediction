import streamlit as st
import pickle
import numpy as np

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
    </style>
""", unsafe_allow_html=True)

model = pickle.load(open('heart_disease_model.sav', 'rb'))

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
        age = st.slider('Age', min_value=1, max_value=120, value=50)
        sex = st.radio('Sex', ['Male', 'Female'], horizontal=True)
        cp = st.selectbox('Chest Pain Type', 
                           options=[0, 1, 2, 3],
                           format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
    
    with st.container(border=True):
        st.markdown('<p class="card-header">Vital Signs</p>', unsafe_allow_html=True)
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
        chol = st.slider('Cholesterol Level (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.slider('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)

with col2:
    with st.container(border=True):
        st.markdown('<p class="card-header">Medical Test Results</p>', unsafe_allow_html=True)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'], horizontal=True)
        restecg = st.selectbox('Resting ECG Result', 
                                options=[0, 1, 2],
                                format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
        exang = st.radio('Exercise Induced Angina', ['No', 'Yes'], horizontal=True)
    
    with st.container(border=True):
        st.markdown('<p class="card-header">Exercise Test Results</p>', unsafe_allow_html=True)
        oldpeak = st.slider('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                              options=[0, 1, 2],
                              format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        ca = st.selectbox('Number of Major Vessels', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia Type', 
                             options=[0, 1, 2, 3],
                             format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])

sex_val = 1 if sex == 'Male' else 0
fbs_val = 1 if fbs == 'Yes' else 0
exang_val = 1 if exang == 'Yes' else 0

st.write("")
st.write("")

st.markdown("""
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        width: 350px !important;
    }
    </style>
""", unsafe_allow_html=True)

predict_btn = st.button('PREDICT HEART DISEASE RISK')

if predict_btn:
    
    input_data = (age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)
    
    st.write("")
    
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    with result_col2:
        if prediction[0] == 0:
            st.markdown("""
                <div class="result-box" style="background-color: #14291A; border: 1px solid #15803D;">
                    <div class="result-title" style="color: #4ADE80;">LOW RISK</div>
                    <div class="result-subtitle" style="color: #86EFAC;">The person does not show signs of heart disease</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box" style="background-color: #2D1414; border: 1px solid #B91C1C;">
                    <div class="result-title" style="color: #F87171;">HIGH RISK</div>
                    <div class="result-subtitle" style="color: #FCA5A5;">The person shows signs of heart disease. Please consult a doctor.</div>
                </div>
            """, unsafe_allow_html=True)
