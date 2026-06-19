import streamlit as st
import pickle
import numpy as np

# Loading the trained model
model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Setting page title
st.title('Heart Disease Prediction App 🫀')
st.write('Enter the patient details below to check heart disease risk')

# Creating input fields for all features
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
chol = st.number_input('Cholesterol Level', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.selectbox('Resting ECG Result (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia Type (0-3)', [0, 1, 2, 3])

# Converting categorical inputs to numerical
sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'Yes' else 0
exang = 1 if exang == 'Yes' else 0

# Predict button
if st.button('Predict'):
    
    # Preparing the input data
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Making prediction
    prediction = model.predict(input_data_reshaped)
    
    # Displaying result
    if prediction[0] == 0:
        st.success('✅ The Person does NOT have Heart Disease')
    else:
        st.error('⚠️ The Person HAS Heart Disease')