# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 19:02:00 2025

@author: USER
"""

import os
import numpy as np
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Get current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the diabetes model from saved_model/trained_model.sav
model_path = os.path.join(working_dir, 'saved_model', 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Streamlit App UI
def main():
    st.title('🧑‍⚕️ Diabetes Prediction Web App')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI Value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

    with col2:
        Age = st.text_input('Age')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(user_input)
        except ValueError:
            diagnosis = 'Please enter valid numeric values in all fields.'

    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()
