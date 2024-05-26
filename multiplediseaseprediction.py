import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
)

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    st.subheader('Enter the following details:')
    
    diabetes_dataset = pd.read_csv("C:/Users/unais/Desktop/datasets/diabetes (1).csv")
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    with st.expander("Enter Patient Data"):
        coll, col2, col3 = st.columns(3)
        with coll:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
            SkinThickness = st.number_input('Skin Thickness value', min_value=0.0)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
        with col2:
            Glucose = st.number_input('Glucose Level', min_value=0.0)
            Insulin = st.number_input('Insulin Level', min_value=0.0)
            Age = st.number_input('Age of the Person', min_value=0, step=1)
        with col3:
            BloodPressure = st.number_input('Blood Pressure value', min_value=0.0)
            BMI = st.number_input('BMI value', min_value=0.0)

    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        with st.spinner('Predicting...'):
            diab_prediction = classifier.predict(std_data)
            if diab_prediction[0] == 1:
                st.error('The person is diabetic')
            else:
                st.success('The person is not diabetic.')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.subheader('Enter the following details:')

    heart_data = pd.read_csv("C:/Users/unais/Desktop/datasets/heart_disease_data (1).csv")
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    with st.expander("Enter Patient Data"):
        coll, col2, col3, col4 = st.columns(4)
        with coll:
            age = st.number_input('Age', min_value=0, step=1)
            sex = st.number_input('Sex (0 for female, 1 for male)', min_value=0, max_value=1)
            cp = st.number_input('Chest Pain Type', min_value=0, step=1)
            trestbps = st.number_input('Resting Blood Pressure', min_value=0.0)
        with col2:
            chol = st.number_input('Serum Cholesterol', min_value=0.0)
            fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1 for True, 0 for False)', min_value=0, max_value=1)
            restecg = st.number_input('Resting Electrocardiographic Results', min_value=0, step=1)
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0.0)
        with col3:
            exang = st.number_input('Exercise Induced Angina (1 for Yes, 0 for No)', min_value=0, max_value=1)
            oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0)
            slope = st.number_input('Slope of the Peak Exercise ST Segment', min_value=0, step=1)
            ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, step=1)
        with col4:
            thal = st.number_input('Thalassemia', min_value=0, step=1)

    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        with st.spinner('Predicting...'):
            prediction = model.predict(input_data_reshaped)
            if prediction[0] == 0:
                st.success('The Person does not have a Heart Disease')
            else:
                st.error('The Person has Heart Disease')
