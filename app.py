import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.set_page_config(page_title="Medical Cost Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details below to predict your estimated medical charges based on our ML model.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Cost", type="primary"):

    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    
    input_df = pd.DataFrame(input_data)
    
    input_encoded = pd.get_dummies(input_df)
    
    model_columns = model.feature_names_in_ 
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)
    
    st.success(f"Estimated Insurance Cost: **€{prediction[0]:,.2f}**")
