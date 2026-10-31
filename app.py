import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.set_page_config(page_title="Medical Cost Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details below to predict your estimated medical charges based on our ML model.")

# Create columns for the layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# When the button is clicked
if st.button("Predict Cost", type="primary"):
    # 1. Create a dictionary with the user inputs
    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # 3. Apply the exact same get_dummies to the input
    input_encoded = pd.get_dummies(input_df)
    
    # 4. Reindex to match the model's expected columns
    # (This handles missing dummy columns based on the specific region/sex chosen)
    model_columns = model.feature_names_in_ 
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 5. Predict
    prediction = model.predict(input_encoded)
    
    # 6. Display result
    st.success(f"Estimated Insurance Cost: **€{prediction[0]:,.2f}**")