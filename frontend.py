import streamlit as st
import requests

API_URL = "http://backend:5000/predict"

st.title("Vehicle Insurance Predictor")

Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age", 18, 100, 30)
Driving_License = st.selectbox("Driving License (0: 'No' , 1: 'Yes')", [0, 1])
Region_Code = st.selectbox("Region Code", list(range(0,53)))
Previously_Insured = st.selectbox("Previously Insured (0: 'No' , 1: 'Yes')", [0, 1])
Vehicle_Age = st.selectbox("Vehicle Age", ["< 1 Year","1-2 Year","> 2 Years"])
Vehicle_Damage = st.selectbox("Vehicle Damage", ["Yes","No"])
Annual_Premium = st.number_input("Annual Premium", 0.0)
Policy_Sales_Channel = st.selectbox("Region Code", list(range(0,165)))
Vintage = st.number_input("Vintage",0)

if st.button("Predict Insurance Requirement Category"):

    payload = {
        "Gender": Gender,
        "Age": Age,
        "Driving_License": Driving_License,
        "Region_Code": Region_Code,
        "Previously_Insured": Previously_Insured,
        "Vehicle_Age": Vehicle_Age,
        "Vehicle_Damage": Vehicle_Damage,
        "Annual_Premium": Annual_Premium,
        "Policy_Sales_Channel": Policy_Sales_Channel,
        "Vintage": Vintage
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        if response.status_code == 200:

            st.success(
                f"Predicted Category :  **{result['prediction']}**"
            )

            st.write("🔍 Confidence:", result["confidence"])

            st.write("📊 Class Probabilities:")
            st.json(result["class_probabilities"])

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server.")