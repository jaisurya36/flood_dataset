try:
    import joblib
    print("joblib imported successfully")
except ModuleNotFoundError:
    print("joblib is missing")

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("flood_model.pkl")

st.title("🌊 Flood Prediction Web App")
st.write("Enter values to predict flood likelihood:")

# User inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0)
river_level = st.number_input("River Level (m)", min_value=0.0, format="%.2f")
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, format="%.2f")
temperature = st.number_input("Temperature (°C)", min_value=-10.0, format="%.1f")
population_density = st.number_input("Population Density", min_value=0)

if st.button("Predict Flood"):
    input_data = np.array([[rainfall, river_level, soil_moisture, temperature, population_density]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 Flood Likely!")
    else:
        st.success("✅ No Flood Expected")
