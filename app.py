import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trained model
model = joblib.load("flood_model.pkl")

st.title("🌊 Flood Prediction Web App")
st.write("Enter values to predict flood occurrence and risk level:")

# Input fields
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=5000, value=100)
river_level = st.number_input("River Level (m)", min_value=0.0, max_value=1000.0, value=5.0)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=1000.0, value=30.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=500.0, value=25.0)
population_density = st.number_input("Population Density", min_value=0, max_value=10000, value=500)

# Prediction button
if st.button("Predict Flood"):
    features = np.array([[rainfall, river_level, soil_moisture, temperature, population_density]])
    
    # Predict class and probability
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1] * 100   # Probability of flood
    
    # Show probability
    st.subheader(f"🌡️ Flood Likelihood: {prob:.2f}%")
    
    # Severity levels
    if prob > 80:
        st.error("🚨 Severe Flood Risk!")
        st.info("💡 Recommendation: Immediate precautions required. Stay updated with disaster alerts.")
    elif prob > 50:
        st.warning("⚠️ Moderate Flood Risk")
        st.info("💡 Recommendation: Stay alert, monitor rainfall & river level updates.")
    else:
        st.success("✅ Safe (Low Risk)")
        st.info("💡 Recommendation: Situation is safe, but keep monitoring weather reports.")
    
    # Visualization of input values
    st.subheader("📊 Your Input Values")
    fig, ax = plt.subplots()
    labels = ["Rainfall", "River Level", "Soil Moisture", "Temperature", "Population Density"]
    ax.bar(labels, features[0], color="skyblue")
    plt.xticks(rotation=30)
    plt.ylabel("Value")
    st.pyplot(fig)

    # Comparison with average flood-causing conditions
    st.subheader("⚖️ Comparison with Typical Flood Conditions")
    
    # Example average flood-causing values (you can adjust based on dataset insights)
    avg_flood_values = [250, 12, 60, 30, 3000]  
    
    df_compare = pd.DataFrame({
        "Feature": labels,
        "Your Input": features[0],
        "Flood Avg": avg_flood_values
    })
    
    fig2, ax2 = plt.subplots()
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, df_compare["Your Input"], 0.4, label="Your Input", color="skyblue")
    ax2.bar(x + 0.2, df_compare["Flood Avg"], 0.4, label="Flood Avg", color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30)
    ax2.set_ylabel("Value")
    ax2.legend()
    st.pyplot(fig2)
