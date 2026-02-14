import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------- TRAIN MODEL INSIDE APP (NO PICKLE) --------

np.random.seed(42)
n = 500

rainfall = np.random.randint(0, 400, n)
temperature = np.random.randint(20, 45, n)
humidity = np.random.randint(30, 100, n)
river_level = np.random.randint(1, 15, n)
wind_speed = np.random.randint(0, 120, n)

risk = []
for i in range(n):
    if rainfall[i] > 250 or river_level[i] > 10 or wind_speed[i] > 90:
        risk.append(2)
    elif rainfall[i] > 120 or humidity[i] > 75:
        risk.append(1)
    else:
        risk.append(0)

df = pd.DataFrame({
    "rainfall": rainfall,
    "temperature": temperature,
    "humidity": humidity,
    "river_level": river_level,
    "wind_speed": wind_speed,
    "risk": risk
})

X = df.drop("risk", axis=1)
y = df["risk"]

model = RandomForestClassifier(n_estimators=150)
model.fit(X, y)

# -------- STREAMLIT UI --------

st.set_page_config(page_title="AI Disaster Alert System", layout="centered")

st.title("🌍 RiskAlert System")
st.markdown("### Predict Disaster Risk")

# USER DETAILS
st.subheader("👤 User Information")
name = st.text_input("Enter Your Name")
location = st.text_input("Enter Your Location")

st.subheader("🌦 Environmental Parameters")

rainfall = st.slider("Rainfall (mm)", 0, 400, 50)
temperature = st.slider("Temperature (°C)", 10, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 60)
river_level = st.slider("River Level (m)", 0, 15, 5)
wind_speed = st.slider("Wind Speed (km/h)", 0, 120, 20)

if st.button("🔍 Predict Disaster Risk"):

    input_data = np.array([[rainfall, temperature, humidity, river_level, wind_speed]])
    prediction = model.predict(input_data)[0]

    st.subheader("📢 Prediction Result")

    if prediction == 0:
        st.success("🟢 LOW RISK")
        st.info("Stay safe. No immediate danger detected.")

    elif prediction == 1:
        st.warning("🟡 MEDIUM RISK")
        st.info("Be alert. Monitor weather updates and stay prepared.")

    else:
        st.error("🔴 HIGH RISK")
        st.warning("⚠ Immediate Safety Measures Required!")
        st.markdown("""
        **Safety Tips:**
        - Move to higher ground
        - Avoid flood areas
        - Keep emergency kit ready
        - Follow government alerts
        """)

    if name and location:
        st.write(f"Stay safe **{name}** from **{location}** ❤️")
