
import streamlit as st
import pandas as pd
import joblib

saved = joblib.load("traffic_model_small.pkl")
model = saved["model"]
preprocessor = saved["preprocessor"]

st.title("Traffic Congestion Prediction")
st.write("Predict hourly traffic volume using weather and time information.")

categorical = preprocessor.named_transformers_["categorical"].categories_

holiday = st.selectbox("Holiday", categorical[0])
weather = st.selectbox("Weather", categorical[1])
description = st.selectbox("Weather Description", categorical[2])

temp = st.number_input("Temperature (Kelvin)", value=290.0)
rain = st.number_input("Rainfall (mm)", value=0.0)
snow = st.number_input("Snowfall (mm)", value=0.0)
clouds = st.slider("Cloud Coverage (%)", 0, 100, 40)

hour = st.slider("Hour", 0, 23, 16)
day = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 1)
month = st.slider("Month", 1, 12, 11)

if st.button("Predict Traffic"):
    input_data = pd.DataFrame([{
        "holiday": holiday,
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "weather_main": weather,
        "weather_description": description,
        "hour": hour,
        "day_of_week": day,
        "month": month,
        "is_weekend": int(day >= 5)
    }])

    prediction = model.predict(
        preprocessor.transform(input_data)
    )[0]

    st.success(
        f"Predicted traffic: {round(prediction)} vehicles per hour"
    )
