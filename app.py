import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import joblib

# Load Booster model
model = xgb.Booster()
model.load_model("xgb_nvidia_model.json")

# Load feature names
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return%', 'Volatility',
       'Year', 'Month', 'DayOfWeek']

st.title("📈 NVIDIA Stock Movement Predictor")
st.markdown("Predict if **tomorrow's closing price** will likely go UP based on today's data.")

# Input UI
open_price = st.number_input("Open Price", min_value=0.0, value=500.0, step=0.1)
high = st.number_input("High Price", min_value=0.0, value=510.0, step=0.1)
low = st.number_input("Low Price", min_value=0.0, value=490.0, step=0.1)
close = st.number_input("Close Price", min_value=0.0, value=505.0, step=0.1)
volume = st.number_input("Volume", min_value=0, value=1000000, step=10000)

if st.button("Predict"):
    return_pct = ((close - open_price) / open_price) * 100
    volatility = ((high - low) / open_price) * 100
    today = datetime.today()
    year = today.year
    month = today.month
    dayofweek = today.weekday()

    input_data = pd.DataFrame([[open_price, high, low, close, volume,
                                return_pct, volatility, year, month, dayofweek]],
                              columns=feature_columns)

    dinput = xgb.DMatrix(input_data)
    prob = model.predict(dinput)[0]  # Probability that target == 1
    prediction = int(prob >= 0.5)

    if prediction == 1:
        st.success(f"📈 Prediction: Price likely to **go UP** (confidence: {prob:.2%})")
    else:
        st.warning(f"📉 Prediction: Price likely to **go DOWN or stay same** (confidence: {1 - prob:.2%})")
