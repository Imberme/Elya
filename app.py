
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# פונקציה לאיסוף נתונים
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# פונקציה לעיבוד נתונים
def prepare_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# פונקציה לבניית המודל
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit ממשק
st.title("מנבא תנודות בבורסה")
st.write("אפליקציה זו משתמשת בלמידת מכונה כדי לחזות תנודות מחירים של מניות.")

# קלטים מהמשתמש
ticker = st.text_input("הזן סמל מניה (לדוגמה: AAPL, TSLA):", "AAPL")
start_date = st.date_input("תאריך התחלה:", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("תאריך סיום:", value=pd.to_datetime("2025-01-01"))
lookback = st.slider("מספר הימים לניתוח (lookback):", min_value=30, max_value=120, value=60)

if st.button("התחל חיזוי"):
    with st.spinner("טוען נתונים ובונה מודל..."):
        # שליפת נתונים
        data = get_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.error("לא נמצאו נתונים למניה זו. נסה שוב.")
        else:
            # עיבוד נתונים
            data = data[['Close']]
            scaler = MinMaxScaler()
            data['Close'] = scaler.fit_transform(data[['Close']])
            dataset = data.values
            X, y = prepare_data(dataset, lookback)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # בניית ואימון המודל
            model = build_model((X.shape[1], 1))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # חיזוי המחיר הבא
            last_lookback = dataset[-lookback:]
            last_lookback = last_lookback.reshape(1, lookback, 1)
            prediction = model.predict(last_lookback)
            next_price = scaler.inverse_transform(prediction)[0, 0]

            st.success(f"מחיר החיזוי הבא למניה {ticker}: ${next_price:.2f}")
