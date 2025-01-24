
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os

# הגבלת שימוש ב-GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# פונקציה לאיסוף נתונים
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error("לא נמצאו נתונים עבור סמל המניה או טווח התאריכים.")
        return data
    except Exception as e:
        st.error(f"שגיאה באיסוף הנתונים: {e}")
        return pd.DataFrame()

# פונקציה לעיבוד נתונים
def prepare_data(data, lookback=60):
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"שגיאה בעיבוד הנתונים: {e}")
        return None, None

# פונקציה לבניית המודל
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # שימוש נכון בהגדרת קלט
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ממשק Streamlit
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
        if not data.empty:
            st.write("נתונים שהתקבלו:")
            st.write(data.head())

            # עיבוד נתונים
            data = data[['Close']]
            scaler = MinMaxScaler()
            data['Close'] = scaler.fit_transform(data[['Close']])
            dataset = data.values
            X, y = prepare_data(dataset, lookback)

            if X is not None and y is not None:
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
            else:
                st.error("שגיאה בעיבוד הנתונים. נסה שוב עם קלטים אחרים.")
