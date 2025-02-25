import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# בסיס נתונים להערכת מהימנות מקורות (לדוגמה בלבד)
source_reliability = {
    "source1": 0.9,
    "source2": 0.8,
    "source3": 0.5
}

# פונקציה לאיסוף חדשות
def fetch_news(ticker):
    try:
        API_KEY = "your_newsapi_key_here"  # הכנס את מפתח ה-API שלך
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={API_KEY}"
        response = requests.get(url)
        news_data = response.json()
        
        if news_data.get("status") != "ok":
            st.error("שגיאה באיסוף נתוני החדשות.")
            return []
        
        # שמירת כותרות ומקורות
        articles = news_data["articles"]
        headlines = [(article["title"], article["source"]["name"]) for article in articles]
        return headlines
    except Exception as e:
        st.error(f"שגיאה באיסוף החדשות: {e}")
        return []

# פונקציה לניתוח סנטימנט
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    weighted_sentiment = []
    for headline, source in headlines:
        sentiment = analyzer.polarity_scores(headline)["compound"]
        reliability = source_reliability.get(source, 0.5)  # ברירת מחדל למהימנות 0.5 אם המקור לא נמצא
        weighted_sentiment.append(sentiment * reliability)
    return np.mean(weighted_sentiment) if weighted_sentiment else 0

# פונקציה לאיסוף נתוני מניה
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error("לא נמצאו נתונים עבור סמל המניה או טווח התאריכים.")
            return pd.DataFrame()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']].dropna()
        return data
    except Exception as e:
        st.error(f"שגיאה באיסוף הנתונים: {e}")
        return pd.DataFrame()

# פונקציה לעיבוד נתונים
def prepare_data(data, sentiment_score, lookback=60):
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            timestep_data = np.hstack((data[i-lookback:i], np.full((lookback, 1), sentiment_score)))
            X.append(timestep_data)
            y.append(data[i, -1])  # עמודת היעד היא מחיר הסגירה
        if len(X) == 0 or len(y) == 0:
            return None, None
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"שגיאה בעיבוד הנתונים: {e}")
        return None, None

# פונקציה לבניית המודל
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ממשק Streamlit
st.title("מנבא תנודות בבורסה עם ניתוח מהימנות חדשות")
st.write("אפליקציה זו משתמשת בניתוח סנטימנט ומהימנות של מקורות חדשותיים לחיזוי מחירים.")

# קלטים מהמשתמש
ticker = st.text_input("הזן סמל מניה (לדוגמה: AAPL, TSLA):", "AAPL")
start_date = st.date_input("תאריך התחלה:", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("תאריך סיום:", value=pd.to_datetime("2025-01-01"))
lookback = st.slider("מספר הימים לניתוח (lookback):", min_value=30, max_value=120, value=60)

if st.button("התחל חיזוי"):
    with st.spinner("טוען נתונים ובונה מודל..."):
        # שליפת נתוני מניה
        data = get_stock_data(ticker, start_date, end_date)
        if not data.empty:
            st.write("### נתונים היסטוריים של המניה:")
            st.write(data.head())

            # שליפת חדשות וניתוח סנטימנט
            headlines = fetch_news(ticker)
            if not headlines:
                st.warning("לא נמצאו כותרות חדשותיות.")
                sentiment_score = 0
            else:
                st.write("### כותרות חדשותיות:")
                for headline, source in headlines:
                    st.write(f"- {headline} (מקור: {source})")
                sentiment_score = analyze_sentiment(headlines)
                st.write(f"### ציון סנטימנט משוקלל: {sentiment_score:.2f}")

            # נרמול הנתונים
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            # בדיקה אם יש מספיק נתונים
            if len(data_scaled) < lookback:
                st.error("לא מספיק נתונים בטווח התאריכים שנבחר. נסה לבחור טווח תאריכים גדול יותר.")
            else:
                X, y = prepare_data(data_scaled, sentiment_score, lookback)

                if X is None or X.shape[0] == 0:
                    st.error("שגיאה בעיבוד הנתונים: לא נוצרו נתוני עיבוד. נסה טווח תאריכים גדול יותר.")
                else:
                    st.write(f"### צורת X לאחר עיבוד: {X.shape}")
                    st.write(f"### צורת y לאחר עיבוד: {y.shape}")

                    # שינוי צורת הנתונים
                    try:
                        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
                        st.write(f"### צורת X לאחר שינוי הצורה: {X.shape}")
                    except Exception as e:
                        st.error(f"שגיאה בשינוי צורת הנתונים: {e}")
                        st.stop()

                    # בניית ואימון המודל
                    model = build_model((X.shape[1], X.shape[2]))
                    st.write("### מודל ה-LSTM מאומן...")
                    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

                    # יצירת last_lookback
                    try:
                        last_lookback = data_scaled[-lookback:, :]  # כל העמודות
                        last_lookback = np.hstack((last_lookback, np.full((lookback, 1), sentiment_score)))
                        last_lookback = last_lookback.reshape(1, last_lookback.shape[0], last_lookback.shape[1])  # התאמה למודל
                        st.write(f"צורת last_lookback: {last_lookback.shape}")
                    except Exception as e:
                        st.error(f"שגיאה ביצירת last_lookback: {e}")
                        st.stop()

                    # בדיקה לצורה תקינה
                    if last_lookback.shape[2] != X.shape[2]:
                        st.error(f"שגיאה: מספר התכונות ב-last_lookback ({last_lookback.shape[2]}) אינו תואם למודל ({X.shape[2]}).")
                        st.stop()

                    # חיזוי המחיר הבא
                    try:
                        prediction = model.predict(last_lookback)
                        next_price = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0, 0, 0]])[0, 3]

                        # מחיר המניה של היום
                        current_price = scaler.inverse_transform([data_scaled[-1]])[0][3]

                        # חישוב הפער
                        price_difference = next_price - current_price
                        percentage_change = (price_difference / current_price) * 100
                        change_direction = "עלייה" if price_difference > 0 else "ירידה"

                        # הצגת התוצאות
                        st.write(f"### מחיר המניה היום: ${current_price:.2f}")
                        st.write(f"### מחיר המניה החזוי למחר: ${next_price:.2f}")
                        st.write(f"### הפער בין המחירים: ${price_difference:.2f} ({percentage_change:.2f}% {change_direction})")

                        # הוספת סיכום תהליך החיזוי
                        st.write("### שורה תחתונה - סיכום תהליך החיזוי:")
                        st.write(f"""
                        - **מספר חלונות הזמן שנלמדו**: {X.shape[0]} חלונות (רצפים של 60 ימים).
                        - **מספר ימים בכל חלון**: {X.shape[1]} ימים.
                        - **מספר הנתונים לכל יום**: {X.shape[2]} נתונים (כולל סנטימנט).
                        - **נתונים אחרונים לחיזוי**: 60 הימים האחרונים + ציון סנטימנט משוקלל.
                        - **תוצאה**: המחיר החזוי של המניה ליום הבא הוא **${next_price:.2f}**.
                        """)

                        # אזור השאלות
                        st.write("### שאלות על הניתוח")
                        user_question = st.text_input("מה תרצה לשאול את המודל על הניתוח שביצע?", "")
                        if user_question:
                            if "סנטימנט" in user_question:
                                st.write(f"הסנטימנט האחרון שנמדד עבור המניה הוא {sentiment_score:.2f}.")
                            elif "חיזוי" in user_question:
                                st.write(f"המחיר החזוי הבא למניה {ticker} הוא ${next_price:.2f}.")
                            elif "נתונים" in user_question:
                                st.write(f"עבור כל יום, המודל השתמש ב-8 נתונים הכוללים: מחיר פתיחה, מחיר סגירה, מחירים גבוהים ונמוכים, נפח מסחר, ממוצעים נעים, וסנטימנט חדשות.")
                            else:
                                st.write("לא הצלחתי להבין את השאלה. נסה לשאול משהו כמו: 'מה היה הסנטימנט?' או 'מה המחיר החזוי?'")
                    except Exception as e:
                        st.error(f"שגיאה בעת חיזוי המחיר: {e}")
                        st.stop()
