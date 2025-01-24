def prepare_data(data, sentiment_score, lookback=60):
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            # מוסיפים את הסנטימנט לכל צעד בזמן (timesteps)
            timestep_data = np.hstack((data[i-lookback:i], np.full((lookback, 1), sentiment_score)))
            X.append(timestep_data)
            y.append(data[i, -1])  # עמודת היעד היא מחיר הסגירה
        if len(X) == 0 or len(y) == 0:
            return None, None
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"שגיאה בעיבוד הנתונים: {e}")
        return None, None


# חלק מהקוד בתוך ה-main logic
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

                    # חיזוי המחיר הבא
                    last_lookback = data_scaled[-lookback:, :-1]
                    last_lookback = np.hstack((last_lookback, np.full((lookback, 1), sentiment_score)))
                    last_lookback = last_lookback.reshape(1, last_lookback.shape[0], last_lookback.shape[1])
                    prediction = model.predict(last_lookback)
                    next_price = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0, 0, 0]])[0, 3]

                    st.write(f"### מחיר החיזוי הבא למניה {ticker}: ${next_price:.2f}")
