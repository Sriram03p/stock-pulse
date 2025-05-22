import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from textblob import TextBlob
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ✅ Your Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = 'B5WJBWUZC8XHRVOF'

# ✅ Get stock news from Finviz
def get_stock_news(symbol):
    try:
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='fullview-news-outer')
        rows = table.find_all('tr')[:10]
        headlines = [row.find_all('td')[1].text.strip() for row in rows]
        return pd.DataFrame({'Headlines': headlines})
    except Exception as e:
        st.warning(f"News retrieval failed: {e}")
        return pd.DataFrame()

# ✅ Get historical stock data (efficient + working)
import requests

def get_stock_data(symbol):
    try:
        api_key = '8a97b0338c804639931f90bff73311cc'
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=100&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        if 'values' not in data:
            st.error(f"API error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.rename(columns={'close': 'Close'})
        df['Close'] = df['Close'].astype(float)
        df.sort_index(inplace=True)
        return df[['Close']]

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()



# ✅ Sentiment Analysis
def analyze_sentiment(text):
    text = ' '.join(text)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_stock_sentiment(df):
    df['Sentiment'] = df['Headlines'].apply(analyze_sentiment)
    pos = (df['Sentiment'] == 'Positive').sum()
    neg = (df['Sentiment'] == 'Negative').sum()
    neu = (df['Sentiment'] == 'Neutral').sum()
    if pos > neg:
        color, msg = 'green', "The reviews of the stock are looking good!"
    elif neg > pos:
        color, msg = 'red', "The reviews of the stock are looking bad!"
    else:
        color, msg = 'orange', "The reviews of the stock are neutral."
    return color, msg

# ✅ Forecasting with ETS
def ets_demand_forecast(stock_data, forecast_period):
    model = ExponentialSmoothing(stock_data['Close'], trend='add', seasonal='add', seasonal_periods=30)
    fit = model.fit()
    forecast = fit.forecast(steps=forecast_period)
    return forecast

# ✅ Plot forecast
def plot_demand_forecast(stock_data, forecast):
    fig, ax = plt.subplots()
    ax.plot(stock_data['Close'], label='Historical', color='blue')
    ax.plot(pd.date_range(start=stock_data.index[-1], periods=len(forecast) + 1, freq='B')[1:], forecast, label='Forecast', color='red')
    ax.set_title("Stock Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

# ✅ Main Streamlit app
def main():
    st.title("📈 Stock Analysis & Forecast App")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)").upper()

    if st.button("Analyze") and stock_symbol:
        df_news = get_stock_news(stock_symbol)
        if not df_news.empty:
            st.subheader("📰 News Headlines")
            st.dataframe(df_news)
            color, message = get_stock_sentiment(df_news)
            st.markdown(f'<p style="color:{color}">{message}</p>', unsafe_allow_html=True)

        st.subheader("📊 Historical Stock Data")
        stock_data = get_stock_data(stock_symbol)
        if not stock_data.empty:
            st.line_chart(stock_data['Close'])

            st.subheader("🔮 Forecast Stock Price")
            forecast_days = st.slider("Forecast Period (days)", 10, 180, 60)
            forecast = ets_demand_forecast(stock_data, forecast_days)
            plot_demand_forecast(stock_data, forecast)

if __name__ == "__main__":
    main()
