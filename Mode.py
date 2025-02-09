import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta, timezone
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv, dotenv_values

# Download the vader_lexicon
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv(dotenv_path="MainKeys.env")
env_vars = dotenv_values(".env")  # Loads environment variables as a dictionary
print(env_vars)  # For debugging: should show your API keys

# Retrieve API keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_bitcoin_news(api_key, page_size=5):
    url = 'https://newsapi.org/v2/everything'
    params = {'q': 'bitcoin', 'apiKey': api_key, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': page_size}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Bitcoin news: {e}")
        return None

def get_historical_bitcoin_data(api_key, start_date, end_date):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical'
    params = {
        'id': 1,  # Bitcoin's CoinMarketCap ID
        'convert': 'USD',
        'time_start': start_date,
        'time_end': end_date,
        'interval': 'daily'
    }
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Log the raw response for further inspection
        logging.info(f"Raw Response from CoinMarketCap: {data}")

        if 'data' not in data or 'quotes' not in data['data']:
            logging.error("Invalid response structure or no historical data found.")
            return None

        # Extracting the prices and timestamps
        historical_data = []
        for quote in data['data']['quotes']:
            time_open = quote.get('time_open', '')
            close_price = quote.get('quote', {}).get('USD', {}).get('close', None)
            if close_price is not None:
                historical_data.append({
                    'time_open': time_open,
                    'close': close_price
                })

        if not historical_data:
            logging.error("No valid historical data available.")
            return None

        return historical_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching historical Bitcoin data: {e}")
        return None

def analyze_sentiment(news_data):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news_data.get('articles', []):
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}".strip()
        sentiment_score = sid.polarity_scores(text)
        sentiments.append(sentiment_score['compound'])  # Only storing compound score for simplicity
    return sentiments

def predict_price(api_key):
    end_date = datetime.now(timezone.utc)  # Make sure end_date is aware
    start_date = end_date - timedelta(days=30)  # 30 days back for one month of data
    start_date = start_date.replace(tzinfo=None)  # Convert to naive datetime
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    historical_data = get_historical_bitcoin_data(api_key, start_date_str, end_date_str)

    if not historical_data:
        logging.error("Failed to fetch historical Bitcoin data.")
        return None

    prices = [item.get('close', 0) for item in historical_data]
    timestamps = [item.get('time_open', '').split('T')[0] for item in historical_data]
    
    if len(prices) == 0 or len(timestamps) == 0:
        logging.error("No historical price data available.")
        return None
    
    timestamps = [datetime.strptime(t, '%Y-%m-%d') for t in timestamps]  # Convert timestamps to naive datetime
    days_since_start = np.array([(t - start_date).days for t in timestamps]).reshape(-1, 1)

    # Filter out any outliers from the historical data
    price_mean = np.mean(prices)
    price_std = np.std(prices)
    price_threshold = price_mean + 2 * price_std  # Remove prices more than 2 standard deviations away
    filtered_prices = [price for price in prices if price <= price_threshold]

    if len(filtered_prices) == 0:
        logging.error("All data points are outliers, unable to train model.")
        return None

    # Fit a linear regression model to predict the price
    model = LinearRegression()
    model.fit(days_since_start[:len(filtered_prices)], filtered_prices)
    
    # Predicting for the end of 2025
    end_of_2025 = datetime(2025, 12, 31, tzinfo=timezone.utc)  # Make end_of_2025 aware (UTC timezone)
    days_until_end_of_2025 = (end_of_2025 - end_date).days  # Now this subtraction is valid
    future_day = np.array([[days_until_end_of_2025]])  # Predicting for Dec 31, 2025
    predicted_price = model.predict(future_day)
    
    plot_bitcoin_prices(days_since_start, filtered_prices, future_day, predicted_price)
    return predicted_price[0]

def plot_bitcoin_prices(days_since_start, prices, future_day, predicted_price):
    plt.figure(figsize=(8, 6))
    plt.plot(days_since_start, prices, color='blue', label='Historical Prices')
    plt.plot(future_day, predicted_price, 'ro', label=f'Predicted Price (2025): ${predicted_price[0]:.2f}')
    plt.xlabel("Days Since Start")
    plt.ylabel("Price (in USD)")
    plt.title("Bitcoin Price Prediction")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def main():
    if not NEWS_API_KEY or not COINMARKETCAP_API_KEY:
        logging.error("Missing API keys. Set them in the .env file.")
        return
    news = get_bitcoin_news(NEWS_API_KEY)
    if news and news.get('status') == 'ok':
        logging.info("Fetched Bitcoin news successfully!")
        sentiments = analyze_sentiment(news)
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            logging.info(f"Average sentiment of the latest news: {avg_sentiment:.2f}")
        else:
            logging.warning("No valid sentiment data available.")
    else:
        logging.error("Failed to fetch Bitcoin news")
    
    predicted_price = predict_price(COINMARKETCAP_API_KEY)
    if predicted_price is not None:
        logging.info(f"Predicted Bitcoin Price in 2025: ${predicted_price:.2f}")
    else:
        logging.error("Failed to predict Bitcoin price.")

if __name__ == '__main__':
    main()
