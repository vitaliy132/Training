import os
import requests
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv

nltk.download("vader_lexicon")

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

app = Flask(__name__)
CORS(app)  

logging.basicConfig(level=logging.INFO)


def get_bitcoin_news():
    """Fetch latest Bitcoin news."""
    url = "https://newsapi.org/v2/everything"
    params = {"q": "bitcoin", "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": 5}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Bitcoin news: {e}")
        return None


def analyze_sentiment(news_data):
    """Perform sentiment analysis on Bitcoin news."""
    sid = SentimentIntensityAnalyzer()
    sentiments = []

    for article in news_data.get("articles", []):
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        sentiment_score = sid.polarity_scores(text)
        sentiments.append(sentiment_score["compound"])  

    return sentiments


def get_historical_bitcoin_data():
    """Fetch historical Bitcoin price data."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=10)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    params = {
        "id": 1, 
        "convert": "USD",
        "time_start": start_date_str,
        "time_end": end_date_str,
        "interval": "daily",
    }
    headers = {"X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("quotes", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching historical Bitcoin data: {e}")
        return None


def predict_price():
    """Predict Bitcoin price for end of 2025 using regression model."""
    historical_data = get_historical_bitcoin_data()
    if not historical_data:
        return None

    prices = [item["quote"]["USD"]["close"] for item in historical_data if "quote" in item]
    timestamps = [datetime.strptime(item["time_open"].split("T")[0], "%Y-%m-%d") for item in historical_data]

    days_since_start = np.array([(t - timestamps[0]).days for t in timestamps]).reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(days_since_start, prices)

    end_of_2025 = datetime(2025, 12, 31, tzinfo=timezone.utc)
    days_until_end_of_2025 = (end_of_2025 - datetime.now(timezone.utc)).days
    future_day = np.array([[days_until_end_of_2025]])
    predicted_price = model.predict(future_day)

    return predicted_price[0]


@app.route("/predict", methods=["GET"])
def predict():
    """API endpoint for Bitcoin price prediction."""
    predicted_price = predict_price()
    if predicted_price is None:
        return jsonify({"error": "Failed to predict Bitcoin price"}), 500
    return jsonify({"predicted_price": predicted_price})


@app.route("/news-sentiment", methods=["GET"])
def news_sentiment():
    """API endpoint for Bitcoin news sentiment analysis."""
    news = get_bitcoin_news()
    if news:
        sentiments = analyze_sentiment(news)
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return jsonify({"average_sentiment": avg_sentiment})
    return jsonify({"error": "Failed to fetch news"}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
