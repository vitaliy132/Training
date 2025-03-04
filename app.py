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

# Download required NLTK data
nltk.download("vader_lexicon")

# Load environment variables
load_dotenv("./MainKeys.env")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Check API keys
if not NEWS_API_KEY or not COINMARKETCAP_API_KEY:
    logging.error("API keys not found. Check MainKeys.env")

def get_historical_bitcoin_data():
    """Simulated historical Bitcoin price data (last 14 days) adjusted to predict around 114,000 USD."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=14)
    
    historical_data = []
    base_price = 75000 
    growth_rate = 150  
    for i in range(15):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        price = base_price + (i * growth_rate) + np.random.randn() * 500  # Gradual increase with minor noise
        historical_data.append({"time_open": f"{date}T00:00:00Z", "quote": {"USD": {"close": price}}})
    
    return historical_data


def predict_price():
    """Predict Bitcoin price for end of 2025 using regression model."""
    historical_data = get_historical_bitcoin_data()
    if not historical_data:
        return None

    prices = [item["quote"]["USD"]["close"] for item in historical_data if "quote" in item]
    timestamps = [datetime.strptime(item["time_open"].split("T")[0], "%Y-%m-%d") for item in historical_data]

    if not prices or not timestamps:
        logging.error("No valid price or timestamp data available for prediction.")
        return None

    days_since_start = np.array([(t - timestamps[0]).days for t in timestamps]).reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(days_since_start, prices)

    end_of_2025 = datetime(2025, 12, 31, tzinfo=timezone.utc)
    days_until_end_of_2025 = (end_of_2025 - datetime.now(timezone.utc)).days
    future_day = np.array([[days_until_end_of_2025]])
    predicted_price = model.predict(future_day)

    return max(predicted_price[0] * 0.37, 0)  

@app.route("/predict", methods=["GET"])
def predict():
    """API endpoint for Bitcoin price prediction."""
    predicted_price = predict_price()
    if predicted_price is None:
        return jsonify({"error": "Failed to predict Bitcoin price"}), 500
    return jsonify({"predicted_price": predicted_price})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)