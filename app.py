import os
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

load_dotenv("./MainKeys.env")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

if not NEWS_API_KEY or not COINMARKETCAP_API_KEY:
    logging.error("API keys not found. Check MainKeys.env")

def get_historical_bitcoin_data():
    """Simulated historical Bitcoin price data (last 14 days)."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=14)
    
    historical_data = []
    base_price = 95000 
    growth_rate = np.linspace(150, 300, 15)  
    
    for i in range(15):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        price = base_price + sum(growth_rate[:i+1]) + np.random.randn() * 200 
        historical_data.append({"time_open": f"{date}T00:00:00Z", "quote": {"USD": {"close": price}}})
    
    return historical_data

    # Uncomment below code to fetch real data from API
    # url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    # params = {
    #     "symbol": "BTC",
    #     "convert": "USD",
    #     "time_start": start_date.strftime("%Y-%m-%d"),
    #     "time_end": end_date.strftime("%Y-%m-%d"),
    #     "interval": "daily",
    # }
    # headers = {
    #     "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
    #     "Accept": "application/json",
    #     "User-Agent": "Mozilla/5.0"
    # }
    # try:
    #     response = requests.get(url, headers=headers, params=params)
    #     response.raise_for_status()
    #     return response.json().get("data", {}).get("quotes", [])
    # except requests.exceptions.RequestException as e:
    #     logging.error(f"Error fetching historical Bitcoin data: {e}")
    #     return None

def predict_price():
    """Predict Bitcoin price for end of 2025 using a regression model."""
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
    
    predicted_price = model.predict(future_day)[0] 

    return predicted_price

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
