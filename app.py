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

# Load secrets from the provided file path
SECRET_FILE_PATH = "/etc/secrets/MainKeys.env"  # Change this to match your filename

if os.path.exists(SECRET_FILE_PATH):
    load_dotenv(SECRET_FILE_PATH)  # Load the secret file

# Retrieve API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

app = Flask(__name__)
CORS(app)  

logging.basicConfig(level=logging.INFO)

# The rest of your existing functions go here...
