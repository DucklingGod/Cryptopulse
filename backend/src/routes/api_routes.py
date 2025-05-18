from flask import Blueprint, jsonify, request
import requests
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Import from mock ML model instead of the real one
from src.mock_ml_model import predict_price

api_bp = Blueprint("api_bp", __name__, url_prefix="/api")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# CoinMarketCap Configuration
CMC_API_KEY = os.getenv("CMC_PRO_API_KEY", "b62f2dac-07a8-45a8-b7fa-185be0cfcea7")
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency"

# CryptoPanic Configuration
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "27b9d6ac8ba3104cb0e505a7ea00f92700d385f0")
CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1"

# Twelvedata Configuration
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "e172100827344b17ab361c0618ae0618")
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"

@api_bp.route("/cmc/listings", methods=["GET"])
def get_cmc_listings():
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
    }
    parameters = {
        "start": request.args.get("start", "1"),
        "limit": request.args.get("limit", "100"),
        "convert": request.args.get("convert", "USD"),
    }
    try:
        response = requests.get(f"{CMC_BASE_URL}/listings/latest", headers=headers, params=parameters)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/cryptopanic/posts", methods=["GET"])
def get_cryptopanic_posts():
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": request.args.get("public", "true"),
        "filter": request.args.get("filter"),
        "currencies": request.args.get("currencies"),
        "regions": request.args.get("regions", "en"),
        "kind": request.args.get("kind")
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = requests.get(f"{CRYPTOPANIC_BASE_URL}/posts/", params=params)
        response.raise_for_status()
        data = response.json()
        if "results" in data:
            for post in data["results"]:
                if "title" in post:
                    sentiment = sia.polarity_scores(post["title"])
                    post["sentiment"] = sentiment
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/twelvedata/time_series", methods=["GET"])
def get_twelvedata_time_series():
    symbol = request.args.get("symbol")
    interval = request.args.get("interval", "1day")
    outputsize = request.args.get("outputsize", "30")

    if not symbol:
        return jsonify({"error": "Symbol parameter is required (e.g., BTC/USD)"}), 400

    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": outputsize
    }
    try:
        response = requests.get(f"{TWELVEDATA_BASE_URL}/time_series", params=params)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/ml/predict_trend", methods=["GET"])
def get_ml_trend_prediction():
    symbol_param = request.args.get("symbol", "BTC_USD") # Default to BTC_USD for the API
    sequence_length_param = request.args.get("sequence_length", "60")
    
    try:
        sequence_length = int(sequence_length_param)
    except ValueError:
        return jsonify({"error": "sequence_length must be an integer"}), 400

    if not symbol_param:
        return jsonify({"error": "Symbol parameter is required (e.g., BTC_USD)"}), 400

    try:
        # Using the mock predict_price function instead of the TensorFlow-based one
        prediction = predict_price(symbol=symbol_param.replace("/", "_"), sequence_length=sequence_length)
        if prediction:
            return jsonify(prediction)
        else:
            return jsonify({"error": "Failed to get prediction, model or data might be unavailable."}), 500
    except Exception as e:
        print(f"Error during ML prediction: {e}") # Log to server console
        return jsonify({"error": f"An internal error occurred during prediction: {str(e)}"}), 500

# Add more routes for other specific API calls as needed
