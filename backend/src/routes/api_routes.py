from flask import Blueprint, jsonify, request
import requests
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Import from mock ML model instead of the real one
from src.mock_ml_model import predict_price

api_bp = Blueprint("api_bp", __name__, url_prefix="/api")
fundamental_bp = Blueprint('fundamental', __name__)

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

@fundamental_bp.route('/fundamental/<symbol>', methods=['GET'])
def get_fundamental_data(symbol):
    """
    Fetch fundamental data for a coin from CoinMarketCap and calculate ROI (since beginning, 1 year, 90 days).
    symbol: e.g. 'BTC', 'ETH', 'SOL'
    """
    try:
        # Fetch CMC listings (limit 100 for now)
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": CMC_API_KEY,
        }
        response = requests.get(f"{CMC_BASE_URL}/listings/latest", headers=headers, params={"limit": 100, "convert": "USD"})
        response.raise_for_status()
        listings = response.json().get('data', [])
        coin = next((c for c in listings if c['symbol'].upper() == symbol.upper()), None)
        if not coin:
            return jsonify({"error": f"Coin {symbol} not found in CMC listings."}), 404
        # Get historical prices for ROI
        # Use Twelvedata for historical price (since CMC doesn't provide historical price in free tier)
        def get_price_n_days_ago(symbol, days):
            td_symbol = f"{symbol.upper()}/USD"
            params = {
                "symbol": td_symbol,
                "interval": "1day",
                "apikey": TWELVEDATA_API_KEY,
                "outputsize": max(days+1, 2)
            }
            resp = requests.get(f"{TWELVEDATA_BASE_URL}/time_series", params=params)
            if resp.status_code != 200:
                return None
            data = resp.json().get('values', [])
            if not data or len(data) <= days:
                return None
            # values are in reverse chronological order (latest first)
            return float(data[days]['close'])
        try:
            current_price = float(coin['quote']['USD']['price'])
        except Exception:
            current_price = None
        price_1y_ago = get_price_n_days_ago(symbol, 365)
        price_90d_ago = get_price_n_days_ago(symbol, 90)
        price_beginning = None
        # Try to get the oldest price available
        td_symbol = f"{symbol.upper()}/USD"
        params = {"symbol": td_symbol, "interval": "1day", "apikey": TWELVEDATA_API_KEY, "outputsize": 5000}
        resp = requests.get(f"{TWELVEDATA_BASE_URL}/time_series", params=params)
        if resp.status_code == 200:
            values = resp.json().get('values', [])
            if values:
                price_beginning = float(values[-1]['close'])
        def calc_roi(cur, old):
            if cur is None or old is None or old == 0:
                return None
            return ((cur - old) / old) * 100
        roi_since_beginning = calc_roi(current_price, price_beginning)
        roi_1y = calc_roi(current_price, price_1y_ago)
        roi_90d = calc_roi(current_price, price_90d_ago)
        # Add ATH/ATL from CoinGecko
        try:
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                # Add more as needed
            }
            coingecko_id = symbol_map.get(symbol.upper(), symbol.lower())
            cg_url = f'https://api.coingecko.com/api/v3/coins/{coingecko_id}'
            cg_resp = requests.get(cg_url, params={"localization": "false", "tickers": "false", "market_data": "true", "community_data": "false", "developer_data": "false", "sparkline": "false"})
            ath = atl = None
            if cg_resp.status_code == 200:
                cg_data = cg_resp.json()
                market_data = cg_data.get('market_data', {})
                ath = market_data.get('ath', {}).get('usd')
                atl = market_data.get('atl', {}).get('usd')
        except Exception:
            ath = atl = None
        result = {
            'name': coin.get('name'),
            'symbol': coin.get('symbol'),
            'market_cap': coin['quote']['USD'].get('market_cap'),
            'circulating_supply': coin.get('circulating_supply'),
            'total_supply': coin.get('total_supply'),
            'max_supply': coin.get('max_supply'),
            'ath': ath,
            'atl': atl,
            'roi_since_beginning': roi_since_beginning,
            'roi_1y': roi_1y,
            'roi_90d': roi_90d,
            'last_updated': coin.get('last_updated'),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/daytrading/insight', methods=['GET'])
def get_daytrading_insight():
    """
    Real-time day trading insight using ML prediction for the selected symbol.
    Query param: symbol (e.g., BTC_USD)
    Returns: entry_point, exit_point, trend, confidence
    """
    symbol = request.args.get('symbol', 'BTC_USD')
    try:
        prediction = predict_price(symbol=symbol)
        # For demo: entry = last price, exit = predicted price
        entry_point = prediction.get('last_actual_close')
        exit_point = prediction.get('predicted_next_close')
        trend = prediction.get('trend')
        confidence = prediction.get('confidence')
        return jsonify({
            'entry_point': entry_point,
            'exit_point': exit_point,
            'trend': trend,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add more routes for other specific API calls as needed
