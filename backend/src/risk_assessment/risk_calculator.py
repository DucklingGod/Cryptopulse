import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import json

# Define constants
DATA_DIR = "/home/ubuntu/crypto_dashboard/backend/data"
RISK_LEVELS = {
    "very_low": (0, 20),
    "low": (20, 40),
    "moderate": (40, 60),
    "high": (60, 80),
    "very_high": (80, 100)
}

class RiskCalculator:
    """
    Calculates risk scores for cryptocurrencies based on:
    1. Price volatility
    2. Sentiment analysis
    3. ML prediction trends
    4. Market conditions (Fear & Greed Index)
    """
    
    def __init__(self, symbol="BTC_USD"):
        self.symbol = symbol
        if symbol == "BTC_USD":
            self.data_path = "/app/data/btc-usd-max.csv"
        else:
            data_dir = os.path.join(os.path.dirname(__file__), '../../data')
            filename = f"ml_prepared_data_{symbol}.csv"
            self.data_path = os.path.normpath(os.path.join(data_dir, filename))
        print(f"[DEBUG] RiskCalculator loading data from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.fear_greed_cache = {"timestamp": None, "data": None}
        self.load_data()
    
    def load_data(self):
        """Load and prepare data for risk assessment"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"[DEBUG] Columns in {self.data_path}: {list(df.columns)}")
        # If BTC_USD, set close as next day's open
        if self.symbol == "BTC_USD" and 'open' in df.columns:
            df['close'] = df['open'].shift(-1)
            df = df.iloc[:-1]  # Drop last row with NaN close
        if 'close' not in df.columns:
            raise ValueError(f"CSV file {self.data_path} must contain a 'close' column. Columns found: {list(df.columns)}")
        self.df = df
        if 'datetime' in self.df.columns:
            self.df.set_index('datetime', inplace=True)
            self.df = self.df.sort_index()
    
    def calculate_volatility(self, window=14):
        """Calculate price volatility using standard deviation of returns"""
        if 'price_change_pct' not in self.df.columns:
            self.df['price_change_pct'] = self.df['close'].pct_change() * 100
        
        # Calculate rolling standard deviation of percentage changes
        volatility = self.df['price_change_pct'].rolling(window=window).std().iloc[-1]
        
        # Normalize volatility to a 0-100 scale (higher volatility = higher risk)
        # Assuming max reasonable volatility is 10% standard deviation
        normalized_volatility = min(100, max(0, volatility * 10))
        
        return normalized_volatility
    
    def calculate_sentiment_risk(self, window=7):
        """Calculate risk based on sentiment trends"""
        if 'avg_sentiment_score' not in self.df.columns or self.df['avg_sentiment_score'].isna().all():
            # If sentiment data is missing, return a neutral score
            return 50
        
        # Get recent sentiment scores
        recent_sentiment = self.df['avg_sentiment_score'].iloc[-window:].mean()
        
        # Convert sentiment to risk (negative sentiment = higher risk)
        # Sentiment typically ranges from -1 to 1, so we convert to 0-100 scale
        # where 0 is very positive sentiment (low risk) and 100 is very negative (high risk)
        sentiment_risk = 50 - (recent_sentiment * 50)
        
        return sentiment_risk
    
    def get_ml_prediction_risk(self, prediction_data=None):
        """
        Calculate risk based on ML prediction trend
        
        Args:
            prediction_data: Optional dict with prediction results from ml_model_trainer.predict_price()
                             If None, a neutral score is returned
        """
        if prediction_data is None:
            return 50
        
        try:
            predicted_price = prediction_data.get('predicted_next_close', 0)
            last_price = prediction_data.get('last_actual_close', 0)
            
            if last_price == 0:
                return 50
            
            # Calculate predicted percent change
            predicted_change_pct = ((predicted_price - last_price) / last_price) * 100
            
            # Convert to risk score (negative prediction = higher risk)
            # Scale: -5% change or worse = 100 risk, +5% change or better = 0 risk
            prediction_risk = 50 - (predicted_change_pct * 10)
            prediction_risk = min(100, max(0, prediction_risk))
            
            return prediction_risk
        except Exception as e:
            print(f"Error calculating ML prediction risk: {e}")
            return 50
    
    def get_fear_greed_index(self, force_refresh=False):
        """
        Get the Fear & Greed Index from Alternative.me API
        
        Returns:
            dict: Fear & Greed Index data with value (0-100) and classification
        """
        # Check if we have cached data less than 1 hour old
        current_time = datetime.now()
        if (not force_refresh and 
            self.fear_greed_cache["timestamp"] is not None and 
            current_time - self.fear_greed_cache["timestamp"] < timedelta(hours=1)):
            return self.fear_greed_cache["data"]
        
        try:
            # Fetch Fear & Greed Index from Alternative.me API
            response = requests.get("https://api.alternative.me/fng/")
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    fear_greed_data = {
                        "value": int(data["data"][0]["value"]),
                        "classification": data["data"][0]["value_classification"],
                        "timestamp": data["data"][0]["timestamp"],
                    }
                    
                    # Update cache
                    self.fear_greed_cache = {
                        "timestamp": current_time,
                        "data": fear_greed_data
                    }
                    
                    return fear_greed_data
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
        
        # Return default values if API call fails
        return {"value": 50, "classification": "Neutral", "timestamp": int(current_time.timestamp())}
    
    def calculate_overall_risk(self, ml_prediction=None):
        """
        Calculate overall risk score based on all factors
        
        Args:
            ml_prediction: Optional dict with prediction results
        
        Returns:
            dict: Risk assessment results with scores and classifications
        """
        # Calculate individual risk components
        volatility_risk = self.calculate_volatility()
        sentiment_risk = self.calculate_sentiment_risk()
        prediction_risk = self.get_ml_prediction_risk(ml_prediction)
        
        # Get Fear & Greed Index
        fear_greed_data = self.get_fear_greed_index()
        # Convert fear & greed to risk (extreme fear = high risk, extreme greed = moderate risk)
        fear_greed_value = fear_greed_data["value"]
        # Adjust scale: 0 (extreme fear) = 100 risk, 100 (extreme greed) = 50 risk
        fear_greed_risk = 100 - (fear_greed_value * 0.5)
        
        # Calculate weighted overall risk
        # Weights can be adjusted based on importance of each factor
        weights = {
            "volatility": 0.3,
            "sentiment": 0.2,
            "prediction": 0.3,
            "fear_greed": 0.2
        }
        
        overall_risk = (
            volatility_risk * weights["volatility"] +
            sentiment_risk * weights["sentiment"] +
            prediction_risk * weights["prediction"] +
            fear_greed_risk * weights["fear_greed"]
        )
        
        # Determine risk level classification
        risk_level = "moderate"  # Default
        for level, (min_val, max_val) in RISK_LEVELS.items():
            if min_val <= overall_risk < max_val:
                risk_level = level
                break
        
        # Prepare result
        result = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "overall_risk": {
                "score": round(overall_risk, 2),
                "level": risk_level
            },
            "components": {
                "volatility": {
                    "score": round(volatility_risk, 2),
                    "weight": weights["volatility"]
                },
                "sentiment": {
                    "score": round(sentiment_risk, 2),
                    "weight": weights["sentiment"]
                },
                "prediction": {
                    "score": round(prediction_risk, 2),
                    "weight": weights["prediction"]
                },
                "fear_greed_index": {
                    "score": round(fear_greed_risk, 2),
                    "value": fear_greed_data["value"],
                    "classification": fear_greed_data["classification"],
                    "weight": weights["fear_greed"]
                }
            }
        }
        
        return result


# Test function
def test_risk_calculator():
    """Test the RiskCalculator with sample data"""
    try:
        calculator = RiskCalculator(symbol="BTC_USD")
        
        # Sample ML prediction
        sample_prediction = {
            "predicted_next_close": 95000.0,
            "last_actual_close": 94265.48,
            "trend": "bullish"
        }
        
        # Calculate risk
        risk_assessment = calculator.calculate_overall_risk(sample_prediction)
        
        print(json.dumps(risk_assessment, indent=2))
        return risk_assessment
    except Exception as e:
        print(f"Error testing risk calculator: {e}")
        return None


if __name__ == "__main__":
    test_risk_calculator()
