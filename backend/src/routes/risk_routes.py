from flask import Blueprint, jsonify, request
from src.risk_assessment.risk_calculator import RiskCalculator
# Import from mock ML model instead of the real one
from src.mock_ml_model import predict_price
import logging

risk_bp = Blueprint("risk_bp", __name__, url_prefix="/api/risk")

@risk_bp.route("/assessment", methods=["GET"])
def get_risk_assessment():
    """
    Get risk assessment for a cryptocurrency
    
    Query Parameters:
    - symbol: Symbol of the cryptocurrency (e.g., BTC_USD)
    
    Returns:
    - JSON with risk assessment data
    """
    symbol_param = request.args.get("symbol", "BTC_USD")  # Default to BTC_USD
    
    try:
        # Get ML prediction for the symbol using mock implementation
        ml_prediction = predict_price(symbol=symbol_param)
        
        # Calculate risk assessment
        risk_calculator = RiskCalculator(symbol=symbol_param)
        risk_assessment = risk_calculator.calculate_overall_risk(ml_prediction)
        
        return jsonify(risk_assessment)
    except FileNotFoundError as fnf:
        logging.error(f"FileNotFoundError in /api/risk/assessment: {fnf}", exc_info=True)
        return jsonify({"error": str(fnf)}), 404
    except Exception as e:
        logging.error(f"Exception in /api/risk/assessment: {e}", exc_info=True)
        import traceback
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500

@risk_bp.route("/fear_greed", methods=["GET"])
def get_fear_greed_index():
    """
    Get the current Fear & Greed Index
    
    Returns:
    - JSON with Fear & Greed Index data
    """
    try:
        # Use BTC_USD as default for Fear & Greed Index (it's market-wide, not coin-specific)
        risk_calculator = RiskCalculator(symbol="BTC_USD")
        fear_greed_data = risk_calculator.get_fear_greed_index(force_refresh=True)
        
        return jsonify(fear_greed_data)
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return jsonify({"error": str(e)}), 500
