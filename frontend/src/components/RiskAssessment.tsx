import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { RiskAssessment as RiskAssessmentType, FearGreedIndex } from '../types';
import './RiskAssessment.css';

interface RiskAssessmentProps {
  symbol: string;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ symbol }) => {
  const [riskData, setRiskData] = useState<RiskAssessmentType | null>(null);
  const [fearGreedData, setFearGreedData] = useState<FearGreedIndex | null>(null);
  const [loading, setLoading] = useState({
    risk: true,
    fearGreed: true
  });
  const [error, setError] = useState({
    risk: null as string | null,
    fearGreed: null as string | null
  });

  const apiBaseUrl = import.meta.env.VITE_BACKEND_API_BASE_URL;

  // Fetch risk assessment data
  useEffect(() => {
    const fetchRiskData = async () => {
      try {
        setLoading(prev => ({ ...prev, risk: true }));
        const response = await axios.get<RiskAssessmentType>(`${apiBaseUrl}/risk/assessment?symbol=${symbol}`);
        // TEMP: Log backend response for debugging
        console.log('Risk assessment API response:', response.data);
        setRiskData(response.data);
        setLoading(prev => ({ ...prev, risk: false }));
      } catch (err) {
        console.error('Error fetching risk assessment data:', err);
        setError(prev => ({ ...prev, risk: 'Failed to fetch risk assessment data' }));
        setLoading(prev => ({ ...prev, risk: false }));
      }
    };

    fetchRiskData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchRiskData, 300000);
    return () => clearInterval(interval);
  }, [symbol]);

  // Fetch Fear & Greed Index data
  useEffect(() => {
    const fetchFearGreedData = async () => {
      try {
        setLoading(prev => ({ ...prev, fearGreed: true }));
        const response = await axios.get<FearGreedIndex>(`${apiBaseUrl}/risk/fear_greed`);
        setFearGreedData(response.data);
        setLoading(prev => ({ ...prev, fearGreed: false }));
      } catch (err) {
        console.error('Error fetching Fear & Greed Index:', err);
        setError(prev => ({ ...prev, fearGreed: 'Failed to fetch Fear & Greed Index' }));
        setLoading(prev => ({ ...prev, fearGreed: false }));
      }
    };

    fetchFearGreedData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchFearGreedData, 300000);
    return () => clearInterval(interval);
  }, []);

  // Helper function to determine color based on risk level
  const getRiskLevelColor = (level: string): string => {
    switch (level.toLowerCase()) {
      case 'very_low':
      case 'very low':
        return '#16c784'; // Green
      case 'low':
        return '#4caf50'; // Light Green
      case 'moderate':
        return '#ff9800'; // Orange
      case 'high':
        return '#f44336'; // Red
      case 'very_high':
      case 'very high':
        return '#d32f2f'; // Dark Red
      default:
        return '#ff9800'; // Default to orange
    }
  };

  // Helper function to determine color based on trend
  const getTrendColor = (trend: string): string => {
    switch (trend.toLowerCase()) {
      case 'bullish':
        return '#16c784'; // Green
      case 'bearish':
        return '#ea3943'; // Red
      case 'neutral':
        return '#888888'; // Gray
      default:
        return '#888888'; // Default to gray
    }
  };

  // Helper function to determine color based on Fear & Greed value
  const getFearGreedColor = (value: number): string => {
    if (value <= 25) return '#ea3943'; // Extreme Fear - Red
    if (value <= 45) return '#ff9800'; // Fear - Orange
    if (value <= 55) return '#ffeb3b'; // Neutral - Yellow
    if (value <= 75) return '#4caf50'; // Greed - Green
    return '#16c784'; // Extreme Greed - Dark Green
  };

  // Determine market trends based on risk components
  const getMarketTrends = () => {
    // Defensive: ensure components is a plain object and not null/array, and matches expected keys
    const components = riskData?.components;
    if (
      !components ||
      typeof components !== 'object' ||
      Array.isArray(components) ||
      !('prediction' in components) ||
      !('sentiment' in components) ||
      !('fear_greed_index' in components)
    ) {
      return { shortTerm: 'Neutral', mediumTerm: 'Neutral', longTerm: 'Neutral' };
    }
    // Use optional chaining for all subfields
    const shortTermScore = components.prediction?.score ?? 50;
    const mediumTermScore = components.sentiment?.score ?? 50;
    const longTermScore = components.fear_greed_index?.score ?? 50;

    const shortTerm = shortTermScore < 40 ? 'Bullish' : shortTermScore > 60 ? 'Bearish' : 'Neutral';
    const mediumTerm = mediumTermScore < 40 ? 'Bullish' : mediumTermScore > 60 ? 'Bearish' : 'Neutral';
    const longTerm = longTermScore < 40 ? 'Bullish' : longTermScore > 60 ? 'Bearish' : 'Neutral';

    return { shortTerm, mediumTerm, longTerm };
  };

  const marketTrends = getMarketTrends();

  return (
    <div className="risk-assessment-container">
      <div className="risk-factors-section">
        <h2>Risk Factors</h2>
        {loading.risk ? (
          <p>Loading risk assessment data...</p>
        ) : error.risk ? (
          <p className="error">{error.risk}</p>
        ) : riskData && riskData.overall_risk && riskData.components &&
          'volatility' in riskData.components &&
          'sentiment' in riskData.components &&
          'prediction' in riskData.components &&
          'fear_greed_index' in riskData.components ? (
          <div className="risk-factors-content">
            <div className="risk-factor">
              <span className="risk-label">Risk level</span>
              <span 
                className="risk-value" 
                style={{ color: getRiskLevelColor(riskData.overall_risk.level) }}
              >
                {riskData.overall_risk.level.charAt(0).toUpperCase() + riskData.overall_risk.level.slice(1).replace('_', ' ')}
              </span>
            </div>
            <div className="risk-factor">
              <span className="risk-label">Volatility</span>
              <span 
                className="risk-value" 
                style={{ 
                  color: riskData.components.volatility && riskData.components.volatility.score > 60 ? '#ea3943' : 
                         riskData.components.volatility && riskData.components.volatility.score > 30 ? '#ff9800' : '#16c784' 
                }}
              >
                {riskData.components.volatility && riskData.components.volatility.score > 60 ? 'High' : 
                 riskData.components.volatility && riskData.components.volatility.score > 30 ? 'Moderate' : 'Low'}
              </span>
            </div>
            <div className="risk-factor">
              <span className="risk-label">External factors</span>
              <span className="risk-value">
                {riskData.components.fear_greed_index?.classification ?? 'N/A'}
              </span>
            </div>
          </div>
        ) : (
          <p>No risk assessment data available.</p>
        )}
      </div>

      <div className="market-trend-section">
        <h2>Market Trend</h2>
        {loading.risk ? (
          <p>Loading market trend data...</p>
        ) : error.risk ? (
          <p className="error">{error.risk}</p>
        ) : (
          <div className="market-trend-content">
            <div className="trend-overall">
              <span 
                className="trend-value" 
                style={{ color: getTrendColor(marketTrends.shortTerm) }}
              >
                {marketTrends.shortTerm}
              </span>
            </div>
            <div className="trend-detail">
              <span className="trend-label">Short-term</span>
              <span 
                className="trend-value" 
                style={{ color: getTrendColor(marketTrends.shortTerm) }}
              >
                {marketTrends.shortTerm}
              </span>
            </div>
            <div className="trend-detail">
              <span className="trend-label">Medium-term</span>
              <span 
                className="trend-value" 
                style={{ color: getTrendColor(marketTrends.mediumTerm) }}
              >
                {marketTrends.mediumTerm}
              </span>
            </div>
            <div className="trend-detail">
              <span className="trend-label">Long-term</span>
              <span 
                className="trend-value" 
                style={{ color: getTrendColor(marketTrends.longTerm) }}
              >
                {marketTrends.longTerm}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="fear-greed-section">
        <h2>Fear & Greed Index</h2>
        {loading.fearGreed ? (
          <p>Loading Fear & Greed Index...</p>
        ) : error.fearGreed ? (
          <p className="error">{error.fearGreed}</p>
        ) : fearGreedData ? (
          <div className="fear-greed-content">
            <div className="fear-greed-meter">
              <div className="fear-greed-value" style={{ color: getFearGreedColor(fearGreedData.value) }}>
                {fearGreedData.value}
              </div>
              <div className="fear-greed-label" style={{ color: getFearGreedColor(fearGreedData.value) }}>
                {fearGreedData.classification}
              </div>
            </div>
            <div className="fear-greed-scale">
              <div className="scale-marker extreme-fear">Extreme Fear</div>
              <div className="scale-marker fear">Fear</div>
              <div className="scale-marker neutral">Neutral</div>
              <div className="scale-marker greed">Greed</div>
              <div className="scale-marker extreme-greed">Extreme Greed</div>
            </div>
          </div>
        ) : (
          <p>No Fear & Greed Index data available.</p>
        )}
      </div>
    </div>
  );
};

export default RiskAssessment;
