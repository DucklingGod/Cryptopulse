import { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './CryptoDataDisplay.css';
import { Coin, NewsItem } from '../types'; // Removed HistoricalValue
import RiskAssessment from './RiskAssessment';
import TradingViewChart from './TradingViewChart';

const apiBaseUrl = import.meta.env.VITE_BACKEND_API_BASE_URL;

const CryptoDataDisplay = () => {
  const [cryptoData, setCryptoData] = useState<Coin[]>([]);
  const [newsData, setNewsData] = useState<NewsItem[]>([]);
  const [selectedCoin, setSelectedCoin] = useState('BTC/USD');
  const [tradingViewSymbol, setTradingViewSymbol] = useState('BINANCE:BTCUSDT');
  const [loading, setLoading] = useState({
    crypto: true,
    news: true,
    historical: true
  });
  const [error, setError] = useState({
    crypto: null as string | null,
    news: null as string | null,
    historical: null as string | null
  });

  // Fetch cryptocurrency data from backend
  useEffect(() => {
    const fetchCryptoData = async () => {
      try {
        setLoading(prev => ({ ...prev, crypto: true }));
        const response = await axios.get<{ data: Coin[] }>(`${apiBaseUrl}/cmc/listings?limit=100`);
        setCryptoData(response.data.data);
        setLoading(prev => ({ ...prev, crypto: false }));
      } catch (err) {
        console.error('Error fetching crypto data:', err);
        setError(prev => ({ ...prev, crypto: 'Failed to fetch cryptocurrency data' }));
        setLoading(prev => ({ ...prev, crypto: false }));
      }
    };

    fetchCryptoData();
    const interval = setInterval(fetchCryptoData, 60000);
    return () => clearInterval(interval);
  }, []);

  // Fetch news data from backend
  useEffect(() => {
    const fetchNewsData = async () => {
      try {
        setLoading(prev => ({ ...prev, news: true }));
        const response = await axios.get<{ results: NewsItem[] }>(`${apiBaseUrl}/cryptopanic/posts`);
        setNewsData(response.data.results);
        setLoading(prev => ({ ...prev, news: false }));
      } catch (err) {
        console.error('Error fetching news data:', err);
        setError(prev => ({ ...prev, news: 'Failed to fetch news data' }));
        setLoading(prev => ({ ...prev, news: false }));
      }
    };

    fetchNewsData();
    const interval = setInterval(fetchNewsData, 60000);
    return () => clearInterval(interval);
  }, []);

  // Fetch historical data for selected coin
  useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!selectedCoin) return;
      try {
        setLoading(prev => ({ ...prev, historical: true }));
        // Removed unused axios call and setHistoricalData
        setLoading(prev => ({ ...prev, historical: false }));
      } catch (err) {
        console.error('Error fetching historical data:', err);
        setError(prev => ({ ...prev, historical: 'Failed to fetch historical data for ' + selectedCoin }));
        setLoading(prev => ({ ...prev, historical: false }));
      }
    };

    fetchHistoricalData();
  }, [selectedCoin]);

  // Update TradingView symbol when selected coin changes
  useEffect(() => {
    // Convert selected coin format (e.g., "BTC/USD") to TradingView format (e.g., "BINANCE:BTCUSDT")
    const parts = selectedCoin.split('/');
    if (parts.length === 2) {
      const base = parts[0];
      const quote = parts[1] === 'USD' ? 'USDT' : parts[1]; // TradingView often uses USDT instead of USD
      setTradingViewSymbol(`BINANCE:${base}${quote}`);
    }
  }, [selectedCoin]);

  // Fetch fundamental data for selected coin from CoinGecko
  const [fundamentalData, setFundamentalData] = useState<any>(null);
  const [loadingFundamental, setLoadingFundamental] = useState(false);
  const [errorFundamental, setErrorFundamental] = useState<string | null>(null);

  useEffect(() => {
    const fetchFundamentalData = async () => {
      if (!selectedCoin) return;
      setLoadingFundamental(true);
      setErrorFundamental(null);
      setFundamentalData(null);
      try {
        // Extract symbol (e.g., BTC from BTC/USD)
        const symbol = selectedCoin.split('/')[0];
        const response = await axios.get(`${apiBaseUrl}/fundamental/${symbol}`);
        setFundamentalData(response.data);
      } catch (err: any) {
        setErrorFundamental('Failed to fetch fundamental data.');
        setFundamentalData(null);
      } finally {
        setLoadingFundamental(false);
      }
    };
    fetchFundamentalData();
  }, [selectedCoin]);

  // Day Trading Insight (real-time)
  const [dayTradingInsight, setDayTradingInsight] = useState<any>(null);
  const [loadingDayTrading, setLoadingDayTrading] = useState(false);
  const [errorDayTrading, setErrorDayTrading] = useState<string | null>(null);

  useEffect(() => {
    const fetchDayTradingInsight = async () => {
      setLoadingDayTrading(true);
      setErrorDayTrading(null);
      setDayTradingInsight(null);
      try {
        const symbol = selectedCoin.replace('/', '_');
        const response = await axios.get(`${apiBaseUrl}/daytrading/insight?symbol=${symbol}`);
        setDayTradingInsight(response.data);
      } catch (err: any) {
        setErrorDayTrading('Failed to fetch day trading insight.');
      } finally {
        setLoadingDayTrading(false);
      }
    };
    fetchDayTradingInsight();
  }, [selectedCoin]);

  const handleCoinChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCoin(e.target.value);
  };

  return (
    <div className="crypto-data-container">
      <div className="dashboard-header">
        <h1>Crypto Market Analysis Dashboard</h1>
        <div className="coin-selector">
          <label htmlFor="coin-select">Select Coin: </label>
          <select id="coin-select" value={selectedCoin} onChange={handleCoinChange}>
            {Array.isArray(cryptoData) && cryptoData.length > 0 ? (
              cryptoData.slice(0,10).filter(Boolean).map(coin => (
                <option key={coin.id} value={`${coin.symbol}/USD`}>{coin.name} ({coin.symbol}/USD)</option>
              ))
            ) : (
              <>
                <option value="BTC/USD">Bitcoin (BTC/USD)</option>
                <option value="ETH/USD">Ethereum (ETH/USD)</option>
              </>
            )}
          </select>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-column">
          <div className="coin-insights-section dashboard-section">
            <h2>Coin Insights</h2>
            {loading.crypto ? (
              <p>Loading cryptocurrency data...</p>
            ) : error.crypto ? (
              <p className="error">{error.crypto}</p>
            ) : (
              <>
                {Array.isArray(cryptoData) && cryptoData.filter(coin => `${coin.symbol}/USD` === selectedCoin).map((coin) => (
                  coin && coin.quote && coin.quote.USD ? (
                    <div key={coin.id} className="coin-detail">
                      <div className="coin-header">
                        <div className="coin-icon">{coin.symbol.charAt(0)}</div>
                        <div className="coin-name-price">
                          <h3>{coin.name} <span className="coin-symbol">({coin.symbol})</span></h3>
                          <div className="coin-price">${coin.quote.USD.price.toFixed(2)}
                            <span className={coin.quote.USD.percent_change_24h > 0 ? 'positive' : 'negative'}>
                              {coin.quote.USD.percent_change_24h > 0 ? '+' : ''}{coin.quote.USD.percent_change_24h.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="coin-metrics">
                        <div className="metric">
                          <span className="metric-label">Market cap</span>
                          <span className="metric-value">${(coin.quote.USD.market_cap / 1e9).toFixed(2)}B</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Market supply</span>
                          <span className="metric-value">{(coin.total_supply / 1e6).toFixed(2)}M</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Circulating supply</span>
                          <span className="metric-value">{(coin.circulating_supply / 1e6).toFixed(2)}M</span>
                        </div>
                      </div>
                      <div className="fundamental-analysis">
                        <h4>Fundamental Analysis</h4>
                        {loadingFundamental ? (
                          <p>Loading fundamental data...</p>
                        ) : errorFundamental ? (
                          <p className="error">{errorFundamental}</p>
                        ) : fundamentalData ? (
                          <>
                            <div className="metric">
                              <span className="metric-label">Market Cap</span>
                              <span className="metric-value">${fundamentalData.market_cap ? (fundamentalData.market_cap / 1e9).toFixed(2) + 'B' : 'N/A'}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Circulating Supply</span>
                              <span className="metric-value">{fundamentalData.circulating_supply ? fundamentalData.circulating_supply.toLocaleString() : 'N/A'}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Total Supply</span>
                              <span className="metric-value">{fundamentalData.total_supply ? fundamentalData.total_supply.toLocaleString() : 'N/A'}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Max Supply</span>
                              <span className="metric-value">{fundamentalData.max_supply ? fundamentalData.max_supply.toLocaleString() : 'N/A'}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">ROI (Since Beginning)</span>
                              <span className={
                                fundamentalData.roi_since_beginning !== null && fundamentalData.roi_since_beginning !== undefined
                                  ? fundamentalData.roi_since_beginning > 0
                                    ? "metric-value positive"
                                    : fundamentalData.roi_since_beginning < 0
                                      ? "metric-value negative"
                                      : "metric-value"
                                  : "metric-value"
                              }>
                                {fundamentalData.roi_since_beginning !== null && fundamentalData.roi_since_beginning !== undefined ? (fundamentalData.roi_since_beginning > 0 ? '+' : '') + fundamentalData.roi_since_beginning.toFixed(2) + '%' : 'N/A'}
                              </span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">ROI (1 Year)</span>
                              <span className={
                                fundamentalData.roi_1y !== null && fundamentalData.roi_1y !== undefined
                                  ? fundamentalData.roi_1y > 0
                                    ? "metric-value positive"
                                    : fundamentalData.roi_1y < 0
                                      ? "metric-value negative"
                                      : "metric-value"
                                  : "metric-value"
                              }>
                                {fundamentalData.roi_1y !== null && fundamentalData.roi_1y !== undefined ? (fundamentalData.roi_1y > 0 ? '+' : '') + fundamentalData.roi_1y.toFixed(2) + '%' : 'N/A'}
                              </span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">ROI (90 Days)</span>
                              <span className={
                                fundamentalData.roi_90d !== null && fundamentalData.roi_90d !== undefined
                                  ? fundamentalData.roi_90d > 0
                                    ? "metric-value positive"
                                    : fundamentalData.roi_90d < 0
                                      ? "metric-value negative"
                                      : "metric-value"
                                  : "metric-value"
                              }>
                                {fundamentalData.roi_90d !== null && fundamentalData.roi_90d !== undefined ? (fundamentalData.roi_90d > 0 ? '+' : '') + fundamentalData.roi_90d.toFixed(2) + '%' : 'N/A'}
                              </span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">ATH</span>
                              <span className="metric-value">{fundamentalData.ath !== null && fundamentalData.ath !== undefined ? '$' + Number(fundamentalData.ath).toLocaleString() : 'N/A'}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">ATL</span>
                              <span className="metric-value">{fundamentalData.atl !== null && fundamentalData.atl !== undefined ? '$' + Number(fundamentalData.atl).toLocaleString() : 'N/A'}</span>
                            </div>
                          </>
                        ) : (
                          <p>No fundamental data available.</p>
                        )}
                      </div>
                    </div>
                  ) : null
                ))}
              </>
            )}
          </div>

          <div className="price-variation-section dashboard-section">
            <h2>Price Variation</h2>
            {loading.historical ? (
              <p>Loading historical data...</p>
            ) : error.historical ? (
              <p className="error">{error.historical}</p>
            ) : (
              <div className="price-chart">
                {/* Simple price chart visualization would go here */}
                <div className="price-alerts">
                  <div className="alert positive">
                    <span className="alert-value">+897%</span>
                    <span className="alert-label">Price spike detected</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <RiskAssessment symbol={selectedCoin.replace('/', '_')} />
        </div>

        <div className="dashboard-column">
          <div className="technical-analysis-section dashboard-section">
            <h2>Technical Analysis</h2>
            <div className="chart-tabs">
              <button className="chart-tab active">RSI</button>
              <button className="chart-tab">MACD</button>
              <button className="chart-tab">BB</button>
              <button className="chart-tab">EQLLA</button>
            </div>
            <TradingViewChart 
              symbol={tradingViewSymbol} 
              theme="dark" 
              interval="1D" 
              height={400} 
            />
          </div>

          <div className="day-trading-section dashboard-section">
            <h2>Day Trading Insights</h2>
            {loadingDayTrading ? (
              <p>Loading day trading insight...</p>
            ) : errorDayTrading ? (
              <p className="error">{errorDayTrading}</p>
            ) : dayTradingInsight ? (
              <div className="trading-insights-grid">
                <div className="trading-insight">
                  <span className="insight-label">Entry Point</span>
                  <span className="insight-value">${dayTradingInsight.entry_point ? dayTradingInsight.entry_point.toLocaleString(undefined, {maximumFractionDigits: 2}) : 'N/A'}</span>
                </div>
                <div className="trading-insight">
                  <span className="insight-label">Exit Point</span>
                  <span className="insight-value">${dayTradingInsight.exit_point ? dayTradingInsight.exit_point.toLocaleString(undefined, {maximumFractionDigits: 2}) : 'N/A'}</span>
                </div>
                <div className="trading-insight">
                  <span className="insight-label">Trend</span>
                  <span className={dayTradingInsight.trend === 'bullish' ? 'insight-value bullish' : dayTradingInsight.trend === 'bearish' ? 'insight-value bearish' : 'insight-value'}>
                    {dayTradingInsight.trend ? dayTradingInsight.trend.charAt(0).toUpperCase() + dayTradingInsight.trend.slice(1) : 'N/A'}
                  </span>
                </div>
                <div className="trading-insight">
                  <span className="insight-label">Confidence
                    <span title="Confidence is based on model uncertainty (lower uncertainty = higher confidence)." style={{marginLeft: 6, cursor: 'help', color: '#888'}}>?</span>
                  </span>
                  <span className="insight-value">{dayTradingInsight.confidence !== undefined && dayTradingInsight.confidence !== null ? (dayTradingInsight.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                </div>
                {dayTradingInsight.uncertainty !== undefined && (
                  <div className="trading-insight">
                    <span className="insight-label">Uncertainty
                      <span title="Uncertainty is the standard deviation of model predictions (lower is better)." style={{marginLeft: 6, cursor: 'help', color: '#888'}}>?</span>
                    </span>
                    <span className="insight-value">{Number(dayTradingInsight.uncertainty).toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                  </div>
                )}
              </div>
            ) : (
              <p>No day trading insight available.</p>
            )}
          </div>

          <div className="news-section dashboard-section">
            <h2>Latest Crypto News</h2>
            {loading.news ? (
              <p>Loading news...</p>
            ) : error.news ? (
              <p className="error">{error.news}</p>
            ) : (
              <div className="news-list">
                {(Array.isArray(newsData) ? newsData : []).slice(0, 5).filter(Boolean).map((news: NewsItem) => (
                  <div key={news.id} className="news-item">
                    <h3><a href={news.url} target="_blank" rel="noopener noreferrer">{news.title}</a></h3>
                    <p className="news-meta">
                      Source: {news.source.title} | Published: {new Date(news.published_at).toLocaleString()}
                    </p>
                    {Array.isArray(news.currencies) && news.currencies.length > 0 && (
                      <div className="news-currencies">
                        Related: {news.currencies.map(c => c.code).join(', ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="crypto-table-section dashboard-section">
        <h2>Top Cryptocurrencies</h2>
        {loading.crypto ? (
          <p>Loading cryptocurrency data...</p>
        ) : error.crypto ? (
          <p className="error">{error.crypto}</p>
        ) : (
          <table className="crypto-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Name</th>
                <th>Symbol</th>
                <th>Price (USD)</th>
                <th>24h Change</th>
                <th>Market Cap</th>
              </tr>
            </thead>
            <tbody>
              {Array.isArray(cryptoData) && cryptoData.length > 0 && cryptoData.slice(0, 20).filter(Boolean).map((coin: Coin) => (
                coin && coin.quote && coin.quote.USD ? (
                  <tr key={coin.id}>
                    <td>{coin.cmc_rank}</td>
                    <td><Link to={`/coin/${coin.id}`}>{coin.name}</Link></td>
                    <td>{coin.symbol}</td>
                    <td>${coin.quote.USD.price.toFixed(2)}</td>
                    <td className={coin.quote.USD.percent_change_24h > 0 ? 'positive' : 'negative'}>
                      {coin.quote.USD.percent_change_24h > 0 ? '+' : ''}{coin.quote.USD.percent_change_24h.toFixed(2)}%
                    </td>
                    <td>${(coin.quote.USD.market_cap / 1e9).toFixed(2)}B</td>
                  </tr>
                ) : null
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default CryptoDataDisplay;
