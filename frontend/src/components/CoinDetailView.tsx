import { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import { Coin, NewsItem, HistoricalData } from '../types';
import TechnicalAnalysisChart from './TechnicalAnalysisChart'; // Import the new chart component
import './CoinDetailView.css';

const CoinDetailView = () => {
  const { coinId } = useParams<{ coinId: string }>();
  const [coinDetails, setCoinDetails] = useState<Coin | null>(null);
  const [coinNews, setCoinNews] = useState<NewsItem[]>([]);
  const [coinHistorical, setCoinHistorical] = useState<HistoricalData>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState('1day'); // Default timeframe for charts

  useEffect(() => {
    const fetchDetails = async () => {
      if (!coinId) return;
      setLoading(true);
      setError(null);
      try {
        const listingsResponse = await axios.get<{ data: Coin[] }>(`/api/cmc/listings?limit=100`);
        const foundCoin = listingsResponse.data.data.find(c => c.id.toString() === coinId || c.symbol.toLowerCase() === coinId.toLowerCase());
        
        if (foundCoin) {
          setCoinDetails(foundCoin);

          const newsResponse = await axios.get<{ results: NewsItem[] }>(`/api/cryptopanic/posts?currencies=${foundCoin.symbol}`);
          setCoinNews(newsResponse.data.results.slice(0, 5));

          // Fetch more data for charting - e.g., 90 days for daily, more for smaller intervals
          let outputSize = "90";
          if (timeframe === "1hour" || timeframe === "1min") {
            outputSize = "100"; // Adjust as needed for smaller timeframes
          }
          const historicalResponse = await axios.get<HistoricalData>(`/api/twelvedata/time_series?symbol=${foundCoin.symbol}/USD&interval=${timeframe}&outputsize=${outputSize}`);
          setCoinHistorical(historicalResponse.data);

        } else {
          setError('Coin not found.');
        }

      } catch (err) {
        console.error("Error fetching coin details:", err);
        setError('Failed to fetch coin details.');
      }
      setLoading(false);
    };
    fetchDetails();
  }, [coinId, timeframe]);

  if (loading) return <p>Loading coin details...</p>;
  if (error) return <p className="error">{error}</p>;
  if (!coinDetails) return <p>No coin data available.</p>;

  const handleTimeframeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setTimeframe(e.target.value);
  };

  return (
    <div className="coin-detail-view">
      <div className="coin-header">
        <h1>{coinDetails.name} ({coinDetails.symbol})</h1>
        <p className="coin-price">${coinDetails.quote.USD.price.toFixed(2)}</p>
        <span className={coinDetails.quote.USD.percent_change_24h > 0 ? 'positive' : 'negative'}>
          {coinDetails.quote.USD.percent_change_24h.toFixed(2)}% (24h)
        </span>
      </div>

      <div className="coin-insights-grid">
        {/* Insight cards remain the same */}
        <div className="insight-card">
          <h3>Market Overview</h3>
          <p><strong>Market Cap:</strong> ${(coinDetails.quote.USD.market_cap / 1e9).toFixed(2)}B</p>
          <p><strong>Fully Diluted Market Cap:</strong> ${(coinDetails.quote.USD.fully_diluted_market_cap / 1e9).toFixed(2)}B</p>
          <p><strong>Volume (24h):</strong> ${(coinDetails.quote.USD.volume_24h / 1e9).toFixed(2)}B</p>
          <p><strong>Circulating Supply:</strong> {coinDetails.circulating_supply.toLocaleString()}</p>
          <p><strong>Total Supply:</strong> {coinDetails.total_supply.toLocaleString()}</p>
          <p><strong>Max Supply:</strong> {coinDetails.max_supply ? coinDetails.max_supply.toLocaleString() : 'N/A'}</p>
          <p><strong>CMC Rank:</strong> {coinDetails.cmc_rank}</p>
        </div>

        <div className="insight-card">
          <h3>Fundamental Metrics (Placeholder)</h3>
          <p><strong>P/E Ratio:</strong> N/A (Requires specific data source)</p>
          <p><strong>ROI (1yr):</strong> N/A (Requires historical price and calculation)</p>
        </div>
        
        <div className="insight-card" id="historical-performance-card">
            <h3>Historical Performance (Last 90 Days - Daily)</h3>
            {coinHistorical.values && coinHistorical.values.length > 0 && timeframe === '1day' ? (
                <ul>
                    <li><strong>Price Change (90d):</strong> 
                        {(() => {
                            const first = parseFloat(coinHistorical.values![coinHistorical.values!.length - 1].open);
                            const last = parseFloat(coinHistorical.values![0].close);
                            const change = ((last - first) / first) * 100;
                            return <span className={change > 0 ? 'positive' : 'negative'}>{change.toFixed(2)}%</span>;
                        })()}
                    </li>
                </ul>
            ) : <p>Historical performance data (90-day daily) not available or different timeframe selected.</p>}
        </div>

        <div className="insight-card" id="recent-news-card">
          <h3>Recent News</h3>
          {coinNews.length > 0 ? (
            <ul>
              {coinNews.map(news => (
                <li key={news.id}>
                  <a href={news.url} target="_blank" rel="noopener noreferrer">{news.title}</a>
                  <span className="news-source"> ({news.source.title})</span>
                </li>
              ))}
            </ul>
          ) : <p>No recent news found for {coinDetails.symbol}.</p>}
        </div>
      </div>

      <div className="technical-analysis-section">
        <h2>Technical Analysis</h2>
        <div className="timeframe-selector">
          <label htmlFor="timeframe-select">Select Timeframe: </label>
          <select id="timeframe-select" value={timeframe} onChange={handleTimeframeChange}>
            <option value="1min">1 Minute</option>
            <option value="1hour">1 Hour</option>
            <option value="1day">1 Day</option>
            <option value="1week">1 Week</option>
          </select>
        </div>
        {coinHistorical.values && coinHistorical.values.length > 0 ? (
          <TechnicalAnalysisChart historicalData={coinHistorical.values} timeframe={timeframe} />
        ) : (
          <p>Loading chart data or not enough data for selected timeframe...</p>
        )}
      </div>

    </div>
  );
};

export default CoinDetailView;

