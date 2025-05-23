# Cryptopulse

A comprehensive real-time cryptocurrency market analysis dashboard with advanced risk assessment, technical analysis, and trading insights.

![Cryptopulse Dashboard](https://github.com/DucklingGod/Cryptopulse/raw/master/screenshots/dashboard.png)

## Features

### Coin Insights
- Detailed cryptocurrency information including market capitalization, circulating supply, and price data
- Fundamental analysis metrics (P/E ratio, ROI, historical performance)
- Real-time price updates from CoinMarketCap API

### Technical Analysis
- Interactive TradingView charts integration
- Technical indicators (RSI, MACD, Bollinger Bands)
- Customizable chart timeframes and visualization options

### Risk Assessment
- Comprehensive risk factor analysis based on multiple data points
- Real-time Fear & Greed Index integration
- Market trend assessment (bullish/bearish indicators for different timeframes)
- Volatility tracking and external factor analysis

### Day Trading Insights
- Entry and exit point recommendations
- Trade volume analysis
- Price spike detection and alerts

### News Integration
- Latest cryptocurrency news from trusted sources
- Sentiment analysis of news articles
- Related currency tagging for targeted information

## Technology Stack

### Frontend
- React with TypeScript
- Vite for fast development and building
- TradingView Charts integration
- Responsive design with CSS Grid and Flexbox

### Backend
- Flask (Python) RESTful API
- Data processing with Pandas and NumPy
- ML integration
- Multiple API integrations (CoinMarketCap, CryptoPanic, Alternative.me)

## Getting Started

### Prerequisites
- Node.js (v16+)
- Python 3.10+
- API keys for:
  - CoinMarketCap
  - CryptoPanic
  - TwelveData

### Installation

#### Backend Setup
1. Clone the repository
   ```bash
   git clone https://github.com/DucklingGod/Cryptopulse.git
   cd Cryptopulse
   ```

2. Set up Python virtual environment
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys
   - Create a `.env` file in the backend directory
   - Add your API keys:
     ```
     CMC_API_KEY=your_coinmarketcap_api_key
     CRYPTOPANIC_API_KEY=your_cryptopanic_api_key
     TWELVEDATA_API_KEY=your_twelvedata_api_key
     ```

5. Start the backend server
   ```bash
   python src/main.py
   ```
   The backend will run on http://localhost:5000

#### Frontend Setup
1. Navigate to the frontend directory
   ```bash
   cd ../frontend
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Configure environment
   - Create a `.env` file in the frontend directory
   - Add:
     ```
     VITE_API_BASE_URL=http://localhost:5000/api
     ```

4. Start the development server
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:5173

## Project Structure

```
cryptopulse/
├── backend/
│   ├── data/                  # Data storage
│   ├── models/                # ML models
│   ├── src/
│   │   ├── risk_assessment/   # Risk calculation logic
│   │   ├── routes/            # API endpoints
│   │   ├── main.py            # Entry point
│   │   └── mock_ml_model.py   # Mock ML implementation
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Utility functions
│   │   ├── App.tsx            # Main application
│   │   └── types.ts           # TypeScript interfaces
│   ├── .env                   # Environment variables
│   └── package.json           # Node dependencies
└── README.md                  # This file
```

## API Endpoints

### Cryptocurrency Data
- `GET /api/cmc/listings` - Get top cryptocurrencies
- `GET /api/cmc/quotes?symbol={symbol}` - Get detailed quote for a specific cryptocurrency

### Risk Assessment
- `GET /api/risk/assessment?symbol={symbol}` - Get risk assessment for a cryptocurrency
- `GET /api/risk/fear_greed` - Get current Fear & Greed Index

### News
- `GET /api/cryptopanic/posts` - Get latest cryptocurrency news

### Technical Data
- `GET /api/twelvedata/time_series?symbol={symbol}` - Get historical price data

## Future Enhancements

- More robust machine learning model integration for price prediction
- User authentication and personalized watchlists
- Portfolio tracking and performance analysis
- Mobile application version
- Advanced alerting system for price movements
- Social sentiment analysis from Twitter/Reddit

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CoinMarketCap](https://coinmarketcap.com/) for cryptocurrency data
- [TradingView](https://www.tradingview.com/) for chart widgets
- [Alternative.me](https://alternative.me/) for Fear & Greed Index
- [CryptoPanic](https://cryptopanic.com/) for cryptocurrency news
