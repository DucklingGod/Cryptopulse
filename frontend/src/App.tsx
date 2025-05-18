import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import CryptoDataDisplay from './components/CryptoDataDisplay';
import CoinDetailView from './components/CoinDetailView';

// Header component with navigation
const Header = () => (
  <header className="app-header">
    <div className="logo">CryptoDash</div>
    <nav>
      <Link to="/">Dashboard</Link>
      <Link to="/coins">Coins</Link>
      <Link to="/exchanges">Exchanges</Link>
      <Link to="/defi">DeFi</Link>
      <Link to="/nft">NFT</Link>
      <Link to="/portfolio">Portfolio</Link>
      <Link to="/watchlist">Watchlist</Link>
    </nav>
    <div className="user-controls">
      <button id="theme-toggle">Toggle Theme</button>
      <input type="search" placeholder="Search..." />
    </div>
  </header>
);

const Footer = () => (
  <footer className="app-footer">
    <p>&copy; 2025 CryptoDash. All rights reserved.</p>
    <div className="feedback-link"><Link to="/feedback">Feedback</Link></div>
  </footer>
);

function App() {
  const [theme, setTheme] = useState('light'); // Default theme

  useEffect(() => {
    document.body.className = theme + '-theme';
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Attach event listener for theme toggle button after component mounts
  useEffect(() => {
    const themeToggleButton = document.getElementById('theme-toggle');
    if (themeToggleButton) {
      themeToggleButton.addEventListener('click', toggleTheme);
    }
    // Cleanup listener on component unmount
    return () => {
      if (themeToggleButton) {
        themeToggleButton.removeEventListener('click', toggleTheme);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array means this effect runs once on mount and cleanup on unmount

  return (
    <Router>
      <div className={`app-container ${theme}-theme`}>
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<CryptoDataDisplay />} />
            <Route path="/coin/:coinId" element={<CoinDetailView />} />
            {/* Other routes will be added as we develop more features */}
            <Route path="*" element={<CryptoDataDisplay />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
