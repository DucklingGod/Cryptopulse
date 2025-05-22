import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import CryptoDataDisplay from './components/CryptoDataDisplay';
import CoinDetailView from './components/CoinDetailView';
import logoImg from './assets/cryptodash-logo.png';
import { FiSun, FiMoon } from 'react-icons/fi';

// Header component with navigation
const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);
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
    <header className="app-header">
      <div className="logo">
        <img src={logoImg} alt="CryptoDash Logo" style={{ height: 128, width: 'auto', maxWidth: 680 }} />
      </div>
      <nav className="dropdown-nav">
        <button className="dropdown-toggle" onClick={() => setMenuOpen((open) => !open)}>
          Menu &#9776;
        </button>
        {menuOpen && (
          <div className="dropdown-menu">
            <Link to="/" onClick={() => setMenuOpen(false)}>Dashboard</Link>
            <Link to="/coins" onClick={() => setMenuOpen(false)}>Coins</Link>
            <Link to="/exchanges" onClick={() => setMenuOpen(false)}>Exchanges</Link>
            <Link to="/defi" onClick={() => setMenuOpen(false)}>DeFi</Link>
            <Link to="/nft" onClick={() => setMenuOpen(false)}>NFT</Link>
            <Link to="/portfolio" onClick={() => setMenuOpen(false)}>Portfolio</Link>
            <Link to="/watchlist" onClick={() => setMenuOpen(false)}>Watchlist</Link>
          </div>
        )}
      </nav>
      <div className="user-controls">
        <button id="theme-toggle" 
          style={{ 
            background: theme === 'light' ? '#fff' : '#232323',
            border: '1px solid #444',
            borderRadius: 8,
            cursor: 'pointer',
            fontSize: 28,
            padding: '6px 12px',
            marginRight: 12,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 1px 4px rgba(0,0,0,0.08)'
          }} 
          aria-label="Toggle theme">
          {theme === 'light' ? <FiSun color="#111" /> : <FiMoon color="#fff" />}
        </button>
        <input type="search" placeholder="Search..." style={{ minWidth: 220, width: '100%', maxWidth: 340, fontSize: 18, padding: '8px 14px', borderRadius: 8, border: '1px solid #444', background: '#181818', color: '#fff' }} />
      </div>
    </header>
  );
};

const Footer = () => (
  <footer className="app-footer">
    <p>&copy; 2025 CryptoPulse. All rights reserved.</p>
    <div className="feedback-link"><Link to="/feedback">Feedback</Link></div>
  </footer>
);

function App() {
  return (
    <Router>
      <div className={`app-container`}>
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
