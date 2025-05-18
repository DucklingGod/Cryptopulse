import React, { useEffect, useRef } from 'react';
import './TradingViewChart.css';

declare global {
  interface Window {
    TradingView: any;
  }
}

interface TradingViewChartProps {
  symbol: string;
  theme?: 'light' | 'dark';
  interval?: string;
  height?: number;
  width?: string;
}

const TradingViewChart: React.FC<TradingViewChartProps> = ({
  symbol = 'BINANCE:BTCUSDT',
  theme = 'dark',
  interval = '1D',
  height = 500,
  width = '100%'
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptRef = useRef<HTMLScriptElement | null>(null);
  const widgetRef = useRef<any>(null);

  useEffect(() => {
    // Load TradingView widget script if not already loaded
    if (!document.getElementById('tradingview-widget-script') && !scriptRef.current) {
      const script = document.createElement('script');
      script.id = 'tradingview-widget-script';
      script.src = 'https://s3.tradingview.com/tv.js';
      script.async = true;
      script.onload = initializeWidget;
      document.body.appendChild(script);
      scriptRef.current = script;
    } else {
      // If script is already loaded, initialize widget directly
      initializeWidget();
    }

    return () => {
      // Clean up widget when component unmounts
      if (widgetRef.current) {
        try {
          // Some widgets might have their own cleanup methods
          if (widgetRef.current.remove) {
            widgetRef.current.remove();
          }
        } catch (error) {
          console.error('Error cleaning up TradingView widget:', error);
        }
        widgetRef.current = null;
      }
    };
  }, [symbol, theme, interval]);

  const initializeWidget = () => {
    if (!containerRef.current || !window.TradingView) {
      // If container ref is not available or TradingView is not loaded yet, retry after a short delay
      setTimeout(initializeWidget, 100);
      return;
    }

    // Clear previous widget if exists
    if (containerRef.current) {
      containerRef.current.innerHTML = '';
    }

    // Create new widget
    widgetRef.current = new window.TradingView.widget({
      container_id: containerRef.current.id,
      symbol: symbol,
      interval: interval,
      theme: theme,
      style: '1',
      locale: 'en',
      toolbar_bg: theme === 'dark' ? '#1E1E1E' : '#f1f3f6',
      enable_publishing: false,
      allow_symbol_change: true,
      save_image: false,
      height: height,
      width: width,
      hide_side_toolbar: false,
      studies: [
        'MASimple@tv-basicstudies',
        'RSI@tv-basicstudies',
        'MACD@tv-basicstudies'
      ],
      show_popup_button: true,
      popup_width: '1000',
      popup_height: '650',
      hide_volume: false,
      withdateranges: true,
      details: true,
      calendar: true,
      hotlist: true,
      news: [
        'headlines'
      ]
    });
  };

  return (
    <div className="tradingview-chart-container">
      <div 
        id="tradingview-widget" 
        ref={containerRef} 
        className="tradingview-chart"
        style={{ height: `${height}px`, width: width }}
      />
    </div>
  );
};

export default TradingViewChart;
