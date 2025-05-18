import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { calculateRSI, calculateMACD, calculateBollingerBands } from '../utils/technicalIndicators'; // Updated import, removed unused MACDOutput, BollingerBandsOutput
import { HistoricalValue } from '../types';
import './TechnicalAnalysisChart.css';

interface TechnicalAnalysisChartProps {
  historicalData: HistoricalValue[];
  timeframe: string;
}

const TechnicalAnalysisChart: React.FC<TechnicalAnalysisChartProps> = ({ historicalData, timeframe }) => {
  if (!historicalData || historicalData.length === 0) {
    return <p>No historical data available to display chart.</p>;
  }

  const closePrices = historicalData.map(d => parseFloat(d.close));

  const rsiResult = calculateRSI(closePrices);
  const macdResult = calculateMACD(closePrices);
  const bollingerBandsResult = calculateBollingerBands(closePrices);

  const chartData = historicalData.map((data, index) => {
    const rsiValue = rsiResult[index];
    const macdPoint = macdResult[index];
    const bbPoint = bollingerBandsResult[index];

    return {
      datetime: new Date(data.datetime).toLocaleDateString(),
      price: parseFloat(data.close),
      rsi: rsiValue,
      macdLine: macdPoint?.MACD,
      signalLine: macdPoint?.signal,
      histogram: macdPoint?.histogram,
      bbUpper: bbPoint?.upper,
      bbMiddle: bbPoint?.middle,
      bbLower: bbPoint?.lower,
    };
  });

  const showBollingerBands = bollingerBandsResult.some(d => d.upper !== undefined && d.middle !== undefined && d.lower !== undefined);
  const showRSI = rsiResult.some(r => r !== undefined);
  const showMACD = macdResult.some(m => m.MACD !== undefined);


  return (
    <div className="technical-chart-container">
      <h4>Price Chart ({timeframe})</h4>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="datetime" />
          <YAxis yAxisId="left" orientation="left" stroke="#8884d8" domain={['auto', 'auto']} />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="price" stroke="#8884d8" name="Price" dot={false}/>
          {showBollingerBands && <Line yAxisId="left" type="monotone" dataKey="bbUpper" stroke="#ccc" name="BB Upper" dot={false} strokeDasharray="5 5"/>}
          {showBollingerBands && <Line yAxisId="left" type="monotone" dataKey="bbMiddle" stroke="#aaa" name="BB Middle" dot={false} />}
          {showBollingerBands && <Line yAxisId="left" type="monotone" dataKey="bbLower" stroke="#ccc" name="BB Lower" dot={false} strokeDasharray="5 5"/>}
        </LineChart>
      </ResponsiveContainer>

      {showRSI && (
        <>
          <h4>Relative Strength Index (RSI)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData.filter(d => d.rsi !== undefined)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="datetime" />
              <YAxis domain={[0, 100]}/>
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="rsi" stroke="#82ca9d" name="RSI" dot={false}/>
              <Line type="basis" dataKey={() => 70} stroke="#ff0000" dot={false} strokeDasharray="5 5" name="Overbought" legendType="none"/>
              <Line type="basis" dataKey={() => 30} stroke="#00ff00" dot={false} strokeDasharray="5 5" name="Oversold" legendType="none"/>
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {showMACD && (
        <>
          <h4>MACD</h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData.filter(d => d.macdLine !== undefined)} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="datetime" />
                <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                {/* <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" /> */}
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="macdLine" stroke="#ff7300" name="MACD Line" dot={false}/>
                <Line yAxisId="left" type="monotone" dataKey="signalLine" stroke="#387908" name="Signal Line" dot={false}/>
            </LineChart>
          </ResponsiveContainer>
          <ResponsiveContainer width="100%" height={150}>            
            <BarChart data={chartData.filter(d => d.histogram !== undefined)} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="datetime" />
                <YAxis yAxisId="right_hist" orientation="right" stroke="#a0a0a0" />
                <Tooltip />
                <Legend />
                <Bar yAxisId="right_hist" dataKey="histogram" name="Histogram">
                  {chartData.filter(d => d.histogram !== undefined).map((entry: any, index: number) => (
                    <rect key={`bar-${index}`} fill={entry.histogram! >= 0 ? '#26a69a' : '#ef5350'} />
                  ))}
                </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
};

export default TechnicalAnalysisChart;

