import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { createChart, ColorType, LineStyle, type IChartApi } from 'lightweight-charts';
import { useTradingStore } from '../stores/tradingStore';

const PanelContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 10px;
`;

const PanelTitle = styled.div`
  color: #ffa500;
  font-size: 11px;
  font-weight: bold;
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChartContainer = styled.div`
  flex: 1;
  position: relative;
`;

const SymbolSelector = styled.select`
  background: #000;
  color: #00ff00;
  border: 1px solid #333;
  padding: 2px 5px;
  font-family: 'Courier New', monospace;
  font-size: 10px;
`;

const ChartPanel: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any | null>(null);
  const [selectedSymbol, setSelectedSymbol] = React.useState('SPY');
  
  const marketData = useTradingStore((state) => state.marketData);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#00ff00',
        fontSize: 10,
        fontFamily: 'Courier New, monospace'
      },
      grid: {
        vertLines: { color: '#333' },
        horzLines: { color: '#333' }
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#ffa500',
          style: LineStyle.Dashed
        },
        horzLine: {
          width: 1,
          color: '#ffa500',
          style: LineStyle.Dashed
        }
      },
      rightPriceScale: {
        borderColor: '#333'
      },
      timeScale: {
        borderColor: '#333',
        timeVisible: true,
        secondsVisible: false
      }
    });

    const candlestickSeries = (chart as any).addCandlestickSeries({
      upColor: '#00ff00',
      downColor: '#ff0000',
      borderDownColor: '#ff0000',
      borderUpColor: '#00ff00',
      wickDownColor: '#ff0000',
      wickUpColor: '#00ff00'
    });

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    // Generate some sample data for the chart
    const generateSampleData = () => {
      const data = [];
      let basePrice = marketData[selectedSymbol]?.price || 450;
      const now = new Date();
      
      for (let i = 100; i >= 0; i--) {
        const time = Math.floor((now.getTime() - i * 60000) / 1000);
        const variation = (Math.random() - 0.5) * 2;
        const open = basePrice + variation;
        const high = open + Math.random() * 2;
        const low = open - Math.random() * 2;
        const close = low + Math.random() * (high - low);
        
        data.push({
          time,
          open,
          high,
          low,
          close
        });
        
        basePrice = close;
      }
      
      return data;
    };

    candlestickSeries.setData(generateSampleData());

    const handleResize = () => {
      chart.applyOptions({
        width: chartContainerRef.current?.clientWidth,
        height: chartContainerRef.current?.clientHeight
      });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [selectedSymbol, marketData]);

  return (
    <PanelContainer>
      <PanelTitle>
        Price Chart
        <SymbolSelector 
          value={selectedSymbol} 
          onChange={(e) => setSelectedSymbol(e.target.value)}
        >
          <option value="SPY">SPY</option>
          <option value="AAPL">AAPL</option>
          <option value="MSFT">MSFT</option>
          <option value="TSLA">TSLA</option>
          <option value="NVDA">NVDA</option>
          {Object.keys(marketData).map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </SymbolSelector>
      </PanelTitle>
      <ChartContainer ref={chartContainerRef} />
    </PanelContainer>
  );
};

export default ChartPanel;