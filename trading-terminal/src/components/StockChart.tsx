import React, { useEffect, useRef, useState } from 'react';

interface StockBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface StockChartProps {
  symbol: string;
  data: StockBar[];
  width?: number;
  height?: number;
}

export const StockChart: React.FC<StockChartProps> = ({ 
  symbol, 
  data, 
  width = 800, 
  height = 400 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (data.length > 0) {
      drawChart();
    }
  }, [data, width, height]);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Chart dimensions
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Find min/max values
    const prices = data.flatMap(bar => [bar.high, bar.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i * chartHeight / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Price labels
      const price = maxPrice - (i * priceRange / 5);
      ctx.fillStyle = '#666';
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), padding - 10, y + 4);
    }

    // Vertical grid lines
    const timeStep = Math.max(1, Math.floor(data.length / 8));
    for (let i = 0; i < data.length; i += timeStep) {
      const x = padding + (i * chartWidth / (data.length - 1));
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      
      // Time labels
      const time = new Date(data[i].timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      });
      ctx.fillStyle = '#666';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(time, x, height - padding + 15);
    }

    // Draw candlesticks
    const candleWidth = Math.max(2, chartWidth / data.length * 0.6);
    
    data.forEach((bar, index) => {
      const x = padding + (index * chartWidth / (data.length - 1));
      const openY = padding + ((maxPrice - bar.open) * chartHeight / priceRange);
      const closeY = padding + ((maxPrice - bar.close) * chartHeight / priceRange);
      const highY = padding + ((maxPrice - bar.high) * chartHeight / priceRange);
      const lowY = padding + ((maxPrice - bar.low) * chartHeight / priceRange);
      
      // Candle color (green if close > open, red if close < open)
      const isGreen = bar.close >= bar.open;
      ctx.fillStyle = isGreen ? '#00ff00' : '#ff0000';
      ctx.strokeStyle = isGreen ? '#00ff00' : '#ff0000';
      ctx.lineWidth = 1;
      
      // High-low line
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();
      
      // Candle body
      const bodyHeight = Math.abs(closeY - openY);
      const bodyY = Math.min(openY, closeY);
      
      if (bodyHeight < 1) {
        // Doji - draw horizontal line
        ctx.beginPath();
        ctx.moveTo(x - candleWidth/2, bodyY);
        ctx.lineTo(x + candleWidth/2, bodyY);
        ctx.stroke();
      } else {
        // Regular candle
        ctx.fillRect(x - candleWidth/2, bodyY, candleWidth, bodyHeight);
      }
    });

    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    
    // Y axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();
    
    // X axis
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Title
    ctx.fillStyle = '#ffa500';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${symbol.toUpperCase()} - Stock Chart`, padding, 30);

    // Current price
    if (data.length > 0) {
      const lastBar = data[data.length - 1];
      const change = lastBar.close - data[0].open;
      const changePercent = (change / data[0].open * 100);
      
      ctx.fillStyle = change >= 0 ? '#00ff00' : '#ff0000';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(
        `$${lastBar.close.toFixed(2)} (${change >= 0 ? '+' : ''}${change.toFixed(2)} ${changePercent.toFixed(2)}%)`,
        width - padding,
        30
      );
    }
  };

  return (
    <div style={{ 
      border: '1px solid #333', 
      borderRadius: '4px',
      backgroundColor: '#0a0a0a',
      padding: '10px'
    }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ display: 'block', maxWidth: '100%' }}
      />
      {data.length === 0 && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#666',
          fontSize: '14px'
        }}>
          {isLoading ? 'Loading chart data...' : 'No data available'}
        </div>
      )}
    </div>
  );
};