// @ts-nocheck
import { createGlobalStyle } from 'styled-components';
import { useState, useEffect, useRef } from 'react';
import webSocketService, { MarketDataMessage, AgentSignalMessage, PortfolioUpdateMessage, RiskAlertMessage, SystemStatusMessage } from './services/websocketService';

const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  body {
    font-family: 'Courier New', 'Consolas', monospace;
    background: #000000;
    color: #00ff00;
    font-size: 10px;
    letter-spacing: -0.5px;
  }
  
  #root {
    height: 100vh;
    width: 100vw;
  }
`;

// Interactive Command Line Component
const InteractiveCommandLine = ({ marketData, agents, setAgents, webSocketConnected }) => {
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState([]);
  const [output, setOutput] = useState([
    { type: 'system', text: 'HIVE TRADE QUANTUM TERMINAL v2.0 READY' },
    { type: 'system', text: 'Type HELP for available commands' }
  ]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const executeCommand = async (cmd) => {
    const parts = cmd.trim().toUpperCase().split(' ');
    const action = parts[0];
    
    // Add command to output
    setOutput(prev => [...prev, { type: 'command', text: `HTQT> ${cmd}` }]);
    
    try {
      switch (action) {
        case 'BUY':
        case 'SELL':
          if (parts.length >= 3) {
            const symbol = parts[1];
            const quantity = parseInt(parts[2]) || 100;
            const price = marketData[symbol]?.price;
            
            if (marketData[symbol]) {
              const orderValue = price * quantity;
              setOutput(prev => [...prev, 
                { type: 'success', text: `${action} ORDER EXECUTED` },
                { type: 'info', text: `Symbol: ${symbol}, Qty: ${quantity}, Price: $${price.toFixed(2)}` },
                { type: 'info', text: `Total Value: $${orderValue.toLocaleString()}` }
              ]);
            } else {
              setOutput(prev => [...prev, { type: 'error', text: `SYMBOL ${symbol} NOT FOUND` }]);
            }
          } else {
            setOutput(prev => [...prev, { type: 'error', text: 'USAGE: BUY/SELL <SYMBOL> <QTY>' }]);
          }
          break;
          
        case 'AGENT':
          if (parts.length >= 3) {
            const agentAction = parts[1]; // START/STOP/STATUS
            const agentName = parts[2].toLowerCase();
            
            if (agents[agentName]) {
              if (agentAction === 'START') {
                setAgents(prev => ({
                  ...prev,
                  [agentName]: { ...prev[agentName], status: 'ACTIVE' }
                }));
                setOutput(prev => [...prev, { type: 'success', text: `AGENT ${agentName.toUpperCase()} ACTIVATED` }]);
              } else if (agentAction === 'STOP') {
                setAgents(prev => ({
                  ...prev,
                  [agentName]: { ...prev[agentName], status: 'INACTIVE' }
                }));
                setOutput(prev => [...prev, { type: 'warning', text: `AGENT ${agentName.toUpperCase()} DEACTIVATED` }]);
              } else if (agentAction === 'STATUS') {
                const agent = agents[agentName];
                setOutput(prev => [...prev,
                  { type: 'info', text: `AGENT: ${agent.name}` },
                  { type: 'info', text: `STATUS: ${agent.status}` },
                  { type: 'info', text: `P&L: $${agent.pnl?.toLocaleString() || 'N/A'}` },
                  { type: 'info', text: `ACCURACY: ${agent.accuracy || 'N/A'}%` }
                ]);
              }
            } else {
              setOutput(prev => [...prev, { type: 'error', text: `AGENT ${agentName.toUpperCase()} NOT FOUND` }]);
            }
          } else {
            setOutput(prev => [...prev, { type: 'error', text: 'USAGE: AGENT START/STOP/STATUS <AGENT_NAME>' }]);
          }
          break;
          
        case 'PORTFOLIO':
          const totalPnL = Object.values(agents)
            .filter(agent => agent.pnl !== undefined)
            .reduce((sum, agent) => sum + agent.pnl, 0);
          const activeAgents = Object.values(agents).filter(agent => agent.status === 'ACTIVE').length;
          
          setOutput(prev => [...prev,
            { type: 'info', text: '=== PORTFOLIO SUMMARY ===' },
            { type: 'success', text: `TOTAL P&L: $${totalPnL.toLocaleString()}` },
            { type: 'info', text: `ACTIVE AGENTS: ${activeAgents}` },
            { type: 'info', text: `TRACKED INSTRUMENTS: ${Object.keys(marketData).length}` }
          ]);
          break;
          
        case 'MARKET':
          const topGainers = Object.values(marketData)
            .sort((a, b) => b.changePercent - a.changePercent)
            .slice(0, 3);
          const topLosers = Object.values(marketData)
            .sort((a, b) => a.changePercent - b.changePercent)
            .slice(0, 3);
          
          setOutput(prev => [...prev,
            { type: 'info', text: '=== MARKET SNAPSHOT ===' },
            { type: 'success', text: 'TOP GAINERS:' },
            ...topGainers.map(stock => ({ 
              type: 'success', 
              text: `${stock.symbol}: +${stock.changePercent.toFixed(2)}%` 
            })),
            { type: 'error', text: 'TOP LOSERS:' },
            ...topLosers.map(stock => ({ 
              type: 'error', 
              text: `${stock.symbol}: ${stock.changePercent.toFixed(2)}%` 
            }))
          ]);
          break;
          
        case 'STATUS':
          setOutput(prev => [...prev,
            { type: 'info', text: '=== SYSTEM STATUS ===' },
            { type: webSocketConnected ? 'success' : 'error', text: `WebSocket: ${webSocketConnected ? 'CONNECTED' : 'DISCONNECTED'}` },
            { type: 'info', text: `Symbols Tracked: ${Object.keys(marketData).length}` },
            { type: 'info', text: `Active Agents: ${Object.values(agents).filter(agent => agent.status === 'ACTIVE').length}` },
            { type: 'info', text: `Terminal: v2.0 READY` }
          ]);
          break;

        case 'CONNECT':
          if (!webSocketConnected) {
            setOutput(prev => [...prev, { type: 'info', text: 'Attempting to reconnect to backend...' }]);
            webSocketService.connect().catch(error => {
              setOutput(prev => [...prev, { type: 'error', text: `Connection failed: ${error.message}` }]);
            });
          } else {
            setOutput(prev => [...prev, { type: 'success', text: 'Already connected to backend' }]);
          }
          break;

        case 'SUBSCRIBE':
          if (parts.length >= 2) {
            const symbols = parts.slice(1).map(s => s.toUpperCase());
            if (webSocketConnected) {
              webSocketService.subscribeToSymbols(symbols);
              setOutput(prev => [...prev, 
                { type: 'success', text: `Subscribed to symbols: ${symbols.join(', ')}` }
              ]);
            } else {
              setOutput(prev => [...prev, { type: 'error', text: 'Cannot subscribe: WebSocket not connected' }]);
            }
          } else {
            setOutput(prev => [...prev, { type: 'error', text: 'USAGE: SUBSCRIBE <SYMBOL1> <SYMBOL2> ...' }]);
          }
          break;

        case 'HELP':
          setOutput(prev => [...prev,
            { type: 'info', text: '=== AVAILABLE COMMANDS ===' },
            { type: 'info', text: 'BUY <SYMBOL> <QTY> - Place buy order' },
            { type: 'info', text: 'SELL <SYMBOL> <QTY> - Place sell order' },
            { type: 'info', text: 'AGENT START/STOP <NAME> - Control agents' },
            { type: 'info', text: 'PORTFOLIO - Portfolio summary' },
            { type: 'info', text: 'MARKET - Market snapshot' },
            { type: 'info', text: 'STATUS - System status' },
            { type: 'info', text: 'CONNECT - Reconnect to backend' },
            { type: 'info', text: 'SUBSCRIBE <SYMBOLS> - Subscribe to symbols' },
            { type: 'info', text: 'CLEAR - Clear terminal' },
            { type: 'info', text: 'Available agents: momentum, meanreversion, optionssentiment, statarb, marketmaking' }
          ]);
          break;
          
        case 'CLEAR':
          setOutput([{ type: 'system', text: 'TERMINAL CLEARED' }]);
          break;
          
        default:
          setOutput(prev => [...prev, { type: 'error', text: `UNKNOWN COMMAND: ${cmd}` }]);
      }
    } catch (error) {
      setOutput(prev => [...prev, { type: 'error', text: `ERROR: ${error.message}` }]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!command.trim()) return;
    
    setHistory(prev => [...prev, command]);
    setHistoryIndex(-1);
    executeCommand(command);
    setCommand('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowUp' && history.length > 0) {
      e.preventDefault();
      const newIndex = Math.min(historyIndex + 1, history.length - 1);
      setHistoryIndex(newIndex);
      setCommand(history[history.length - 1 - newIndex]);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setCommand(history[history.length - 1 - newIndex]);
      } else {
        setHistoryIndex(-1);
        setCommand('');
      }
    }
  };

  return (
    <div style={{
      background: '#0a0a0a',
      borderTop: '1px solid #333',
      padding: '8px',
      fontSize: '10px',
      height: '120px',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Command output */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '8px',
        fontSize: '9px'
      }}>
        {output.slice(-8).map((line, i) => (
          <div key={i} style={{
            color: line.type === 'command' ? '#ffa500' :
                   line.type === 'success' ? '#00ff00' :
                   line.type === 'error' ? '#ff0000' :
                   line.type === 'warning' ? '#ffa500' :
                   line.type === 'system' ? '#00ffff' : '#ccc',
            marginBottom: '2px',
            fontFamily: 'Courier New, monospace'
          }}>
            {line.text}
          </div>
        ))}
      </div>
      
      {/* Command input */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center' }}>
        <span style={{ color: '#ffa500', marginRight: '8px', fontWeight: 'bold' }}>HTQT&gt;</span>
        <input
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: '#00ff00',
            fontFamily: 'Courier New, monospace',
            fontSize: '11px',
            flex: 1
          }}
          placeholder="Enter command (type HELP for commands)"
          autoComplete="off"
        />
      </form>
    </div>
  );
};

// Professional Chart Component
const ProfessionalChart = ({ symbol, marketData, timeframe, indicators }) => {
  const chartRef = useRef(null);
  const [candleData, setCandleData] = useState([]);
  const [priceData, setPriceData] = useState([]);
  const [volumeData, setVolumeData] = useState([]);

  useEffect(() => {
    // Generate realistic candlestick data
    const generateData = () => {
      const candles = [];
      const prices = [];
      const volumes = [];
      const currentPrice = marketData[symbol]?.price || 450;
      const now = Date.now();
      
      let basePrice = currentPrice * 0.95; // Start slightly lower
      
      for (let i = 200; i >= 0; i--) {
        const time = now - i * 60000;
        
        // More realistic price movement
        const trend = (Math.random() - 0.48) * 0.02; // Slight upward bias
        const volatility = (marketData[symbol]?.iv || 0.3) * 0.1;
        const noise = (Math.random() - 0.5) * volatility * basePrice;
        
        basePrice = Math.max(0.01, basePrice * (1 + trend) + noise);
        
        const open = basePrice;
        const close = open * (1 + (Math.random() - 0.5) * volatility);
        const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.5);
        const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.5);
        const volume = Math.random() * 1000000 + 100000;
        
        candles.push({ time, open, high, low, close, volume });
        prices.push({ time, price: close });
        volumes.push({ time, volume });
        
        basePrice = close;
      }
      
      return { candles, prices, volumes };
    };

    const data = generateData();
    setCandleData(data.candles);
    setPriceData(data.prices);
    setVolumeData(data.volumes);
  }, [symbol, marketData, timeframe]);

  const chartHeight = 400;
  const priceHeight = 300;
  const volumeHeight = 80;
  const padding = { top: 20, right: 60, bottom: 20, left: 60 };

  // Calculate price range
  const priceRange = candleData.length > 0 ? {
    min: Math.min(...candleData.map(d => d.low)) * 0.998,
    max: Math.max(...candleData.map(d => d.high)) * 1.002
  } : { min: 0, max: 100 };

  const volumeRange = volumeData.length > 0 ? {
    min: 0,
    max: Math.max(...volumeData.map(d => d.volume)) * 1.1
  } : { min: 0, max: 1000000 };

  // Scale functions
  const xScale = (i) => padding.left + (i / (candleData.length - 1)) * (800 - padding.left - padding.right);
  const yScale = (price) => padding.top + (1 - (price - priceRange.min) / (priceRange.max - priceRange.min)) * priceHeight;
  const volumeYScale = (volume) => priceHeight + padding.top + 10 + (1 - volume / volumeRange.max) * volumeHeight;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#000' }}>
      {/* Chart Toolbar */}
      <div style={{ 
        position: 'absolute', 
        top: '10px', 
        left: '10px', 
        zIndex: 10,
        display: 'flex',
        gap: '10px',
        alignItems: 'center'
      }}>
        {/* Timeframe buttons */}
        <div style={{ display: 'flex', gap: '2px' }}>
          {['1M', '5M', '15M', '1H', '4H', '1D'].map(tf => (
            <button
              key={tf}
              onClick={() => setChartTimeframe(tf)}
              style={{
                background: chartTimeframe === tf ? '#ffa500' : '#333',
                color: chartTimeframe === tf ? '#000' : '#ffa500',
                border: '1px solid #666',
                padding: '4px 8px',
                fontSize: '9px',
                cursor: 'pointer'
              }}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Indicators */}
        <div style={{ display: 'flex', gap: '2px' }}>
          {['MA', 'RSI', 'MACD', 'BB'].map(ind => (
            <button
              key={ind}
              onClick={() => {
                setChartIndicators(prev => 
                  prev.includes(ind) 
                    ? prev.filter(i => i !== ind)
                    : [...prev, ind]
                );
              }}
              style={{
                background: chartIndicators.includes(ind) ? '#00ff00' : '#333',
                color: chartIndicators.includes(ind) ? '#000' : '#00ff00',
                border: '1px solid #666',
                padding: '4px 6px',
                fontSize: '8px',
                cursor: 'pointer'
              }}
            >
              {ind}
            </button>
          ))}
        </div>
      </div>

      {/* Price and symbol display */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 10,
        textAlign: 'right'
      }}>
        <div style={{ color: '#ffa500', fontSize: '16px', fontWeight: 'bold' }}>
          {symbol}
        </div>
        <div style={{ color: '#00ff00', fontSize: '14px' }}>
          ${marketData[symbol]?.price.toFixed(2)}
        </div>
        <div style={{ 
          color: marketData[symbol]?.changePercent >= 0 ? '#00ff00' : '#ff0000',
          fontSize: '12px'
        }}>
          {marketData[symbol]?.changePercent >= 0 ? '+' : ''}{marketData[symbol]?.changePercent.toFixed(2)}%
        </div>
      </div>

      {/* Main Chart SVG */}
      <svg width="100%" height={chartHeight} style={{ position: 'absolute', top: 0 }}>
        <defs>
          <linearGradient id="volumeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#ffa500', stopOpacity: 0.8 }} />
            <stop offset="100%" style={{ stopColor: '#ffa500', stopOpacity: 0.2 }} />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
          const y = padding.top + ratio * priceHeight;
          const price = priceRange.max - ratio * (priceRange.max - priceRange.min);
          return (
            <g key={ratio}>
              <line
                x1={padding.left}
                y1={y}
                x2={800 - padding.right}
                y2={y}
                stroke="#333"
                strokeWidth="0.5"
                strokeDasharray="2,2"
              />
              <text
                x={800 - padding.right + 5}
                y={y + 3}
                fill="#666"
                fontSize="10"
                fontFamily="Courier New"
              >
                {price.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Vertical time grid */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map(ratio => {
          const x = padding.left + ratio * (800 - padding.left - padding.right);
          return (
            <line
              key={ratio}
              x1={x}
              y1={padding.top}
              x2={x}
              y2={priceHeight + padding.top}
              stroke="#333"
              strokeWidth="0.5"
              strokeDasharray="2,2"
            />
          );
        })}

        {/* Candlesticks */}
        {candleData.map((candle, i) => {
          const x = xScale(i);
          const openY = yScale(candle.open);
          const closeY = yScale(candle.close);
          const highY = yScale(candle.high);
          const lowY = yScale(candle.low);
          const isGreen = candle.close >= candle.open;
          const bodyHeight = Math.abs(closeY - openY);

          return (
            <g key={i}>
              {/* Wick */}
              <line
                x1={x}
                y1={highY}
                x2={x}
                y2={lowY}
                stroke={isGreen ? '#00ff00' : '#ff0000'}
                strokeWidth="1"
              />
              {/* Body */}
              <rect
                x={x - 2}
                y={Math.min(openY, closeY)}
                width="4"
                height={Math.max(bodyHeight, 1)}
                fill={isGreen ? '#00ff00' : '#ff0000'}
                stroke={isGreen ? '#00ff00' : '#ff0000'}
                strokeWidth="1"
              />
            </g>
          );
        })}

        {/* Moving Average (if selected) */}
        {indicators.includes('MA') && (
          <polyline
            fill="none"
            stroke="#ffa500"
            strokeWidth="2"
            opacity="0.8"
            points={candleData.map((candle, i) => {
              const ma = candleData
                .slice(Math.max(0, i - 19), i + 1)
                .reduce((sum, c) => sum + c.close, 0) / Math.min(20, i + 1);
              return `${xScale(i)},${yScale(ma)}`;
            }).join(' ')}
          />
        )}

        {/* Volume bars */}
        {volumeData.map((vol, i) => {
          const x = xScale(i);
          const height = volumeHeight * (vol.volume / volumeRange.max);
          const y = priceHeight + padding.top + 10 + volumeHeight - height;
          
          return (
            <rect
              key={i}
              x={x - 1}
              y={y}
              width="2"
              height={height}
              fill="url(#volumeGradient)"
            />
          );
        })}

        {/* Volume axis labels */}
        <text
          x="5"
          y={priceHeight + padding.top + 25}
          fill="#666"
          fontSize="9"
          fontFamily="Courier New"
        >
          VOL
        </text>
      </svg>

      {/* RSI indicator (if selected) */}
      {indicators.includes('RSI') && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '60px',
          right: '60px',
          height: '60px',
          border: '1px solid #333',
          background: 'rgba(0,0,0,0.8)'
        }}>
          <div style={{ color: '#ffa500', fontSize: '9px', padding: '4px' }}>RSI (14)</div>
          <svg width="100%" height="50">
            <line x1="0" y1="15" x2="100%" y2="15" stroke="#ff0000" strokeWidth="1" strokeDasharray="2,2" />
            <line x1="0" y1="35" x2="100%" y2="35" stroke="#00ff00" strokeWidth="1" strokeDasharray="2,2" />
            <polyline
              fill="none"
              stroke="#00ffff"
              strokeWidth="2"
              points={candleData.map((_, i) => {
                const rsi = 30 + Math.random() * 40; // Mock RSI
                return `${(i / (candleData.length - 1)) * 100}%,${50 - (rsi - 50)}`;
              }).join(' ')}
            />
          </svg>
        </div>
      )}
    </div>
  );
};

function App() {
  // Selected symbol for charts
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  
  // Active tabs state
  const [activeTab, setActiveTab] = useState('MARKET_DATA');
  const [chartTimeframe, setChartTimeframe] = useState('1M');
  const [chartIndicators, setChartIndicators] = useState(['MA', 'RSI']);
  
  // WebSocket connection state
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [realtimeUpdates, setRealtimeUpdates] = useState(0);
  
  // Comprehensive market data state
  const [marketData, setMarketData] = useState({
    // MAJOR INDICES
    'SPY': { symbol: 'SPY', price: 445.32, change: 2.15, changePercent: 0.48, volume: 15234567, iv: 0.18 },
    'QQQ': { symbol: 'QQQ', price: 378.91, change: -1.23, changePercent: -0.32, volume: 8456789, iv: 0.22 },
    'IWM': { symbol: 'IWM', price: 198.45, change: 3.21, changePercent: 1.64, volume: 4567890, iv: 0.25 },
    'DIA': { symbol: 'DIA', price: 345.67, change: 0.89, changePercent: 0.26, volume: 2345678, iv: 0.16 },
    
    // MEGA CAP TECH
    'AAPL': { symbol: 'AAPL', price: 178.25, change: -1.35, changePercent: -0.75, volume: 8456789, iv: 0.28 },
    'MSFT': { symbol: 'MSFT', price: 378.45, change: -2.87, changePercent: -0.75, volume: 6543210, iv: 0.24 },
    'GOOGL': { symbol: 'GOOGL', price: 134.82, change: 2.45, changePercent: 1.85, volume: 5432109, iv: 0.32 },
    'AMZN': { symbol: 'AMZN', price: 145.67, change: 4.21, changePercent: 2.98, volume: 7654321, iv: 0.35 },
    'META': { symbol: 'META', price: 298.34, change: -3.45, changePercent: -1.14, volume: 4321098, iv: 0.41 },
    'TSLA': { symbol: 'TSLA', price: 234.67, change: 8.42, changePercent: 3.72, volume: 12345678, iv: 0.65 },
    'NVDA': { symbol: 'NVDA', price: 456.89, change: 12.34, changePercent: 2.78, volume: 9876543, iv: 0.58 },
    'NFLX': { symbol: 'NFLX', price: 423.56, change: -5.67, changePercent: -1.32, volume: 3456789, iv: 0.45 },
    
    // HIGH GROWTH TECH
    'AMD': { symbol: 'AMD', price: 112.34, change: 3.45, changePercent: 3.17, volume: 6789012, iv: 0.52 },
    'CRM': { symbol: 'CRM', price: 234.56, change: 1.89, changePercent: 0.81, volume: 2345678, iv: 0.38 },
    'ADBE': { symbol: 'ADBE', price: 567.89, change: -4.32, changePercent: -0.75, volume: 1876543, iv: 0.29 },
    'ORCL': { symbol: 'ORCL', price: 98.76, change: 0.87, changePercent: 0.89, volume: 4567890, iv: 0.31 },
    
    // FINANCIALS
    'JPM': { symbol: 'JPM', price: 145.32, change: 2.45, changePercent: 1.71, volume: 8765432, iv: 0.22 },
    'BAC': { symbol: 'BAC', price: 32.45, change: 0.67, changePercent: 2.10, volume: 15432109, iv: 0.26 },
    'WFC': { symbol: 'WFC', price: 45.67, change: 1.23, changePercent: 2.77, volume: 12345678, iv: 0.28 },
    'GS': { symbol: 'GS', price: 387.65, change: 5.43, changePercent: 1.42, volume: 2109876, iv: 0.24 },
    
    // CRYPTO
    'BTC': { symbol: 'BTC-USD', price: 43256.78, change: 1234.56, changePercent: 2.94, volume: 15678901, iv: 0.85 },
    'ETH': { symbol: 'ETH-USD', price: 2567.89, change: -89.34, changePercent: -3.37, volume: 9876543, iv: 0.92 },
    'SOL': { symbol: 'SOL-USD', price: 98.76, change: 4.32, changePercent: 4.58, volume: 5432109, iv: 1.15 },
    'ADA': { symbol: 'ADA-USD', price: 0.487, change: 0.023, changePercent: 4.95, volume: 23456789, iv: 1.28 }
  });

  // Agent status state
  const [agents, setAgents] = useState({
    // Data Layer
    dataPreprocessing: { name: 'Data Preprocessing', status: 'ACTIVE', accuracy: 99.2, latency: 12, processed: 1543267 },
    
    // Core Trading Agents (YOUR ORIGINAL)
    momentum: { name: 'Momentum Agent', status: 'ACTIVE', accuracy: 73.4, pnl: 15432.67, trades: 156, winRate: 68.2, model: 'LSTM+RL' },
    meanReversion: { name: 'Mean Reversion', status: 'ACTIVE', accuracy: 67.8, pnl: 8765.43, trades: 89, winRate: 71.9, model: 'XGBoost+RL' },
    optionsSentiment: { name: 'Options/Sentiment', status: 'ACTIVE', accuracy: 81.3, pnl: 23456.78, trades: 67, winRate: 79.1, model: 'BERT+RL' },
    
    // Advanced Agents (MY ADDITIONS)
    statArb: { name: 'Statistical Arbitrage', status: 'ACTIVE', accuracy: 89.7, pnl: 34567.89, trades: 234, winRate: 85.3, model: 'Cointegration+ML' },
    marketMaking: { name: 'Market Making', status: 'ACTIVE', accuracy: 92.1, pnl: 12345.67, trades: 1543, winRate: 54.7, model: 'Orderbook+RL' },
    multiAsset: { name: 'Multi-Asset Momentum', status: 'ACTIVE', accuracy: 76.5, pnl: 19876.54, trades: 123, winRate: 73.2, model: 'Ensemble+RL' },
    cryptoArb: { name: 'Crypto Arbitrage', status: 'TRAINING', accuracy: 94.3, pnl: 45678.90, trades: 345, winRate: 91.6, model: 'CEX/DEX+RL' },
    volatility: { name: 'Volatility Trading', status: 'ACTIVE', accuracy: 71.8, pnl: 16789.23, trades: 78, winRate: 69.2, model: 'BlackScholes+ML' },
    metaLearning: { name: 'Meta Learning', status: 'ACTIVE', accuracy: 87.9, pnl: 28934.56, trades: 45, winRate: 84.4, model: 'MAML+Ensemble' },
    
    // Coordination Layer
    trainer: { name: 'Trainer/Coordinator', status: 'ACTIVE', modelsTraining: 7, epoch: 234, loss: 0.0023, eta: '2.3h' },
    riskManager: { name: 'Risk Manager', status: 'ACTIVE', var: 0.025, maxDrawdown: 0.087, sharpe: 1.94, exposure: 0.73 },
    portfolioAllocator: { name: 'Portfolio Allocator', status: 'ACTIVE', positions: 23, totalValue: 2.3e6, allocation: 'Optimal' },
    masterAgent: { name: 'Master Agent', status: 'ACTIVE', signals: 12, confidence: 0.847, action: 'BUY_SIGNAL', target: 'NVDA' },
    execution: { name: 'Execution Agent', status: 'ACTIVE', ordersPerSec: 23.4, latency: 3.2, fillRate: 99.7, slippage: 0.0012 },
    performance: { name: 'Performance Monitor', status: 'ACTIVE', totalPnL: 234567.89, roi: 23.4, sharpe: 2.13, calmar: 1.87 }
  });

  // NEWS & SENTIMENT state
  const [newsItems, setNewsItems] = useState([
    {
      title: "Fed Signals Dovish Stance as Inflation Cools Below 3%",
      summary: "Federal Reserve officials hint at potential rate cuts as core inflation shows signs of sustained cooling, with markets pricing in 75% chance of March cut.",
      sentiment: "BULLISH",
      impact: "HIGH",
      timestamp: "14:32",
      source: "REUTERS",
      symbols: ["SPY", "QQQ", "IWM"],
      tags: ["FED", "RATES", "INFLATION"],
      sentiment_score: 0.73,
      confidence: 0.89,
      reach: 2340,
      virality: 87
    },
    {
      title: "NVDA Earnings Beat Estimates by 15%, Revenue Guidance Raised",
      summary: "NVIDIA reports Q4 earnings of $0.74 vs $0.64 expected, with data center revenue up 427% YoY. Company raises full-year guidance citing AI demand.",
      sentiment: "BULLISH",
      impact: "HIGH",
      timestamp: "14:28",
      source: "BLOOMBERG",
      symbols: ["NVDA", "AMD", "INTC"],
      tags: ["EARNINGS", "AI", "SEMICONDUCTORS"],
      sentiment_score: 0.91,
      confidence: 0.94,
      reach: 5670,
      virality: 95
    },
    {
      title: "Oil Prices Surge 4% on Middle East Tensions",
      summary: "Crude oil jumps after reports of drone strikes on major pipeline infrastructure. Supply concerns mount as geopolitical risks escalate.",
      sentiment: "NEUTRAL",
      impact: "MEDIUM",
      timestamp: "14:15",
      source: "WSJ",
      symbols: ["XLE", "USO", "OIL"],
      tags: ["ENERGY", "GEOPOLITICS", "COMMODITIES"],
      sentiment_score: -0.12,
      confidence: 0.76,
      reach: 1230,
      virality: 34
    },
    {
      title: "Tesla Recalls 350K Vehicles Over Autopilot Safety Issue",
      summary: "NHTSA orders recall of Model 3 and Model Y vehicles manufactured 2022-2024 due to Full Self-Driving software malfunction affecting intersection safety.",
      sentiment: "BEARISH",
      impact: "MEDIUM",
      timestamp: "13:55",
      source: "CNBC",
      symbols: ["TSLA", "GM", "F"],
      tags: ["AUTOMOTIVE", "SAFETY", "RECALLS"],
      sentiment_score: -0.68,
      confidence: 0.85,
      reach: 3450,
      virality: 76
    },
    {
      title: "Banking Sector Rally Continues as Credit Losses Decline",
      summary: "Major banks report lowest credit loss provisions in 3 years as consumer spending remains robust. JP Morgan leads sector gains with 5% jump.",
      sentiment: "BULLISH",
      impact: "MEDIUM",
      timestamp: "13:42",
      source: "MARKETWATCH",
      symbols: ["JPM", "BAC", "WFC", "XLF"],
      tags: ["BANKS", "CREDIT", "EARNINGS"],
      sentiment_score: 0.45,
      confidence: 0.71,
      reach: 890,
      virality: 23
    }
  ]);

  const [sentimentData, setSentimentData] = useState({
    overall: 12.3, // Bullish bias
    sectors: {
      'TECH': 18.7,
      'FINANCE': 8.9,
      'HEALTH': 4.2,
      'ENERGY': -3.4,
      'CONSUMER': 6.1,
      'INDUSTRIAL': -1.8,
      'TELECOM': 2.3,
      'UTILITIES': -0.9
    }
  });

  const [socialBuzz, setSocialBuzz] = useState([
    { symbol: 'NVDA', platform: 'Twitter', mentions: 23.4, sentiment: 0.87 },
    { symbol: 'TSLA', platform: 'Reddit', mentions: 18.9, sentiment: -0.23 },
    { symbol: 'AAPL', platform: 'Twitter', mentions: 15.6, sentiment: 0.34 },
    { symbol: 'AMD', platform: 'Discord', mentions: 12.3, sentiment: 0.65 },
    { symbol: 'MSFT', platform: 'Twitter', mentions: 11.8, sentiment: 0.41 },
    { symbol: 'GOOGL', platform: 'Reddit', mentions: 9.7, sentiment: 0.12 },
    { symbol: 'AMZN', platform: 'Twitter', mentions: 8.4, sentiment: 0.28 },
    { symbol: 'META', platform: 'Reddit', mentions: 7.9, sentiment: -0.45 }
  ]);

  const [fearGreedIndex, setFearGreedIndex] = useState(67); // Greed territory

  const [upcomingEvents, setUpcomingEvents] = useState([
    {
      time: "15:30",
      title: "FOMC Minutes Release",
      description: "Federal Open Market Committee meeting minutes from December session",
      impact: "HIGH"
    },
    {
      time: "16:00",
      title: "EIA Crude Oil Inventories",
      description: "Weekly petroleum status report showing crude oil stock changes",
      impact: "MEDIUM"
    },
    {
      time: "Tomorrow 08:30",
      title: "Non-Farm Payrolls",
      description: "Monthly employment data release, consensus +180K jobs",
      impact: "HIGH"
    },
    {
      time: "Tomorrow 10:00",
      title: "ISM Services PMI",
      description: "Institute for Supply Management Services PMI for December",
      impact: "MEDIUM"
    },
    {
      time: "Tomorrow 14:00",
      title: "Treasury Auction",
      description: "30-year bond auction, $24B offering",
      impact: "LOW"
    }
  ]);

  // PERFORMANCE & BACKTESTING state
  const [performanceMetrics, setPerformanceMetrics] = useState([
    { label: 'TOTAL RETURN', value: '+23.4%', change: '+2.1% vs benchmark', positive: true },
    { label: 'ANNUAL RETURN', value: '+31.2%', change: 'vs 18.7% S&P 500', positive: true },
    { label: 'SHARPE RATIO', value: '2.13', change: 'Above 2.0 target', positive: true },
    { label: 'MAX DRAWDOWN', value: '-8.7%', change: 'Within 10% limit', positive: null },
    { label: 'WIN RATE', value: '68.3%', change: '+4.2% vs last month', positive: true },
    { label: 'PROFIT FACTOR', value: '1.89', change: 'Above 1.5 target', positive: true },
    { label: 'CALMAR RATIO', value: '3.58', change: 'Excellent risk-adj return', positive: true },
    { label: 'VAR (95%)', value: '-$23.4K', change: '2.3% of portfolio', positive: null }
  ]);

  const [equityCurveData, setEquityCurveData] = useState(() => {
    const data = [1000000];
    for (let i = 1; i < 60; i++) {
      const prevValue = data[i - 1];
      const dailyReturn = (Math.random() - 0.45) * 0.02; // Slight upward bias
      const newValue = prevValue * (1 + dailyReturn);
      data.push(Math.max(900000, newValue)); // Floor at 90% of starting capital
    }
    return data;
  });

  const [benchmarkData, setBenchmarkData] = useState(() => {
    const data = [1000000];
    for (let i = 1; i < 60; i++) {
      const prevValue = data[i - 1];
      const dailyReturn = (Math.random() - 0.48) * 0.015; // More conservative than portfolio
      const newValue = prevValue * (1 + dailyReturn);
      data.push(Math.max(920000, newValue));
    }
    return data;
  });

  const [strategyPerformance, setStrategyPerformance] = useState([
    {
      name: 'Momentum Trading',
      active: true,
      pnl: 45623,
      winRate: 73.4,
      sharpe: 2.13,
      trades: 156,
      maxDrawdown: 0.087,
      allocation: 0.25
    },
    {
      name: 'Mean Reversion',
      active: true,
      pnl: 32890,
      winRate: 67.8,
      sharpe: 1.89,
      trades: 89,
      maxDrawdown: 0.063,
      allocation: 0.20
    },
    {
      name: 'Statistical Arbitrage',
      active: true,
      pnl: 67234,
      winRate: 89.7,
      sharpe: 3.42,
      trades: 234,
      maxDrawdown: 0.034,
      allocation: 0.30
    },
    {
      name: 'Options Sentiment',
      active: true,
      pnl: 23456,
      winRate: 81.3,
      sharpe: 2.67,
      trades: 67,
      maxDrawdown: 0.091,
      allocation: 0.15
    },
    {
      name: 'Market Making',
      active: false,
      pnl: -8934,
      winRate: 54.7,
      sharpe: 0.89,
      trades: 1543,
      maxDrawdown: 0.156,
      allocation: 0.10
    },
    {
      name: 'Crypto Arbitrage',
      active: true,
      pnl: 89012,
      winRate: 94.3,
      sharpe: 4.12,
      trades: 345,
      maxDrawdown: 0.023,
      allocation: 0.00
    }
  ]);

  const [riskMetrics, setRiskMetrics] = useState([
    { name: 'Portfolio Beta', value: '0.73', status: 'GOOD' },
    { name: 'Correlation to SPY', value: '0.61', status: 'GOOD' },
    { name: 'Volatility (Ann.)', value: '18.4%', status: 'GOOD' },
    { name: 'Downside Deviation', value: '12.7%', status: 'GOOD' },
    { name: 'Sortino Ratio', value: '2.89', status: 'GOOD' },
    { name: 'Information Ratio', value: '1.23', status: 'GOOD' },
    { name: 'Maximum DD Duration', value: '23 days', status: 'WARNING' },
    { name: 'Current Exposure', value: '87.3%', status: 'WARNING' }
  ]);

  const [recentTrades, setRecentTrades] = useState([
    {
      symbol: 'NVDA',
      side: 'SELL',
      quantity: 200,
      price: 891.23,
      time: '14:32:15',
      strategy: 'Momentum',
      pnl: 2340
    },
    {
      symbol: 'TSLA',
      side: 'BUY',
      quantity: 500,
      price: 178.45,
      time: '14:28:07',
      strategy: 'Mean Rev',
      pnl: -890
    },
    {
      symbol: 'SPY',
      side: 'SELL',
      quantity: 1000,
      price: 456.78,
      time: '14:15:23',
      strategy: 'Stat Arb',
      pnl: 1560
    },
    {
      symbol: 'QQQ',
      side: 'BUY',
      quantity: 800,
      price: 387.92,
      time: '13:58:44',
      strategy: 'Options',
      pnl: 3240
    },
    {
      symbol: 'IWM',
      side: 'SELL',
      quantity: 1200,
      price: 198.34,
      time: '13:45:12',
      strategy: 'Momentum',
      pnl: -450
    },
    {
      symbol: 'VIX',
      side: 'BUY',
      quantity: 300,
      price: 23.45,
      time: '13:22:56',
      strategy: 'Options',
      pnl: 780
    }
  ]);

  // ORDERS & EXECUTION state
  const [orderTab, setOrderTab] = useState('ACTIVE');
  const [orderEntry, setOrderEntry] = useState({
    symbol: 'NVDA',
    side: 'BUY',
    quantity: 100,
    type: 'LIMIT',
    price: 890.00,
    tif: 'DAY'
  });

  const [activeOrders, setActiveOrders] = useState([
    { symbol: 'NVDA', side: 'BUY', quantity: 500, price: '889.50', filled: 0, remaining: 500, type: 'LIMIT', time: '09:31:24' },
    { symbol: 'TSLA', side: 'SELL', quantity: 1000, price: 'MKT', filled: 0, remaining: 1000, type: 'MARKET', time: '10:15:33' },
    { symbol: 'AAPL', side: 'BUY', quantity: 200, price: '195.25', filled: 50, remaining: 150, type: 'LIMIT', time: '11:22:17' }
  ]);

  const [pendingOrders, setPendingOrders] = useState([
    { symbol: 'SPY', side: 'SELL', quantity: 2000, price: '458.00', filled: 0, remaining: 2000, type: 'LIMIT', time: '08:30:00' },
    { symbol: 'QQQ', side: 'BUY', quantity: 800, price: '385.75', filled: 0, remaining: 800, type: 'LIMIT', time: '09:00:15' }
  ]);

  const [filledOrders, setFilledOrders] = useState([
    { symbol: 'MSFT', side: 'BUY', quantity: 300, price: '415.80', filled: 300, remaining: 0, type: 'MARKET', time: '14:32:55' },
    { symbol: 'GOOGL', side: 'SELL', quantity: 100, price: '178.90', filled: 100, remaining: 0, type: 'LIMIT', time: '13:45:12' },
    { symbol: 'AMZN', side: 'BUY', quantity: 150, price: '168.35', filled: 150, remaining: 0, type: 'MARKET', time: '12:18:44' },
    { symbol: 'META', side: 'SELL', quantity: 400, price: '528.75', filled: 400, remaining: 0, type: 'LIMIT', time: '11:55:23' }
  ]);

  const [cancelledOrders, setCancelledOrders] = useState([
    { symbol: 'NFLX', side: 'BUY', quantity: 100, price: '654.20', filled: 0, remaining: 100, type: 'LIMIT', time: '10:30:15' },
    { symbol: 'AMD', side: 'SELL', quantity: 500, price: '152.40', filled: 0, remaining: 500, type: 'LIMIT', time: '09:45:33' }
  ]);

  const [executionStats, setExecutionStats] = useState([
    { label: 'FILL RATE', value: '99.7%', positive: true },
    { label: 'AVG LATENCY', value: '3.2ms', positive: true },
    { label: 'SLIPPAGE', value: '0.12bp', positive: true },
    { label: 'TOTAL FEES', value: '$2,342', positive: null }
  ]);

  const [executionFeed, setExecutionFeed] = useState([
    { symbol: 'NVDA', side: 'SELL', quantity: 200, price: 891.23, time: '14:32:15', status: 'FILLED' },
    { symbol: 'TSLA', side: 'BUY', quantity: 500, price: 178.45, time: '14:28:07', status: 'PARTIAL' },
    { symbol: 'SPY', side: 'SELL', quantity: 1000, price: 456.78, time: '14:15:23', status: 'FILLED' },
    { symbol: 'QQQ', side: 'BUY', quantity: 800, price: 387.92, time: '13:58:44', status: 'FILLED' },
    { symbol: 'AAPL', side: 'SELL', quantity: 300, price: 195.67, time: '13:45:12', status: 'CANCELLED' },
    { symbol: 'MSFT', side: 'BUY', quantity: 400, price: 415.89, time: '13:22:56', status: 'FILLED' },
    { symbol: 'GOOGL', side: 'SELL', quantity: 100, price: 178.34, time: '12:58:33', status: 'FILLED' }
  ]);

  // WebSocket integration
  useEffect(() => {
    const initializeWebSocket = async () => {
      try {
        // Set up connection handler
        webSocketService.onConnection((connected) => {
          setIsConnected(connected);
          if (connected) {
            setConnectionError(null);
            console.log('✅ Connected to trading backend');
          } else {
            console.log('❌ Disconnected from trading backend');
          }
        });

        // Set up error handler
        webSocketService.onError((error) => {
          setConnectionError(error.message);
          console.error('WebSocket error:', error);
        });

        // Set up market data handler
        webSocketService.onMessage<MarketDataMessage>('market_data', (message) => {
          setMarketData(prev => ({
            ...prev,
            [message.symbol]: {
              symbol: message.symbol,
              price: message.data.price,
              change: message.data.change,
              changePercent: message.data.changePercent,
              volume: message.data.volume,
              high: message.data.high,
              low: message.data.low,
              open: message.data.open,
              vwap: message.data.vwap || message.data.price,
              iv: message.data.iv || 0.3
            }
          }));
          setRealtimeUpdates(prev => prev + 1);
        });

        // Set up agent signals handler
        webSocketService.onMessage<AgentSignalMessage>('agent_signal', (message) => {
          console.log(`🤖 Agent Signal from ${message.agent}: ${message.data.signal} ${message.symbol} @ ${message.data.confidence}% confidence`);
          
          // Update agent status based on signals
          setAgents(prev => ({
            ...prev,
            [message.agent.toLowerCase().replace(' ', '')]: {
              ...prev[message.agent.toLowerCase().replace(' ', '')],
              lastSignal: message.data.signal,
              lastSignalTime: new Date().toLocaleTimeString(),
              confidence: message.data.confidence
            }
          }));
        });

        // Set up portfolio updates handler
        webSocketService.onMessage<PortfolioUpdateMessage>('portfolio_update', (message) => {
          console.log('💰 Portfolio update:', message.data);
          // Update portfolio-related agents
          setAgents(prev => ({
            ...prev,
            performance: {
              ...prev.performance,
              totalPnL: message.data.pnl,
              exposure: message.data.exposure
            }
          }));
        });

        // Set up risk alerts handler
        webSocketService.onMessage<RiskAlertMessage>('risk_alert', (message) => {
          console.log(`⚠️ Risk Alert [${message.data.severity}]: ${message.data.message}`);
          // You could show these as notifications in the UI
        });

        // Set up system status handler
        webSocketService.onMessage<SystemStatusMessage>('system_status', (message) => {
          console.log('📊 System status:', message.data.status);
          // Update system status indicators in header
        });

        // Set up news updates handler
        webSocketService.onMessage('news_update', (message) => {
          console.log('📰 News update:', message.data.title);
          setNewsItems(prev => {
            const newItems = [message.data, ...prev.slice(0, 19)]; // Keep latest 20 items
            return newItems;
          });
        });

        // Set up sentiment updates handler
        webSocketService.onMessage('sentiment_update', (message) => {
          console.log('😊 Sentiment update:', message.data);
          if (message.data.overall !== undefined) {
            setSentimentData(prev => ({
              ...prev,
              overall: message.data.overall,
              sectors: { ...prev.sectors, ...message.data.sectors }
            }));
          }
          if (message.data.fear_greed !== undefined) {
            setFearGreedIndex(message.data.fear_greed);
          }
        });

        // Set up social buzz handler
        webSocketService.onMessage('social_buzz', (message) => {
          console.log('📱 Social buzz update:', message.data);
          setSocialBuzz(message.data);
        });

        // Set up performance updates handler
        webSocketService.onMessage('performance_update', (message) => {
          console.log('📈 Performance update:', message.data);
          if (message.data.metrics) {
            setPerformanceMetrics(message.data.metrics);
          }
          if (message.data.equity_curve) {
            setEquityCurveData(message.data.equity_curve);
          }
          if (message.data.strategy_performance) {
            setStrategyPerformance(message.data.strategy_performance);
          }
        });

        // Set up trade execution handler
        webSocketService.onMessage('trade_execution', (message) => {
          console.log('💰 Trade executed:', message.data);
          setRecentTrades(prev => {
            const newTrades = [message.data, ...prev.slice(0, 9)]; // Keep latest 10 trades
            return newTrades;
          });
          // Also add to execution feed
          setExecutionFeed(prev => {
            const newExecution = {
              symbol: message.data.symbol,
              side: message.data.side,
              quantity: message.data.quantity,
              price: message.data.price,
              time: message.data.time,
              status: 'FILLED'
            };
            return [newExecution, ...prev.slice(0, 9)];
          });
        });

        // Set up order updates handler
        webSocketService.onMessage('order_update', (message) => {
          console.log('📋 Order update:', message.data);
          const order = message.data;
          
          // Update appropriate order list based on status
          if (order.status === 'NEW' || order.status === 'PARTIALLY_FILLED') {
            setActiveOrders(prev => {
              const existingIndex = prev.findIndex(o => o.id === order.id);
              if (existingIndex >= 0) {
                const updated = [...prev];
                updated[existingIndex] = order;
                return updated;
              }
              return [order, ...prev];
            });
          } else if (order.status === 'FILLED') {
            // Move to filled orders
            setActiveOrders(prev => prev.filter(o => o.id !== order.id));
            setFilledOrders(prev => [order, ...prev.slice(0, 19)]);
          } else if (order.status === 'CANCELLED') {
            // Move to cancelled orders
            setActiveOrders(prev => prev.filter(o => o.id !== order.id));
            setCancelledOrders(prev => [order, ...prev.slice(0, 19)]);
          }
        });

        // Connect to WebSocket
        await webSocketService.connect();

        // Subscribe to all symbols we're tracking
        const symbols = Object.keys(marketData);
        webSocketService.subscribeToSymbols(symbols);

      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        setConnectionError('Failed to connect to trading backend');
        
        // Fallback to simulated data if WebSocket fails
        const interval = setInterval(() => {
          setMarketData(prev => {
            const updated = { ...prev };
            Object.keys(updated).forEach(symbol => {
              const data = updated[symbol];
              const volatility = data.iv || 0.3;
              const change = (Math.random() - 0.5) * volatility * data.price * 0.01;
              updated[symbol] = {
                ...data,
                price: Math.max(0.01, data.price + change),
                change: data.change + change,
                changePercent: ((data.change + change) / (data.price - data.change)) * 100,
                volume: data.volume + Math.floor(Math.random() * 10000)
              };
            });
            return updated;
          });
        }, 2000); // Slower updates for fallback mode

        return () => clearInterval(interval);
      }
    };

    initializeWebSocket();

    // Cleanup on unmount
    return () => {
      webSocketService.disconnect();
      webSocketService.removeAllHandlers();
    };
  }, []); // Empty dependency array for one-time setup

  return (
    <>
      <GlobalStyle />
      <div style={{
        background: '#000000',
        color: '#00ff00',
        fontFamily: 'Courier New, Consolas, monospace',
        height: '100vh',
        width: '100vw',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Header */}
        <div style={{
          background: '#0a0a0a',
          borderBottom: '1px solid #333',
          padding: '8px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: '40px',
          fontSize: '11px'
        }}>
          <div style={{ color: '#ffa500', fontWeight: 'bold', fontSize: '14px', letterSpacing: '2px' }}>
            HIVE TRADE QUANTUM TERMINAL v2.0
          </div>
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
            <div style={{ color: isConnected ? '#00ff00' : '#ff0000' }}>
              ● {isConnected ? 'CONNECTED' : 'OFFLINE'} {connectionError && `(${connectionError})`}
            </div>
            <div style={{ color: '#00ff00' }}>● MARKET OPEN</div>
            <div style={{ color: '#ffa500', fontSize: '10px' }}>
              Updates: {realtimeUpdates}
            </div>
            <div style={{ color: '#ffa500', fontSize: '10px' }}>
              {new Date().toLocaleString()}
            </div>
          </div>
        </div>

        {/* Main Content with Tabs */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column'
        }}>
          
          {/* Tab Navigation */}
          <div style={{
            background: '#0a0a0a',
            borderBottom: '1px solid #333',
            padding: '0 10px',
            display: 'flex',
            gap: '2px'
          }}>
            {[
              { id: 'CHARTS', label: 'CHARTS', icon: '📈' },
              { id: 'MARKET_DATA', label: 'MARKET DATA', icon: '💹' },
              { id: 'OPTIONS', label: 'OPTIONS', icon: '🎯' },
              { id: 'AGENTS', label: 'AI AGENTS', icon: '🤖' },
              { id: 'NEWS', label: 'NEWS & SENTIMENT', icon: '📰' },
              { id: 'PERFORMANCE', label: 'PERFORMANCE', icon: '📊' },
              { id: 'ORDERS', label: 'ORDERS', icon: '📋' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  background: activeTab === tab.id ? '#ffa500' : '#333',
                  color: activeTab === tab.id ? '#000' : '#ffa500',
                  border: 'none',
                  padding: '8px 16px',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontFamily: 'Courier New, monospace',
                  borderRadius: '4px 4px 0 0'
                }}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div style={{
            flex: 1,
            background: '#0a0a0a',
            overflow: 'hidden',
            padding: '10px'
          }}>
            {activeTab === 'CHARTS' && (
              <ProfessionalChart 
                symbol={selectedSymbol} 
                marketData={marketData}
                timeframe={chartTimeframe}
                indicators={chartIndicators}
              />
            )}

            {activeTab === 'MARKET_DATA' && (
              <div style={{
                height: '100%',
                background: '#0a0a0a',
                border: '1px solid #222',
                padding: '10px',
                overflow: 'hidden'
              }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '10px',
              borderBottom: '1px solid #333',
              paddingBottom: '5px',
              color: '#ffa500', 
              fontSize: '11px', 
              fontWeight: 'bold'
            }}>
              <span>MARKET DATA - {Object.keys(marketData).length} INSTRUMENTS</span>
              <select 
                style={{
                  background: '#000',
                  color: '#00ff00',
                  border: '1px solid #333',
                  padding: '2px 5px',
                  fontSize: '9px'
                }} 
                onChange={(e) => setSelectedSymbol(e.target.value)}
                value={selectedSymbol}
              >
                <option value="SPY">SPY</option>
                <option value="AAPL">AAPL</option>
                <option value="TSLA">TSLA</option>
                <option value="NVDA">NVDA</option>
                <option value="BTC">BTC-USD</option>
                <option value="ETH">ETH-USD</option>
              </select>
            </div>
            
            {/* Header Row */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '60px 70px 60px 50px 60px 40px', 
              gap: '8px', 
              fontSize: '9px', 
              marginBottom: '8px',
              color: '#ffa500',
              fontWeight: 'bold',
              borderBottom: '1px solid #333',
              paddingBottom: '4px'
            }}>
              <div>SYMBOL</div>
              <div>PRICE</div>
              <div>CHANGE</div>
              <div>CHANGE%</div>
              <div>VOLUME</div>
              <div>IV</div>
            </div>
            
            {/* Scrollable Market Data */}
            <div style={{ 
              height: 'calc(100% - 45px)', 
              overflowY: 'auto',
              fontSize: '9px'
            }}>
              {Object.values(marketData).map((stock) => (
                <div key={stock.symbol} style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '60px 70px 60px 50px 60px 40px', 
                  gap: '8px', 
                  marginBottom: '3px',
                  padding: '2px 0',
                  borderBottom: stock.symbol.includes('BTC') || stock.symbol.includes('ETH') ? '1px solid #333' : 'none'
                }}>
                  <div style={{ 
                    color: stock.symbol.includes('BTC') || stock.symbol.includes('ETH') ? '#00ffff' : '#ffa500',
                    fontWeight: 'bold'
                  }}>
                    {stock.symbol}
                  </div>
                  <div style={{ 
                    color: stock.changePercent >= 0 ? '#00ff00' : '#ff0000',
                    textAlign: 'right'
                  }}>
                    {stock.price.toLocaleString('en-US', { 
                      minimumFractionDigits: stock.price < 1 ? 4 : 2, 
                      maximumFractionDigits: stock.price < 1 ? 4 : 2 
                    })}
                  </div>
                  <div style={{ 
                    color: stock.changePercent >= 0 ? '#00ff00' : '#ff0000',
                    textAlign: 'right'
                  }}>
                    {stock.changePercent >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                  </div>
                  <div style={{ 
                    color: stock.changePercent >= 0 ? '#00ff00' : '#ff0000',
                    textAlign: 'right'
                  }}>
                    {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                  </div>
                  <div style={{ color: '#ccc', textAlign: 'right' }}>
                    {stock.volume >= 1000000 ? 
                      `${(stock.volume / 1000000).toFixed(1)}M` : 
                      `${(stock.volume / 1000).toFixed(0)}K`
                    }
                  </div>
                  <div style={{ 
                    color: stock.iv > 0.5 ? '#ff0000' : stock.iv > 0.3 ? '#ffa500' : '#00ff00',
                    textAlign: 'right'
                  }}>
                    {(stock.iv * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
              </div>
            )}

            {activeTab === 'OPTIONS' && (
              <div style={{
                height: '100%',
                background: '#0a0a0a',
                border: '1px solid #222',
                padding: '10px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  marginBottom: '10px',
                  borderBottom: '1px solid #333',
                  paddingBottom: '5px',
                  color: '#ffa500', 
                  fontSize: '11px', 
                  fontWeight: 'bold'
                }}>
                  <span>OPTIONS & POSITIONS</span>
                  <select 
                    style={{
                      background: '#000',
                      color: '#00ff00',
                      border: '1px solid #333',
                      padding: '2px 5px',
                      fontSize: '9px'
                    }} 
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    value={selectedSymbol}
                  >
                    <option value="SPY">SPY</option>
                    <option value="AAPL">AAPL</option>
                    <option value="TSLA">TSLA</option>
                    <option value="NVDA">NVDA</option>
                    <option value="MSFT">MSFT</option>
                    <option value="GOOGL">GOOGL</option>
                  </select>
                </div>

                <div style={{ height: 'calc(100% - 30px)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {/* Options Chain */}
                  <div>
                    <div style={{ color: '#00ffff', fontSize: '9px', fontWeight: 'bold', marginBottom: '6px' }}>
                      OPTIONS CHAIN - {selectedSymbol} (${marketData[selectedSymbol]?.price.toFixed(2)})
                    </div>
                    
                    <div style={{ 
                      height: 'calc(100% - 25px)', 
                      overflowY: 'auto', 
                      fontSize: '8px',
                      border: '1px solid #333'
                    }}>
                      {/* Options Chain Header */}
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '60px 40px 40px 40px 10px 60px 40px 40px 40px',
                        gap: '4px',
                        padding: '4px',
                        background: '#1a1a1a',
                        borderBottom: '1px solid #333',
                        fontSize: '7px',
                        fontWeight: 'bold',
                        color: '#ffa500'
                      }}>
                        <div>CALLS</div>
                        <div>BID</div>
                        <div>ASK</div>
                        <div>VOL</div>
                        <div>|</div>
                        <div>PUTS</div>
                        <div>BID</div>
                        <div>ASK</div>
                        <div>VOL</div>
                      </div>

                      {/* Generate options chain for current symbol */}
                      {(() => {
                        const currentPrice = marketData[selectedSymbol]?.price || 450;
                        const strikes = [];
                        
                        // Generate strikes around current price
                        const baseStrike = Math.floor(currentPrice / 5) * 5;
                        for (let i = -10; i <= 10; i++) {
                          strikes.push(baseStrike + i * 5);
                        }
                        
                        return strikes.map(strike => {
                          const isITM_Call = currentPrice > strike;
                          const isITM_Put = currentPrice < strike;
                          const isATM = Math.abs(currentPrice - strike) < 2.5;
                          
                          // Mock options data
                          const callBid = Math.max(0.01, currentPrice - strike + Math.random() * 2);
                          const callAsk = callBid + 0.05 + Math.random() * 0.10;
                          const putBid = Math.max(0.01, strike - currentPrice + Math.random() * 2);
                          const putAsk = putBid + 0.05 + Math.random() * 0.10;
                          
                          return (
                            <div key={strike} style={{
                              display: 'grid',
                              gridTemplateColumns: '60px 40px 40px 40px 10px 60px 40px 40px 40px',
                              gap: '4px',
                              padding: '2px 4px',
                              background: isATM ? '#002200' : 'transparent',
                              borderBottom: '1px solid #222'
                            }}>
                              {/* Calls */}
                              <div style={{ 
                                color: isITM_Call ? '#00ff00' : '#666',
                                fontSize: '7px'
                              }}>
                                {selectedSymbol}{strike}C
                              </div>
                              <div style={{ color: '#00ff00', textAlign: 'right' }}>
                                {callBid.toFixed(2)}
                              </div>
                              <div style={{ color: '#ff0000', textAlign: 'right' }}>
                                {callAsk.toFixed(2)}
                              </div>
                              <div style={{ color: '#ccc', textAlign: 'right' }}>
                                {Math.floor(Math.random() * 500)}
                              </div>
                              
                              {/* Strike separator */}
                              <div style={{ 
                                textAlign: 'center', 
                                color: isATM ? '#ffa500' : '#666',
                                fontWeight: isATM ? 'bold' : 'normal'
                              }}>
                                {strike}
                              </div>
                              
                              {/* Puts */}
                              <div style={{ 
                                color: isITM_Put ? '#00ff00' : '#666',
                                fontSize: '7px'
                              }}>
                                {selectedSymbol}{strike}P
                              </div>
                              <div style={{ color: '#00ff00', textAlign: 'right' }}>
                                {putBid.toFixed(2)}
                              </div>
                              <div style={{ color: '#ff0000', textAlign: 'right' }}>
                                {putAsk.toFixed(2)}
                              </div>
                              <div style={{ color: '#ccc', textAlign: 'right' }}>
                                {Math.floor(Math.random() * 500)}
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>
                  </div>

                  {/* Positions & Greeks */}
                  <div>
                    <div style={{ color: '#ff00ff', fontSize: '9px', fontWeight: 'bold', marginBottom: '6px' }}>
                      POSITIONS & GREEKS
                    </div>
                    
                    <div style={{ 
                      height: 'calc(100% - 25px)', 
                      overflowY: 'auto', 
                      fontSize: '8px'
                    }}>
                      {/* Equity Positions */}
                      <div style={{ marginBottom: '8px', border: '1px solid #333', padding: '4px' }}>
                        <div style={{ color: '#00ff00', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          EQUITY POSITIONS
                        </div>
                        
                        {[
                          { symbol: 'AAPL', qty: 500, avgCost: 175.23, currentPrice: marketData['AAPL']?.price || 178.25, pnl: 1510 },
                          { symbol: 'TSLA', qty: -200, avgCost: 240.50, currentPrice: marketData['TSLA']?.price || 234.67, pnl: 1166 },
                          { symbol: 'NVDA', qty: 150, avgCost: 450.20, currentPrice: marketData['NVDA']?.price || 456.89, pnl: 1003.50 }
                        ].map((pos, i) => (
                          <div key={i} style={{ 
                            display: 'grid', 
                            gridTemplateColumns: '40px 30px 50px 50px 50px', 
                            gap: '4px',
                            marginBottom: '2px',
                            padding: '2px',
                            backgroundColor: i % 2 === 0 ? '#111' : 'transparent',
                            fontSize: '7px'
                          }}>
                            <div style={{ color: '#ffa500', fontWeight: 'bold' }}>{pos.symbol}</div>
                            <div style={{ color: pos.qty >= 0 ? '#00ff00' : '#ff0000' }}>{pos.qty}</div>
                            <div style={{ color: '#ccc' }}>${pos.avgCost.toFixed(2)}</div>
                            <div style={{ color: '#ccc' }}>${pos.currentPrice.toFixed(2)}</div>
                            <div style={{ color: pos.pnl >= 0 ? '#00ff00' : '#ff0000' }}>
                              ${pos.pnl.toLocaleString()}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Options Positions */}
                      <div style={{ marginBottom: '8px', border: '1px solid #333', padding: '4px' }}>
                        <div style={{ color: '#ff00ff', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          OPTIONS POSITIONS
                        </div>
                        
                        {[
                          { 
                            symbol: 'AAPL', 
                            strike: 180, 
                            expiry: '03/15', 
                            type: 'CALL', 
                            qty: 10, 
                            premium: 3.25, 
                            current: 4.10,
                            delta: 0.67,
                            gamma: 0.045,
                            theta: -0.12,
                            vega: 0.23
                          },
                          { 
                            symbol: 'TSLA', 
                            strike: 240, 
                            expiry: '03/22', 
                            type: 'PUT', 
                            qty: -5, 
                            premium: 8.50, 
                            current: 6.75,
                            delta: -0.45,
                            gamma: 0.032,
                            theta: -0.18,
                            vega: 0.31
                          },
                          {
                            symbol: 'NVDA',
                            strike: 460,
                            expiry: '04/19',
                            type: 'CALL',
                            qty: 3,
                            premium: 15.50,
                            current: 18.75,
                            delta: 0.72,
                            gamma: 0.038,
                            theta: -0.25,
                            vega: 0.41
                          }
                        ].map((opt, i) => (
                          <div key={i} style={{ marginBottom: '6px', border: '1px solid #444', padding: '4px', background: '#0f0f0f' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '3px' }}>
                              <div style={{ color: '#ff00ff', fontSize: '8px', fontWeight: 'bold' }}>
                                {opt.symbol} ${opt.strike} {opt.type} {opt.expiry}
                              </div>
                              <div style={{ color: opt.qty >= 0 ? '#00ff00' : '#ff0000', fontSize: '8px', textAlign: 'right' }}>
                                {opt.qty} contracts
                              </div>
                            </div>
                            
                            {/* Greeks Display */}
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '3px', marginBottom: '3px', fontSize: '6px' }}>
                              <div style={{ 
                                color: Math.abs(opt.delta) > 0.5 ? '#00ff00' : '#ffa500',
                                textAlign: 'center',
                                background: '#1a1a1a',
                                padding: '1px'
                              }}>
                                Δ: {opt.delta.toFixed(3)}
                              </div>
                              <div style={{ 
                                color: opt.gamma > 0.04 ? '#ff0000' : '#ffa500',
                                textAlign: 'center',
                                background: '#1a1a1a',
                                padding: '1px'
                              }}>
                                Γ: {opt.gamma.toFixed(3)}
                              </div>
                              <div style={{ 
                                color: opt.theta < -0.15 ? '#ff0000' : '#ffa500',
                                textAlign: 'center',
                                background: '#1a1a1a',
                                padding: '1px'
                              }}>
                                Θ: {opt.theta.toFixed(3)}
                              </div>
                              <div style={{ 
                                color: opt.vega > 0.3 ? '#00ffff' : '#ffa500',
                                textAlign: 'center',
                                background: '#1a1a1a',
                                padding: '1px'
                              }}>
                                ν: {opt.vega.toFixed(3)}
                              </div>
                            </div>
                            
                            {/* P&L Display */}
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '7px' }}>
                              <div>
                                <span style={{ color: '#ccc' }}>Entry: </span>
                                <span style={{ color: '#ffa500' }}>${opt.premium.toFixed(2)}</span>
                              </div>
                              <div>
                                <span style={{ color: '#ccc' }}>Current: </span>
                                <span style={{ color: '#ffa500' }}>${opt.current.toFixed(2)}</span>
                              </div>
                            </div>
                            
                            <div style={{ marginTop: '2px', textAlign: 'center' }}>
                              <span style={{ color: '#ccc', fontSize: '7px' }}>P&L: </span>
                              <span style={{ 
                                color: (opt.current - opt.premium) * opt.qty >= 0 ? '#00ff00' : '#ff0000',
                                fontSize: '8px',
                                fontWeight: 'bold'
                              }}>
                                ${((opt.current - opt.premium) * opt.qty * 100).toLocaleString()}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Portfolio Greeks Summary */}
                      <div style={{ border: '1px solid #333', padding: '4px' }}>
                        <div style={{ color: '#00ffff', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          PORTFOLIO GREEKS
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '8px' }}>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>TOTAL DELTA</div>
                            <div style={{ color: '#00ff00', fontWeight: 'bold' }}>+1.24</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>TOTAL GAMMA</div>
                            <div style={{ color: '#ffa500', fontWeight: 'bold' }}>0.115</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>TOTAL THETA</div>
                            <div style={{ color: '#ff0000', fontWeight: 'bold' }}>-$127</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>TOTAL VEGA</div>
                            <div style={{ color: '#00ffff', fontWeight: 'bold' }}>+$834</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'AGENTS' && (
              <div style={{
                height: '100%',
                background: '#0a0a0a',
                border: '1px solid #222',
                padding: '10px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  marginBottom: '10px',
                  borderBottom: '1px solid #333',
                  paddingBottom: '5px',
                  color: '#ffa500', 
                  fontSize: '11px', 
                  fontWeight: 'bold'
                }}>
                  <span>AI AGENT PIPELINE - {Object.keys(agents).length} AGENTS ACTIVE</span>
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <div style={{ color: isConnected ? '#00ff00' : '#ff0000', fontSize: '9px' }}>
                      {isConnected ? '● LIVE' : '● OFFLINE'}
                    </div>
                    <div style={{ color: '#ffa500', fontSize: '9px' }}>
                      Updates: {realtimeUpdates}
                    </div>
                  </div>
                </div>

                <div style={{ height: 'calc(100% - 30px)', display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '10px' }}>
                  {/* Agent Status Matrix */}
                  <div>
                    <div style={{ color: '#00ffff', fontSize: '9px', fontWeight: 'bold', marginBottom: '6px' }}>
                      AGENT STATUS MATRIX
                    </div>
                    
                    <div style={{ 
                      height: 'calc(100% - 25px)', 
                      overflowY: 'auto', 
                      fontSize: '8px'
                    }}>
                      {/* Data Layer */}
                      <div style={{ marginBottom: '8px', borderBottom: '1px solid #444', paddingBottom: '4px' }}>
                        <div style={{ color: '#00ffff', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px', display: 'flex', alignItems: 'center' }}>
                          <span style={{ marginRight: '10px' }}>🔄 DATA LAYER</span>
                          <div style={{ 
                            width: '8px', 
                            height: '8px', 
                            borderRadius: '50%', 
                            background: agents.dataPreprocessing.status === 'ACTIVE' ? '#00ff00' : '#ff0000' 
                          }}></div>
                        </div>
                        
                        <div style={{ 
                          display: 'grid', 
                          gridTemplateColumns: '120px 50px 50px 60px 60px 80px', 
                          gap: '6px',
                          padding: '4px',
                          backgroundColor: '#1a1a1a',
                          border: '1px solid #333',
                          marginBottom: '2px'
                        }}>
                          <div style={{ color: '#00ffff', fontWeight: 'bold' }}>{agents.dataPreprocessing.name}</div>
                          <div style={{ color: '#00ff00', textAlign: 'right' }}>{agents.dataPreprocessing.accuracy}%</div>
                          <div style={{ color: '#ffa500', textAlign: 'right' }}>{agents.dataPreprocessing.latency}ms</div>
                          <div style={{ color: '#ccc', textAlign: 'right' }}>{agents.dataPreprocessing.processed?.toLocaleString() || 'N/A'}</div>
                          <div style={{ 
                            color: agents.dataPreprocessing.status === 'ACTIVE' ? '#00ff00' : '#ff0000',
                            textAlign: 'center',
                            fontWeight: 'bold'
                          }}>
                            {agents.dataPreprocessing.status}
                          </div>
                          <div style={{ textAlign: 'center' }}>
                            <button style={{
                              background: agents.dataPreprocessing.status === 'ACTIVE' ? '#ff0000' : '#00ff00',
                              color: '#000',
                              border: 'none',
                              padding: '2px 6px',
                              fontSize: '7px',
                              cursor: 'pointer'
                            }}>
                              {agents.dataPreprocessing.status === 'ACTIVE' ? 'STOP' : 'START'}
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* Core Trading Agents */}
                      <div style={{ marginBottom: '8px', borderBottom: '1px solid #444', paddingBottom: '4px' }}>
                        <div style={{ color: '#ffa500', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          🎯 CORE TRADING AGENTS
                        </div>
                        
                        {[agents.momentum, agents.meanReversion, agents.optionsSentiment].map((agent, idx) => (
                          <div key={idx} style={{ 
                            display: 'grid', 
                            gridTemplateColumns: '120px 50px 50px 50px 50px 60px 80px', 
                            gap: '6px',
                            padding: '4px',
                            marginBottom: '2px',
                            backgroundColor: idx % 2 === 0 ? '#111' : '#0f0f0f',
                            border: '1px solid #333'
                          }}>
                            <div style={{ color: '#ffa500', fontWeight: 'bold', fontSize: '8px' }}>{agent.name}</div>
                            <div style={{ color: '#00ff00', textAlign: 'right' }}>${agent.pnl?.toLocaleString() || '0'}</div>
                            <div style={{ color: '#00ff00', textAlign: 'right' }}>{agent.winRate || 0}%</div>
                            <div style={{ color: '#ccc', textAlign: 'right' }}>{agent.accuracy || 0}%</div>
                            <div style={{ color: '#ccc', textAlign: 'right' }}>{agent.trades || 0}</div>
                            <div style={{ 
                              color: agent.status === 'ACTIVE' ? '#00ff00' : agent.status === 'TRAINING' ? '#ffa500' : '#ff0000',
                              textAlign: 'center',
                              fontWeight: 'bold',
                              fontSize: '7px'
                            }}>
                              {agent.status}
                            </div>
                            <div style={{ textAlign: 'center' }}>
                              <button style={{
                                background: agent.status === 'ACTIVE' ? '#ff0000' : '#00ff00',
                                color: '#000',
                                border: 'none',
                                padding: '2px 6px',
                                fontSize: '7px',
                                cursor: 'pointer',
                                marginRight: '2px'
                              }} onClick={() => {
                                // Toggle agent status
                                setAgents(prev => ({
                                  ...prev,
                                  [Object.keys(prev).find(key => prev[key] === agent)]: {
                                    ...agent,
                                    status: agent.status === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE'
                                  }
                                }));
                              }}>
                                {agent.status === 'ACTIVE' ? 'STOP' : 'START'}
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Advanced Agents */}
                      <div style={{ marginBottom: '8px', borderBottom: '1px solid #444', paddingBottom: '4px' }}>
                        <div style={{ color: '#00ff00', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          🚀 ADVANCED AGENTS
                        </div>
                        
                        {[agents.statArb, agents.marketMaking, agents.multiAsset, agents.cryptoArb, agents.volatility, agents.metaLearning].map((agent, idx) => (
                          <div key={idx} style={{ 
                            display: 'grid', 
                            gridTemplateColumns: '120px 50px 50px 50px 50px 60px 80px', 
                            gap: '6px',
                            padding: '4px',
                            marginBottom: '2px',
                            backgroundColor: idx % 2 === 0 ? '#111' : '#0f0f0f',
                            border: '1px solid #333'
                          }}>
                            <div style={{ color: '#00ff00', fontWeight: 'bold', fontSize: '8px' }}>{agent.name}</div>
                            <div style={{ color: '#00ff00', textAlign: 'right' }}>${agent.pnl?.toLocaleString() || '0'}</div>
                            <div style={{ color: '#00ff00', textAlign: 'right' }}>{agent.winRate || 0}%</div>
                            <div style={{ color: '#ccc', textAlign: 'right' }}>{agent.accuracy || 0}%</div>
                            <div style={{ color: '#ccc', textAlign: 'right' }}>{agent.trades || 0}</div>
                            <div style={{ 
                              color: agent.status === 'ACTIVE' ? '#00ff00' : agent.status === 'TRAINING' ? '#ffa500' : '#ff0000',
                              textAlign: 'center',
                              fontWeight: 'bold',
                              fontSize: '7px'
                            }}>
                              {agent.status}
                            </div>
                            <div style={{ textAlign: 'center' }}>
                              <button style={{
                                background: agent.status === 'ACTIVE' ? '#ff0000' : '#00ff00',
                                color: '#000',
                                border: 'none',
                                padding: '2px 6px',
                                fontSize: '7px',
                                cursor: 'pointer'
                              }}>
                                {agent.status === 'ACTIVE' ? 'STOP' : 'START'}
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Coordination Layer */}
                      <div style={{ marginBottom: '4px' }}>
                        <div style={{ color: '#ff00ff', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          🎛️ COORDINATION LAYER
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                          {/* Trainer/Coordinator */}
                          <div style={{ padding: '6px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
                            <div style={{ color: '#ff00ff', fontSize: '8px', fontWeight: 'bold', marginBottom: '3px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>TRAINER/COORDINATOR</span>
                              <div style={{ 
                                width: '6px', 
                                height: '6px', 
                                borderRadius: '50%', 
                                background: agents.trainer.status === 'ACTIVE' ? '#00ff00' : '#ff0000' 
                              }}></div>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Models Training: </span>
                              <span style={{ color: '#ffa500' }}>{agents.trainer.modelsTraining}</span>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Epoch: </span>
                              <span style={{ color: '#ffa500' }}>{agents.trainer.epoch}</span>
                            </div>
                            <div style={{ fontSize: '7px' }}>
                              <span style={{ color: '#ccc' }}>ETA: </span>
                              <span style={{ color: '#00ff00' }}>{agents.trainer.eta}</span>
                            </div>
                          </div>
                          
                          {/* Risk Manager */}
                          <div style={{ padding: '6px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
                            <div style={{ color: '#ff00ff', fontSize: '8px', fontWeight: 'bold', marginBottom: '3px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>RISK MANAGER</span>
                              <div style={{ 
                                width: '6px', 
                                height: '6px', 
                                borderRadius: '50%', 
                                background: agents.riskManager.status === 'ACTIVE' ? '#00ff00' : '#ff0000' 
                              }}></div>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>VaR: </span>
                              <span style={{ color: agents.riskManager.var > 0.03 ? '#ff0000' : '#ffa500' }}>
                                {(agents.riskManager.var * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Sharpe: </span>
                              <span style={{ color: '#00ff00' }}>{agents.riskManager.sharpe}</span>
                            </div>
                            <div style={{ fontSize: '7px' }}>
                              <span style={{ color: '#ccc' }}>Exposure: </span>
                              <span style={{ color: agents.riskManager.exposure > 0.8 ? '#ff0000' : '#00ff00' }}>
                                {(agents.riskManager.exposure * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginTop: '6px' }}>
                          {/* Master Agent */}
                          <div style={{ padding: '6px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
                            <div style={{ color: '#ff00ff', fontSize: '8px', fontWeight: 'bold', marginBottom: '3px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>MASTER AGENT</span>
                              <div style={{ 
                                width: '6px', 
                                height: '6px', 
                                borderRadius: '50%', 
                                background: agents.masterAgent.status === 'ACTIVE' ? '#00ff00' : '#ff0000' 
                              }}></div>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Confidence: </span>
                              <span style={{ color: '#00ff00' }}>{(agents.masterAgent.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Action: </span>
                              <span style={{ 
                                color: agents.masterAgent.action === 'BUY_SIGNAL' ? '#00ff00' : 
                                       agents.masterAgent.action === 'SELL_SIGNAL' ? '#ff0000' : '#ffa500' 
                              }}>
                                {agents.masterAgent.action}
                              </span>
                            </div>
                            <div style={{ fontSize: '7px' }}>
                              <span style={{ color: '#ccc' }}>Target: </span>
                              <span style={{ color: '#00ff00' }}>{agents.masterAgent.target}</span>
                            </div>
                          </div>
                          
                          {/* Execution Engine */}
                          <div style={{ padding: '6px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
                            <div style={{ color: '#ff00ff', fontSize: '8px', fontWeight: 'bold', marginBottom: '3px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>EXECUTION ENGINE</span>
                              <div style={{ 
                                width: '6px', 
                                height: '6px', 
                                borderRadius: '50%', 
                                background: agents.execution.status === 'ACTIVE' ? '#00ff00' : '#ff0000' 
                              }}></div>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Orders/sec: </span>
                              <span style={{ color: '#00ff00' }}>{agents.execution.ordersPerSec}</span>
                            </div>
                            <div style={{ fontSize: '7px', marginBottom: '2px' }}>
                              <span style={{ color: '#ccc' }}>Fill Rate: </span>
                              <span style={{ color: '#00ff00' }}>{agents.execution.fillRate}%</span>
                            </div>
                            <div style={{ fontSize: '7px' }}>
                              <span style={{ color: '#ccc' }}>Latency: </span>
                              <span style={{ color: '#ffa500' }}>{agents.execution.latency}ms</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Agent Performance & Controls */}
                  <div>
                    <div style={{ color: '#ff00ff', fontSize: '9px', fontWeight: 'bold', marginBottom: '6px' }}>
                      PERFORMANCE & CONTROLS
                    </div>
                    
                    <div style={{ 
                      height: 'calc(100% - 25px)', 
                      overflowY: 'auto', 
                      fontSize: '8px'
                    }}>
                      {/* System Health */}
                      <div style={{ marginBottom: '8px', border: '1px solid #333', padding: '6px' }}>
                        <div style={{ color: '#00ffff', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          SYSTEM HEALTH
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '8px' }}>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>CPU USAGE</div>
                            <div style={{ color: '#00ff00', fontWeight: 'bold' }}>23.4%</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>MEMORY</div>
                            <div style={{ color: '#ffa500', fontWeight: 'bold' }}>67.8%</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>LATENCY</div>
                            <div style={{ color: '#00ff00', fontWeight: 'bold' }}>12ms</div>
                          </div>
                          <div style={{ textAlign: 'center', background: '#1a1a1a', padding: '4px' }}>
                            <div style={{ color: '#ffa500', fontSize: '7px' }}>UPTIME</div>
                            <div style={{ color: '#00ff00', fontWeight: 'bold' }}>47.2h</div>
                          </div>
                        </div>
                      </div>

                      {/* Top Performers */}
                      <div style={{ marginBottom: '8px', border: '1px solid #333', padding: '6px' }}>
                        <div style={{ color: '#00ff00', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          TOP PERFORMERS (24H)
                        </div>
                        
                        {[
                          { name: 'Statistical Arbitrage', pnl: 3456, roi: 12.3, color: '#00ff00' },
                          { name: 'Meta Learning', pnl: 2891, roi: 8.7, color: '#00ff00' },
                          { name: 'Market Making', pnl: 1234, roi: 4.2, color: '#ffa500' },
                          { name: 'Volatility Trading', pnl: -567, roi: -2.1, color: '#ff0000' }
                        ].map((performer, i) => (
                          <div key={i} style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 60px 40px',
                            gap: '4px',
                            marginBottom: '3px',
                            padding: '3px',
                            background: '#1a1a1a',
                            fontSize: '7px'
                          }}>
                            <div style={{ color: '#ccc', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {performer.name}
                            </div>
                            <div style={{ color: performer.color, textAlign: 'right', fontWeight: 'bold' }}>
                              ${performer.pnl.toLocaleString()}
                            </div>
                            <div style={{ color: performer.color, textAlign: 'right' }}>
                              {performer.roi > 0 ? '+' : ''}{performer.roi}%
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Recent Signals */}
                      <div style={{ marginBottom: '8px', border: '1px solid #333', padding: '6px' }}>
                        <div style={{ color: '#ffa500', fontSize: '9px', fontWeight: 'bold', marginBottom: '4px' }}>
                          RECENT SIGNALS
                        </div>
                        
                        {[
                          { time: '14:32', agent: 'Momentum', symbol: 'NVDA', signal: 'BUY', confidence: 87 },
                          { time: '14:30', agent: 'StatArb', symbol: 'AAPL', signal: 'SELL', confidence: 93 },
                          { time: '14:28', agent: 'Options', symbol: 'TSLA', signal: 'BUY', confidence: 76 },
                          { time: '14:25', agent: 'Mean Rev', symbol: 'SPY', signal: 'HOLD', confidence: 65 }
                        ].map((signal, i) => (
                          <div key={i} style={{
                            display: 'grid',
                            gridTemplateColumns: '30px 60px 35px 35px 25px',
                            gap: '3px',
                            marginBottom: '2px',
                            padding: '2px',
                            background: i % 2 === 0 ? '#111' : 'transparent',
                            fontSize: '7px'
                          }}>
                            <div style={{ color: '#666' }}>{signal.time}</div>
                            <div style={{ color: '#00ffff', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {signal.agent}
                            </div>
                            <div style={{ color: '#ffa500' }}>{signal.symbol}</div>
                            <div style={{ 
                              color: signal.signal === 'BUY' ? '#00ff00' : 
                                     signal.signal === 'SELL' ? '#ff0000' : '#ffa500',
                              fontWeight: 'bold'
                            }}>
                              {signal.signal}
                            </div>
                            <div style={{ color: '#ccc', textAlign: 'right' }}>{signal.confidence}%</div>
                          </div>
                        ))}
                      </div>

                      {/* Master Controls */}
                      <div style={{ border: '1px solid #333', padding: '6px' }}>
                        <div style={{ color: '#ff0000', fontSize: '9px', fontWeight: 'bold', marginBottom: '6px' }}>
                          MASTER CONTROLS
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '6px' }}>
                          <button style={{
                            background: '#00ff00',
                            color: '#000',
                            border: 'none',
                            padding: '6px',
                            fontSize: '8px',
                            fontWeight: 'bold',
                            cursor: 'pointer'
                          }}>
                            START ALL
                          </button>
                          <button style={{
                            background: '#ff0000',
                            color: '#fff',
                            border: 'none',
                            padding: '6px',
                            fontSize: '8px',
                            fontWeight: 'bold',
                            cursor: 'pointer'
                          }}>
                            STOP ALL
                          </button>
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '3px' }}>
                          <button style={{
                            background: '#ffa500',
                            color: '#000',
                            border: 'none',
                            padding: '4px',
                            fontSize: '7px',
                            cursor: 'pointer'
                          }}>
                            EMERGENCY STOP
                          </button>
                          <button style={{
                            background: '#333',
                            color: '#ffa500',
                            border: '1px solid #666',
                            padding: '4px',
                            fontSize: '7px',
                            cursor: 'pointer'
                          }}>
                            RESET SYSTEM
                          </button>
                          <button style={{
                            background: '#333',
                            color: '#00ffff',
                            border: '1px solid #666',
                            padding: '4px',
                            fontSize: '7px',
                            cursor: 'pointer'
                          }}>
                            EXPORT LOGS
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'NEWS' && (
              <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 300px', gap: '10px', padding: '10px' }}>
                {/* Left Panel - News Feed */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* News Header */}
                  <div style={{
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    padding: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr auto auto',
                    gap: '10px',
                    alignItems: 'center'
                  }}>
                    <div style={{ color: '#ffa500', fontSize: '12px', fontWeight: 'bold' }}>
                      LIVE NEWS FEED
                    </div>
                    <div></div>
                    <div style={{ color: '#00ff00', fontSize: '10px' }}>
                      ●LIVE {newsItems.length} ITEMS
                    </div>
                    <button style={{
                      background: '#333',
                      color: '#ffa500',
                      border: '1px solid #666',
                      padding: '4px 8px',
                      fontSize: '9px',
                      cursor: 'pointer'
                    }}>
                      REFRESH
                    </button>
                  </div>

                  {/* News Items List */}
                  <div style={{
                    flex: 1,
                    background: '#000',
                    border: '1px solid #333',
                    padding: '5px',
                    overflow: 'auto'
                  }}>
                    {newsItems.map((item, index) => (
                      <div key={index} style={{
                        border: '1px solid #222',
                        marginBottom: '6px',
                        padding: '8px',
                        background: '#0a0a0a'
                      }}>
                        {/* News Item Header */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <div style={{
                            color: item.sentiment === 'BULLISH' ? '#00ff00' : 
                                   item.sentiment === 'BEARISH' ? '#ff0000' : '#ffa500',
                            fontSize: '9px',
                            fontWeight: 'bold'
                          }}>
                            {item.sentiment} | {item.impact}
                          </div>
                          <div style={{ color: '#666', fontSize: '8px' }}>
                            {item.timestamp} | {item.source}
                          </div>
                        </div>

                        {/* Title */}
                        <div style={{
                          color: '#fff',
                          fontSize: '11px',
                          fontWeight: 'bold',
                          marginBottom: '4px',
                          cursor: 'pointer'
                        }}>
                          {item.title}
                        </div>

                        {/* Summary */}
                        <div style={{
                          color: '#ccc',
                          fontSize: '9px',
                          marginBottom: '4px',
                          lineHeight: '1.3'
                        }}>
                          {item.summary}
                        </div>

                        {/* Tags & Symbols */}
                        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '4px' }}>
                          {item.symbols.map(symbol => (
                            <span key={symbol} style={{
                              background: '#333',
                              color: '#00ffff',
                              padding: '2px 4px',
                              fontSize: '8px',
                              border: '1px solid #555'
                            }}>
                              {symbol}
                            </span>
                          ))}
                          {item.tags.map(tag => (
                            <span key={tag} style={{
                              background: '#1a1a1a',
                              color: '#ffa500',
                              padding: '2px 4px',
                              fontSize: '8px',
                              border: '1px solid #333'
                            }}>
                              {tag}
                            </span>
                          ))}
                        </div>

                        {/* Metrics */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', fontSize: '8px' }}>
                          <div>
                            <span style={{ color: '#666' }}>SCORE:</span>
                            <span style={{ 
                              color: item.sentiment_score > 0 ? '#00ff00' : '#ff0000',
                              marginLeft: '2px'
                            }}>
                              {item.sentiment_score.toFixed(2)}
                            </span>
                          </div>
                          <div>
                            <span style={{ color: '#666' }}>CONF:</span>
                            <span style={{ color: '#ffa500', marginLeft: '2px' }}>
                              {(item.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div>
                            <span style={{ color: '#666' }}>REACH:</span>
                            <span style={{ color: '#00ffff', marginLeft: '2px' }}>
                              {item.reach}K
                            </span>
                          </div>
                          <div>
                            <span style={{ color: '#666' }}>VIRAL:</span>
                            <span style={{ 
                              color: item.virality > 50 ? '#ff0000' : '#ffa500',
                              marginLeft: '2px'
                            }}>
                              {item.virality}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right Panel - Sentiment Dashboard */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* Market Sentiment Overview */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      MARKET SENTIMENT
                    </div>
                    
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                        <span style={{ fontSize: '8px', color: '#ccc' }}>OVERALL</span>
                        <span style={{ fontSize: '8px', color: sentimentData.overall > 0 ? '#00ff00' : '#ff0000' }}>
                          {sentimentData.overall > 0 ? 'BULLISH' : 'BEARISH'} {Math.abs(sentimentData.overall).toFixed(1)}
                        </span>
                      </div>
                      <div style={{
                        height: '4px',
                        background: '#333',
                        position: 'relative',
                        border: '1px solid #555'
                      }}>
                        <div style={{
                          position: 'absolute',
                          top: 0,
                          left: sentimentData.overall < 0 ? `${50 + sentimentData.overall}%` : '50%',
                          width: `${Math.abs(sentimentData.overall)}%`,
                          height: '100%',
                          background: sentimentData.overall > 0 ? '#00ff00' : '#ff0000'
                        }}></div>
                        <div style={{
                          position: 'absolute',
                          top: '-1px',
                          left: '50%',
                          width: '1px',
                          height: '6px',
                          background: '#fff'
                        }}></div>
                      </div>
                    </div>

                    {/* Sector Sentiment */}
                    <div style={{ fontSize: '8px' }}>
                      {Object.entries(sentimentData.sectors).map(([sector, value]) => (
                        <div key={sector} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{ color: '#666' }}>{sector}</span>
                          <span style={{ color: value > 0 ? '#00ff00' : '#ff0000' }}>
                            {value > 0 ? '+' : ''}{value.toFixed(1)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Social Media Buzz */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      SOCIAL BUZZ
                    </div>
                    
                    {socialBuzz.map((item, index) => (
                      <div key={index} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '4px',
                        fontSize: '8px'
                      }}>
                        <div>
                          <span style={{ color: '#00ffff' }}>{item.symbol}</span>
                          <span style={{ color: '#666', marginLeft: '4px' }}>{item.platform}</span>
                        </div>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          <span style={{ color: '#ffa500' }}>{item.mentions}</span>
                          <span style={{ 
                            color: item.sentiment > 0 ? '#00ff00' : '#ff0000'
                          }}>
                            {item.sentiment > 0 ? '+' : ''}{item.sentiment.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Fear & Greed Index */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      FEAR & GREED INDEX
                    </div>
                    
                    <div style={{ textAlign: 'center', marginBottom: '8px' }}>
                      <div style={{ 
                        color: fearGreedIndex >= 75 ? '#ff0000' : 
                               fearGreedIndex >= 55 ? '#ffa500' :
                               fearGreedIndex >= 45 ? '#ffff00' :
                               fearGreedIndex >= 25 ? '#00ff00' : '#00ffff',
                        fontSize: '16px',
                        fontWeight: 'bold'
                      }}>
                        {fearGreedIndex}
                      </div>
                      <div style={{ 
                        color: '#ccc',
                        fontSize: '8px'
                      }}>
                        {fearGreedIndex >= 75 ? 'EXTREME GREED' : 
                         fearGreedIndex >= 55 ? 'GREED' :
                         fearGreedIndex >= 45 ? 'NEUTRAL' :
                         fearGreedIndex >= 25 ? 'FEAR' : 'EXTREME FEAR'}
                      </div>
                    </div>

                    {/* Fear/Greed Gauge */}
                    <div style={{
                      height: '6px',
                      background: 'linear-gradient(to right, #00ffff, #00ff00, #ffff00, #ffa500, #ff0000)',
                      border: '1px solid #333',
                      position: 'relative',
                      marginBottom: '8px'
                    }}>
                      <div style={{
                        position: 'absolute',
                        top: '-2px',
                        left: `${fearGreedIndex}%`,
                        width: '2px',
                        height: '10px',
                        background: '#fff',
                        border: '1px solid #000'
                      }}></div>
                    </div>

                    <div style={{ fontSize: '7px', color: '#666' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>FEAR</span>
                        <span>NEUTRAL</span>
                        <span>GREED</span>
                      </div>
                    </div>
                  </div>

                  {/* Event Calendar */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px', flex: 1 }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      UPCOMING EVENTS
                    </div>
                    
                    {upcomingEvents.map((event, index) => (
                      <div key={index} style={{
                        marginBottom: '6px',
                        padding: '4px',
                        background: '#0a0a0a',
                        border: '1px solid #222',
                        fontSize: '8px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{
                            color: event.impact === 'HIGH' ? '#ff0000' :
                                   event.impact === 'MEDIUM' ? '#ffa500' : '#00ff00'
                          }}>
                            {event.impact}
                          </span>
                          <span style={{ color: '#666' }}>{event.time}</span>
                        </div>
                        <div style={{ color: '#fff', marginBottom: '2px' }}>{event.title}</div>
                        <div style={{ color: '#ccc', fontSize: '7px' }}>{event.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'PERFORMANCE' && (
              <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 400px', gap: '10px', padding: '10px' }}>
                {/* Left Panel - Performance Analytics */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* Performance Header */}
                  <div style={{
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    padding: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr auto auto auto',
                    gap: '10px',
                    alignItems: 'center'
                  }}>
                    <div style={{ color: '#ffa500', fontSize: '12px', fontWeight: 'bold' }}>
                      PORTFOLIO PERFORMANCE
                    </div>
                    <div></div>
                    <select style={{
                      background: '#333',
                      color: '#ffa500',
                      border: '1px solid #666',
                      padding: '4px',
                      fontSize: '9px'
                    }}>
                      <option>1D</option>
                      <option>1W</option>
                      <option selected>1M</option>
                      <option>3M</option>
                      <option>YTD</option>
                      <option>1Y</option>
                    </select>
                    <div style={{ color: '#00ff00', fontSize: '10px' }}>
                      ●LIVE TRACKING
                    </div>
                    <button style={{
                      background: '#333',
                      color: '#ffa500',
                      border: '1px solid #666',
                      padding: '4px 8px',
                      fontSize: '9px',
                      cursor: 'pointer'
                    }}>
                      EXPORT
                    </button>
                  </div>

                  {/* Performance Summary Cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
                    {performanceMetrics.map((metric, index) => (
                      <div key={index} style={{
                        background: '#1a1a1a',
                        border: '1px solid #333',
                        padding: '12px',
                        textAlign: 'center'
                      }}>
                        <div style={{ color: '#666', fontSize: '8px', marginBottom: '4px' }}>
                          {metric.label}
                        </div>
                        <div style={{
                          color: metric.positive ? '#00ff00' : metric.positive === false ? '#ff0000' : '#ffa500',
                          fontSize: '16px',
                          fontWeight: 'bold',
                          marginBottom: '2px'
                        }}>
                          {metric.value}
                        </div>
                        <div style={{ color: '#666', fontSize: '7px' }}>
                          {metric.change}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Performance Chart */}
                  <div style={{ background: '#000', border: '1px solid #333', padding: '10px', flex: 1, minHeight: '200px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '10px' }}>
                      EQUITY CURVE
                    </div>
                    
                    <svg width="100%" height="180" viewBox="0 0 800 180">
                      {/* Grid lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
                        const y = 20 + ratio * 140;
                        const value = 1000000 + (1 - ratio) * 500000;
                        return (
                          <g key={ratio}>
                            <line
                              x1={40}
                              y1={y}
                              x2={760}
                              y2={y}
                              stroke="#333"
                              strokeWidth="0.5"
                              strokeDasharray="2,2"
                            />
                            <text
                              x={35}
                              y={y + 3}
                              fill="#666"
                              fontSize="8"
                              textAnchor="end"
                              fontFamily="Courier New"
                            >
                              ${(value/1000).toFixed(0)}K
                            </text>
                          </g>
                        );
                      })}

                      {/* Equity curve */}
                      <polyline
                        fill="none"
                        stroke="#00ff00"
                        strokeWidth="2"
                        opacity="0.9"
                        points={equityCurveData.map((point, i) => {
                          const x = 40 + (i / (equityCurveData.length - 1)) * 720;
                          const y = 20 + (1 - (point - 1000000) / 500000) * 140;
                          return `${x},${y}`;
                        }).join(' ')}
                      />

                      {/* Benchmark comparison */}
                      <polyline
                        fill="none"
                        stroke="#666"
                        strokeWidth="1"
                        strokeDasharray="3,3"
                        opacity="0.7"
                        points={benchmarkData.map((point, i) => {
                          const x = 40 + (i / (benchmarkData.length - 1)) * 720;
                          const y = 20 + (1 - (point - 1000000) / 500000) * 140;
                          return `${x},${y}`;
                        }).join(' ')}
                      />
                    </svg>

                    <div style={{ display: 'flex', gap: '15px', marginTop: '8px', fontSize: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <div style={{ width: '8px', height: '2px', background: '#00ff00' }}></div>
                        <span style={{ color: '#ccc' }}>Portfolio</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <div style={{ width: '8px', height: '2px', background: '#666', borderTop: '1px dashed #666' }}></div>
                        <span style={{ color: '#ccc' }}>S&P 500</span>
                      </div>
                    </div>
                  </div>

                  {/* Strategy Performance Breakdown */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '10px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      STRATEGY PERFORMANCE BREAKDOWN
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                      {strategyPerformance.map((strategy, index) => (
                        <div key={index} style={{
                          background: '#0a0a0a',
                          border: '1px solid #222',
                          padding: '8px'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                            <div style={{ color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>
                              {strategy.name}
                            </div>
                            <div style={{
                              color: strategy.active ? '#00ff00' : '#666',
                              fontSize: '8px'
                            }}>
                              {strategy.active ? 'ACTIVE' : 'INACTIVE'}
                            </div>
                          </div>
                          
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '8px' }}>
                            <div>
                              <div style={{ color: '#666' }}>P&L</div>
                              <div style={{ color: strategy.pnl > 0 ? '#00ff00' : '#ff0000', fontWeight: 'bold' }}>
                                ${strategy.pnl.toLocaleString()}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#666' }}>WIN%</div>
                              <div style={{ color: '#ffa500', fontWeight: 'bold' }}>
                                {strategy.winRate.toFixed(1)}%
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#666' }}>SHARP</div>
                              <div style={{ color: '#00ffff', fontWeight: 'bold' }}>
                                {strategy.sharpe.toFixed(2)}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#666' }}>TRADES</div>
                              <div style={{ color: '#ccc' }}>
                                {strategy.trades}
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#666' }}>MAX DD</div>
                              <div style={{ color: '#ff0000' }}>
                                {(strategy.maxDrawdown * 100).toFixed(1)}%
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#666' }}>ALLOC</div>
                              <div style={{ color: '#ffa500' }}>
                                {(strategy.allocation * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right Panel - Backtesting & Analysis */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* Backtesting Controls */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      BACKTESTING ENGINE
                    </div>
                    
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Date Range</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                        <input
                          type="date"
                          defaultValue="2024-01-01"
                          style={{
                            background: '#333',
                            border: '1px solid #555',
                            color: '#ffa500',
                            padding: '4px',
                            fontSize: '8px'
                          }}
                        />
                        <input
                          type="date"
                          defaultValue="2024-12-31"
                          style={{
                            background: '#333',
                            border: '1px solid #555',
                            color: '#ffa500',
                            padding: '4px',
                            fontSize: '8px'
                          }}
                        />
                      </div>
                    </div>

                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Initial Capital</div>
                      <input
                        type="text"
                        defaultValue="$1,000,000"
                        style={{
                          background: '#333',
                          border: '1px solid #555',
                          color: '#00ff00',
                          padding: '4px',
                          fontSize: '9px',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Strategies to Test</div>
                      {['Momentum', 'Mean Reversion', 'Statistical Arbitrage', 'Options Sentiment', 'Market Making'].map(strategy => (
                        <div key={strategy} style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
                          <input type="checkbox" defaultChecked style={{ scale: '0.8' }} />
                          <span style={{ fontSize: '8px', color: '#ccc' }}>{strategy}</span>
                        </div>
                      ))}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                      <button style={{
                        background: '#00ff00',
                        color: '#000',
                        border: 'none',
                        padding: '6px',
                        fontSize: '9px',
                        fontWeight: 'bold',
                        cursor: 'pointer'
                      }}>
                        RUN BACKTEST
                      </button>
                      <button style={{
                        background: '#333',
                        color: '#ffa500',
                        border: '1px solid #666',
                        padding: '6px',
                        fontSize: '9px',
                        cursor: 'pointer'
                      }}>
                        OPTIMIZE
                      </button>
                    </div>
                  </div>

                  {/* Risk Metrics */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      RISK ANALYSIS
                    </div>
                    
                    {riskMetrics.map((metric, index) => (
                      <div key={index} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '4px',
                        fontSize: '8px'
                      }}>
                        <span style={{ color: '#ccc' }}>{metric.name}</span>
                        <span style={{
                          color: metric.status === 'GOOD' ? '#00ff00' :
                                 metric.status === 'WARNING' ? '#ffa500' : '#ff0000'
                        }}>
                          {metric.value}
                        </span>
                      </div>
                    ))}

                    <div style={{ marginTop: '8px', padding: '6px', background: '#0a0a0a', border: '1px solid #222' }}>
                      <div style={{ color: '#ffa500', fontSize: '8px', fontWeight: 'bold', marginBottom: '4px' }}>
                        POSITION SIZING
                      </div>
                      <div style={{ fontSize: '7px', color: '#ccc', lineHeight: '1.3' }}>
                        Kelly Criterion: 2.3% per trade<br/>
                        Max Position: 5% per symbol<br/>
                        Portfolio Heat: 15% max
                      </div>
                    </div>
                  </div>

                  {/* Recent Trades */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px', flex: 1 }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      RECENT TRADES
                    </div>
                    
                    {recentTrades.map((trade, index) => (
                      <div key={index} style={{
                        background: '#0a0a0a',
                        border: '1px solid #222',
                        padding: '6px',
                        marginBottom: '4px',
                        fontSize: '8px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{ color: '#00ffff' }}>{trade.symbol}</span>
                          <span style={{
                            color: trade.side === 'BUY' ? '#00ff00' : '#ff0000'
                          }}>
                            {trade.side} {trade.quantity}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{ color: '#666' }}>{trade.time}</span>
                          <span style={{ color: '#ffa500' }}>@${trade.price.toFixed(2)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#666' }}>{trade.strategy}</span>
                          <span style={{
                            color: trade.pnl > 0 ? '#00ff00' : '#ff0000'
                          }}>
                            {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(0)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'ORDERS' && (
              <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 350px', gap: '10px', padding: '10px' }}>
                {/* Left Panel - Order Management */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* Orders Header */}
                  <div style={{
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    padding: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr auto auto auto',
                    gap: '10px',
                    alignItems: 'center'
                  }}>
                    <div style={{ color: '#ffa500', fontSize: '12px', fontWeight: 'bold' }}>
                      ORDER MANAGEMENT
                    </div>
                    <div></div>
                    <button style={{
                      background: '#333',
                      color: '#ffa500',
                      border: '1px solid #666',
                      padding: '4px 8px',
                      fontSize: '9px',
                      cursor: 'pointer'
                    }}>
                      FILTER
                    </button>
                    <div style={{ color: '#00ff00', fontSize: '10px' }}>
                      ●{activeOrders.length + pendingOrders.length} ORDERS
                    </div>
                    <button style={{
                      background: '#ff0000',
                      color: '#fff',
                      border: 'none',
                      padding: '4px 8px',
                      fontSize: '9px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      CANCEL ALL
                    </button>
                  </div>

                  {/* Order Tabs */}
                  <div style={{ display: 'flex', gap: '2px' }}>
                    {['ACTIVE', 'PENDING', 'FILLED', 'CANCELLED'].map(tab => (
                      <button key={tab} style={{
                        background: orderTab === tab ? '#ffa500' : '#333',
                        color: orderTab === tab ? '#000' : '#ccc',
                        border: '1px solid #666',
                        padding: '6px 12px',
                        fontSize: '9px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        flex: 1
                      }}
                      onClick={() => setOrderTab(tab)}
                      >
                        {tab} ({
                          tab === 'ACTIVE' ? activeOrders.length :
                          tab === 'PENDING' ? pendingOrders.length :
                          tab === 'FILLED' ? filledOrders.length :
                          cancelledOrders.length
                        })
                      </button>
                    ))}
                  </div>

                  {/* Orders List */}
                  <div style={{
                    flex: 1,
                    background: '#000',
                    border: '1px solid #333',
                    overflow: 'auto'
                  }}>
                    {/* Column Headers */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '80px 60px 80px 80px 80px 80px 60px 120px 80px',
                      gap: '4px',
                      padding: '8px',
                      background: '#1a1a1a',
                      borderBottom: '1px solid #333',
                      fontSize: '8px',
                      color: '#666',
                      fontWeight: 'bold'
                    }}>
                      <div>SYMBOL</div>
                      <div>SIDE</div>
                      <div>QTY</div>
                      <div>PRICE</div>
                      <div>FILLED</div>
                      <div>REMAIN</div>
                      <div>TYPE</div>
                      <div>TIME</div>
                      <div>ACTION</div>
                    </div>

                    {/* Order Rows */}
                    {(orderTab === 'ACTIVE' ? activeOrders :
                      orderTab === 'PENDING' ? pendingOrders :
                      orderTab === 'FILLED' ? filledOrders :
                      cancelledOrders).map((order, index) => (
                      <div key={index} style={{
                        display: 'grid',
                        gridTemplateColumns: '80px 60px 80px 80px 80px 80px 60px 120px 80px',
                        gap: '4px',
                        padding: '6px 8px',
                        borderBottom: '1px solid #222',
                        fontSize: '9px',
                        alignItems: 'center',
                        background: index % 2 === 0 ? '#0a0a0a' : 'transparent'
                      }}>
                        <div style={{ color: '#00ffff', fontWeight: 'bold' }}>{order.symbol}</div>
                        <div style={{
                          color: order.side === 'BUY' ? '#00ff00' : '#ff0000',
                          fontWeight: 'bold'
                        }}>
                          {order.side}
                        </div>
                        <div style={{ color: '#ccc' }}>{order.quantity.toLocaleString()}</div>
                        <div style={{ color: '#ffa500' }}>
                          {order.price === 'MKT' ? 'MKT' : `$${parseFloat(order.price).toFixed(2)}`}
                        </div>
                        <div style={{ color: '#00ff00' }}>{order.filled}</div>
                        <div style={{ color: '#ccc' }}>{order.remaining}</div>
                        <div style={{ color: '#666' }}>{order.type}</div>
                        <div style={{ color: '#666' }}>{order.time}</div>
                        <div>
                          {orderTab === 'ACTIVE' && (
                            <button style={{
                              background: '#ff0000',
                              color: '#fff',
                              border: 'none',
                              padding: '2px 6px',
                              fontSize: '7px',
                              cursor: 'pointer'
                            }}>
                              CANCEL
                            </button>
                          )}
                          {orderTab === 'PENDING' && (
                            <button style={{
                              background: '#ffa500',
                              color: '#000',
                              border: 'none',
                              padding: '2px 6px',
                              fontSize: '7px',
                              cursor: 'pointer'
                            }}>
                              MODIFY
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Execution Statistics */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      EXECUTION STATISTICS
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
                      {executionStats.map((stat, index) => (
                        <div key={index} style={{ textAlign: 'center' }}>
                          <div style={{ color: '#666', fontSize: '8px', marginBottom: '2px' }}>
                            {stat.label}
                          </div>
                          <div style={{
                            color: stat.positive ? '#00ff00' : stat.positive === false ? '#ff0000' : '#ffa500',
                            fontSize: '14px',
                            fontWeight: 'bold'
                          }}>
                            {stat.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right Panel - Order Entry & Execution */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {/* Order Entry */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '10px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      ORDER ENTRY
                    </div>
                    
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Symbol</div>
                      <input
                        type="text"
                        value={orderEntry.symbol}
                        onChange={(e) => setOrderEntry(prev => ({...prev, symbol: e.target.value.toUpperCase()}))}
                        style={{
                          background: '#333',
                          border: '1px solid #555',
                          color: '#00ffff',
                          padding: '6px',
                          fontSize: '11px',
                          width: '100%',
                          boxSizing: 'border-box',
                          fontWeight: 'bold'
                        }}
                        placeholder="Enter symbol..."
                      />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '8px' }}>
                      <div>
                        <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Side</div>
                        <select
                          value={orderEntry.side}
                          onChange={(e) => setOrderEntry(prev => ({...prev, side: e.target.value}))}
                          style={{
                            background: '#333',
                            border: '1px solid #555',
                            color: orderEntry.side === 'BUY' ? '#00ff00' : '#ff0000',
                            padding: '6px',
                            fontSize: '10px',
                            width: '100%',
                            fontWeight: 'bold'
                          }}
                        >
                          <option value="BUY" style={{background: '#333', color: '#00ff00'}}>BUY</option>
                          <option value="SELL" style={{background: '#333', color: '#ff0000'}}>SELL</option>
                        </select>
                      </div>
                      <div>
                        <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Quantity</div>
                        <input
                          type="number"
                          value={orderEntry.quantity}
                          onChange={(e) => setOrderEntry(prev => ({...prev, quantity: parseInt(e.target.value) || 0}))}
                          style={{
                            background: '#333',
                            border: '1px solid #555',
                            color: '#ffa500',
                            padding: '6px',
                            fontSize: '10px',
                            width: '100%',
                            boxSizing: 'border-box'
                          }}
                          placeholder="0"
                        />
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '8px' }}>
                      <div>
                        <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Order Type</div>
                        <select
                          value={orderEntry.type}
                          onChange={(e) => setOrderEntry(prev => ({...prev, type: e.target.value}))}
                          style={{
                            background: '#333',
                            border: '1px solid #555',
                            color: '#ffa500',
                            padding: '6px',
                            fontSize: '9px',
                            width: '100%'
                          }}
                        >
                          <option value="MARKET">MARKET</option>
                          <option value="LIMIT">LIMIT</option>
                          <option value="STOP">STOP</option>
                          <option value="STOP_LIMIT">STOP LIMIT</option>
                        </select>
                      </div>
                      <div>
                        <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Price</div>
                        <input
                          type="number"
                          step="0.01"
                          value={orderEntry.price}
                          onChange={(e) => setOrderEntry(prev => ({...prev, price: parseFloat(e.target.value) || 0}))}
                          disabled={orderEntry.type === 'MARKET'}
                          style={{
                            background: orderEntry.type === 'MARKET' ? '#222' : '#333',
                            border: '1px solid #555',
                            color: orderEntry.type === 'MARKET' ? '#666' : '#ffa500',
                            padding: '6px',
                            fontSize: '10px',
                            width: '100%',
                            boxSizing: 'border-box'
                          }}
                          placeholder={orderEntry.type === 'MARKET' ? 'MARKET' : '0.00'}
                        />
                      </div>
                    </div>

                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ color: '#ccc', fontSize: '8px', marginBottom: '3px' }}>Time in Force</div>
                      <select
                        value={orderEntry.tif}
                        onChange={(e) => setOrderEntry(prev => ({...prev, tif: e.target.value}))}
                        style={{
                          background: '#333',
                          border: '1px solid #555',
                          color: '#ffa500',
                          padding: '6px',
                          fontSize: '9px',
                          width: '100%'
                        }}
                      >
                        <option value="DAY">DAY</option>
                        <option value="GTC">GTC</option>
                        <option value="IOC">IOC</option>
                        <option value="FOK">FOK</option>
                      </select>
                    </div>

                    {/* Order Preview */}
                    <div style={{
                      background: '#0a0a0a',
                      border: '1px solid #333',
                      padding: '8px',
                      marginBottom: '8px',
                      fontSize: '8px'
                    }}>
                      <div style={{ color: '#ffa500', fontWeight: 'bold', marginBottom: '4px' }}>
                        ORDER PREVIEW
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                        <div>
                          <span style={{ color: '#666' }}>Notional:</span>
                          <span style={{ color: '#00ff00', marginLeft: '4px' }}>
                            ${(orderEntry.quantity * (orderEntry.price || marketData[orderEntry.symbol]?.price || 0)).toLocaleString()}
                          </span>
                        </div>
                        <div>
                          <span style={{ color: '#666' }}>Est. Fee:</span>
                          <span style={{ color: '#ffa500', marginLeft: '4px' }}>
                            ${(orderEntry.quantity * (orderEntry.price || marketData[orderEntry.symbol]?.price || 0) * 0.0005).toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                      <button style={{
                        background: '#00ff00',
                        color: '#000',
                        border: 'none',
                        padding: '10px',
                        fontSize: '10px',
                        fontWeight: 'bold',
                        cursor: 'pointer'
                      }}>
                        SUBMIT ORDER
                      </button>
                      <button style={{
                        background: '#333',
                        color: '#ffa500',
                        border: '1px solid #666',
                        padding: '10px',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        CLEAR
                      </button>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px' }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      QUICK ACTIONS
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '6px' }}>
                      <button style={{
                        background: '#333',
                        color: '#ffa500',
                        border: '1px solid #666',
                        padding: '6px',
                        fontSize: '8px',
                        cursor: 'pointer'
                      }}>
                        FLATTEN ALL
                      </button>
                      <button style={{
                        background: '#333',
                        color: '#ffa500',
                        border: '1px solid #666',
                        padding: '6px',
                        fontSize: '8px',
                        cursor: 'pointer'
                      }}>
                        HEDGE PORTFOLIO
                      </button>
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                      <button style={{
                        background: '#ff0000',
                        color: '#fff',
                        border: 'none',
                        padding: '6px',
                        fontSize: '8px',
                        fontWeight: 'bold',
                        cursor: 'pointer'
                      }}>
                        EMERGENCY STOP
                      </button>
                      <button style={{
                        background: '#333',
                        color: '#00ff00',
                        border: '1px solid #666',
                        padding: '6px',
                        fontSize: '8px',
                        cursor: 'pointer'
                      }}>
                        RESUME TRADING
                      </button>
                    </div>
                  </div>

                  {/* Live Execution Feed */}
                  <div style={{ background: '#1a1a1a', border: '1px solid #333', padding: '8px', flex: 1 }}>
                    <div style={{ color: '#ffa500', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px' }}>
                      EXECUTION FEED
                    </div>
                    
                    <div style={{ maxHeight: '200px', overflow: 'auto' }}>
                      {executionFeed.map((execution, index) => (
                        <div key={index} style={{
                          background: '#0a0a0a',
                          border: '1px solid #222',
                          padding: '4px',
                          marginBottom: '2px',
                          fontSize: '8px'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1px' }}>
                            <span style={{ color: '#00ffff' }}>{execution.symbol}</span>
                            <span style={{ color: '#666' }}>{execution.time}</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{
                              color: execution.side === 'BUY' ? '#00ff00' : '#ff0000'
                            }}>
                              {execution.side} {execution.quantity} @ ${execution.price}
                            </span>
                            <span style={{ color: '#ffa500' }}>{execution.status}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Interactive Command Line */}
        <InteractiveCommandLine 
          marketData={marketData} 
          agents={agents} 
          setAgents={setAgents}
          webSocketConnected={isConnected}
        />
      </div>
    </>
  );
}

export default App;
