import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { Activity, TrendingUp, Brain, Shield, Database, Zap, Settings, AlertTriangle, CheckCircle, Clock, DollarSign, Target, Users, Cpu, ArrowUp, ArrowDown } from 'lucide-react';

interface PortfolioData {
  time: string;
  portfolio: number;
  benchmark: number;
  volume: number;
}

interface Position {
  symbol: string;
  qty: number;
  price: number;
  mktVal: number;
  pnl: number;
  pnlPct: number;
  side: 'LONG' | 'SHORT';
}

interface AgentSignal {
  agent: string;
  signal: 'BUY' | 'SELL' | 'HOLD' | 'NEUTRAL';
  strength: number;
  symbol: string;
  price: number;
  size: number;
  confidence: number;
  timestamp: string;
}

interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  chg: number;
  chgPct: number;
  vol: string;
  high: number;
  low: number;
}

interface SystemStat {
  label: string;
  value: number | string;
  format: 'currency' | 'percent' | 'number' | 'ms';
}

interface RiskMetric {
  metric: string;
  value: number;
  pct?: number;
  limit: number;
  status: 'OK' | 'WARNING' | 'ALERT';
}

const LiveTradingDashboard: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedPage, setSelectedPage] = useState('MAIN');
  const [blinkingElements, setBlinkingElements] = useState<Record<string, boolean>>({});
  const [isConnected, setIsConnected] = useState(false);
  
  // Live data state
  const [portfolioData, setPortfolioData] = useState<PortfolioData[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [agentSignals, setAgentSignals] = useState<AgentSignal[]>([]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStat[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetric[]>([]);
  const [totalPnL, setTotalPnL] = useState(0);
  const [dayPnL, setDayPnL] = useState(0);
  const [portfolioValue, setPortfolioValue] = useState(100000);
  
  // Live connection to backend
  useEffect(() => {
    const connectToBackend = async () => {
      try {
        // Try to connect to the live trading backend
        const response = await fetch('http://localhost:8001/api/dashboard/status');
        if (response.ok) {
          setIsConnected(true);
          fetchLiveData();
        } else {
          // Fallback to simulated data if backend not available
          setIsConnected(false);
          startSimulatedData();
        }
      } catch (error) {
        console.log('Backend not available, using simulated data');
        setIsConnected(false);
        startSimulatedData();
      }
    };

    connectToBackend();
  }, []);

  // Fetch live data from backend
  const fetchLiveData = async () => {
    try {
      const [portfolioRes, positionsRes, signalsRes, marketRes, statsRes, riskRes] = await Promise.all([
        fetch('http://localhost:8001/api/dashboard/portfolio'),
        fetch('http://localhost:8001/api/dashboard/positions'),
        fetch('http://localhost:8001/api/dashboard/signals'),
        fetch('http://localhost:8001/api/dashboard/market'),
        fetch('http://localhost:8001/api/dashboard/stats'),
        fetch('http://localhost:8001/api/dashboard/risk')
      ]);

      if (portfolioRes.ok) {
        const data = await portfolioRes.json();
        setPortfolioData(data.performance || []);
        setPortfolioValue(data.totalValue || 100000);
        setTotalPnL(data.totalPnL || 0);
        setDayPnL(data.dayPnL || 0);
      }

      if (positionsRes.ok) {
        const data = await positionsRes.json();
        setPositions(data.positions || []);
      }

      if (signalsRes.ok) {
        const data = await signalsRes.json();
        setAgentSignals(data.signals || []);
      }

      if (marketRes.ok) {
        const data = await marketRes.json();
        setMarketData(data.market || []);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setSystemStats(data.stats || []);
      }

      if (riskRes.ok) {
        const data = await riskRes.json();
        setRiskMetrics(data.metrics || []);
      }

    } catch (error) {
      console.error('Error fetching live data:', error);
    }
  };

  // Simulated live data when backend is not available
  const startSimulatedData = () => {
    // Initialize with base data
    const basePortfolioValue = 100000;
    let currentPnL = 0;
    
    const simulateRealTimeData = () => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString().slice(0, 5);
      
      // Simulate portfolio changes
      const pnlChange = (Math.random() - 0.5) * 100; // Random P&L change
      currentPnL += pnlChange;
      const newPortfolioValue = basePortfolioValue + currentPnL;
      
      setPortfolioValue(newPortfolioValue);
      setTotalPnL(currentPnL);
      setDayPnL(currentPnL * 0.3); // Day P&L as portion of total
      
      // Update portfolio performance data
      setPortfolioData(prev => {
        const newData = [...prev.slice(-20), {
          time: timeStr,
          portfolio: newPortfolioValue,
          benchmark: basePortfolioValue + (currentPnL * 0.4), // Benchmark performance
          volume: Math.random() * 3 + 1
        }];
        return newData;
      });

      // Update positions with live P&L
      setPositions([
        { symbol: 'AAPL', qty: 1250, price: 178.25 + (Math.random() - 0.5) * 2, mktVal: 222812, pnl: 8745 + Math.random() * 1000, pnlPct: 4.08 + Math.random() * 2, side: 'LONG' },
        { symbol: 'MSFT', qty: 850, price: 342.18 + (Math.random() - 0.5) * 3, mktVal: 290853, pnl: -2134 + Math.random() * 2000, pnlPct: -0.73 + Math.random() * 1.5, side: 'LONG' },
        { symbol: 'GOOGL', qty: 450, price: 128.76 + (Math.random() - 0.5) * 1.5, mktVal: 57942, pnl: 1876 + Math.random() * 500, pnlPct: 3.34 + Math.random() * 1, side: 'LONG' },
        { symbol: 'TSLA', qty: -200, price: 245.80 + (Math.random() - 0.5) * 5, mktVal: -49160, pnl: 3421 + Math.random() * 800, pnlPct: 7.48 + Math.random() * 2, side: 'SHORT' },
        { symbol: 'SPY', qty: 2100, price: 432.50 + (Math.random() - 0.5) * 2, mktVal: 908250, pnl: 12456 + Math.random() * 2000, pnlPct: 1.39 + Math.random() * 1, side: 'LONG' }
      ]);

      // Update agent signals
      const agentNames = ['MOMENTUM_01', 'SENTIMENT_02', 'MEAN_REV_03', 'NEWS_NLP_04', 'RISK_MGR_05', 'ARBIT_06'];
      const symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA'];
      const signals = ['BUY', 'SELL', 'HOLD'] as const;
      
      setAgentSignals(agentNames.map(agent => {
        const signal = signals[Math.floor(Math.random() * signals.length)];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        return {
          agent,
          signal,
          strength: Math.random() * 0.4 + 0.6, // 0.6 to 1.0
          symbol,
          price: 100 + Math.random() * 400,
          size: signal === 'HOLD' ? 0 : Math.floor(Math.random() * 500) + 100,
          confidence: Math.random() * 0.3 + 0.7, // 0.7 to 1.0
          timestamp: now.toISOString()
        };
      }));

      // Update market data
      setMarketData([
        { symbol: 'SPY', bid: 432.48, ask: 432.52, last: 432.50 + (Math.random() - 0.5) * 2, chg: Math.random() * 10 - 5, chgPct: (Math.random() - 0.5) * 3, vol: `${(Math.random() * 50 + 20).toFixed(1)}M`, high: 434.12, low: 428.34 },
        { symbol: 'QQQ', bid: 368.72, ask: 368.78, last: 368.75 + (Math.random() - 0.5) * 3, chg: Math.random() * 8 - 4, chgPct: (Math.random() - 0.5) * 2.5, vol: `${(Math.random() * 40 + 15).toFixed(1)}M`, high: 371.45, low: 367.23 },
        { symbol: 'IWM', bid: 198.45, ask: 198.51, last: 198.48 + (Math.random() - 0.5) * 1.5, chg: Math.random() * 4 - 2, chgPct: (Math.random() - 0.5) * 2, vol: `${(Math.random() * 35 + 10).toFixed(1)}M`, high: 199.23, low: 196.78 },
        { symbol: 'VIX', bid: 16.23, ask: 16.28, last: 16.25 + (Math.random() - 0.5) * 1, chg: Math.random() * 2 - 1, chgPct: (Math.random() - 0.5) * 5, vol: `${(Math.random() * 25 + 5).toFixed(1)}M`, high: 17.12, low: 15.89 }
      ]);

      // Update system stats
      setSystemStats([
        { label: 'TOTAL_PNL', value: currentPnL, format: 'currency' },
        { label: 'DAY_PNL', value: currentPnL * 0.3, format: 'currency' },
        { label: 'POSITIONS', value: 5 + Math.floor(Math.random() * 8), format: 'number' },
        { label: 'EXEC_LAT', value: 1.5 + Math.random() * 2, format: 'ms' },
        { label: 'SYS_UPTIME', value: 99.95 + Math.random() * 0.04, format: 'percent' },
        { label: 'DATA_FEEDS', value: 8, format: 'number' }
      ]);

      // Update risk metrics
      setRiskMetrics([
        { metric: 'PORT_VAR_95', value: -23450.67 + Math.random() * 5000, limit: -50000, status: 'OK' },
        { metric: 'MAX_DD', value: -41234.89 + Math.random() * 8000, limit: -100000, status: 'OK' },
        { metric: 'SHARPE_RTD', value: 2.84 + (Math.random() - 0.5) * 0.5, limit: 2.0, status: 'OK' },
        { metric: 'BETA_SPX', value: 0.67 + (Math.random() - 0.5) * 0.2, limit: 1.0, status: 'OK' },
        { metric: 'GAMMA_EXP', value: 12456.78 + Math.random() * 2000, limit: 50000, status: 'OK' },
        { metric: 'THETA_DECAY', value: -234.56 - Math.random() * 50, limit: -1000, status: 'OK' }
      ]);
    };

    // Update every 3 seconds for live feel
    const interval = setInterval(simulateRealTimeData, 3000);
    simulateRealTimeData(); // Initial call
    
    return () => clearInterval(interval);
  };

  // Real-time updates when connected to backend
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(fetchLiveData, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, [isConnected]);

  // Clock update
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Bloomberg-style blinking for updates
  useEffect(() => {
    const blinkTimer = setInterval(() => {
      setBlinkingElements(prev => ({
        ...prev,
        [`element_${Math.floor(Math.random() * 20)}`]: true
      }));
      setTimeout(() => {
        setBlinkingElements({});
      }, 200);
    }, 3000);
    return () => clearInterval(blinkTimer);
  }, []);

  const formatValue = (value: number | string, format: string) => {
    if (typeof value === 'string') return value;
    
    switch (format) {
      case 'currency':
        return `${value >= 0 ? '+' : ''}$${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      case 'percent':
        return `${value.toFixed(2)}%`;
      case 'ms':
        return `${value.toFixed(2)}ms`;
      default:
        return value.toString();
    }
  };

  const getChangeColor = (value: number) => {
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-300';
  };

  const getMarketStatus = () => {
    const now = new Date();
    const day = now.getDay(); // 0 = Sunday, 6 = Saturday
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const totalMinutes = hours * 60 + minutes;
    
    // Convert to EST (assuming local time)
    const marketOpen = 9 * 60 + 30; // 9:30 AM
    const marketClose = 16 * 60; // 4:00 PM
    const extendedOpen = 4 * 60; // 4:00 AM
    const extendedClose = 20 * 60; // 8:00 PM
    
    if (day === 0 || day === 6) {
      return { status: 'CLOSED', color: 'text-red-400', message: 'Weekend' };
    }
    
    if (totalMinutes >= marketOpen && totalMinutes < marketClose) {
      return { status: 'OPEN', color: 'text-green-400', message: 'Regular Hours' };
    } else if (totalMinutes >= extendedOpen && totalMinutes < extendedClose) {
      return { status: 'EXTENDED', color: 'text-yellow-400', message: 'Extended Hours' };
    } else {
      return { status: 'CLOSED', color: 'text-red-400', message: 'After Hours' };
    }
  };

  const marketStatus = getMarketStatus();

  return (
    <div className="min-h-screen bg-black text-orange-400 font-mono text-xs overflow-hidden">
      
      {/* Bloomberg-style Header */}
      <div className="bg-orange-500 text-black px-4 py-1 flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="font-bold text-sm">HIVE TRADE</div>
          <div className="flex items-center space-x-4">
            <button 
              className={`px-2 py-1 ${selectedPage === 'MAIN' ? 'bg-black text-orange-400' : ''}`}
              onClick={() => setSelectedPage('MAIN')}
            >
              MAIN
            </button>
            <button 
              className={`px-2 py-1 ${selectedPage === 'PORT' ? 'bg-black text-orange-400' : ''}`}
              onClick={() => setSelectedPage('PORT')}
            >
              PORT
            </button>
            <button 
              className={`px-2 py-1 ${selectedPage === 'RISK' ? 'bg-black text-orange-400' : ''}`}
              onClick={() => setSelectedPage('RISK')}
            >
              RISK
            </button>
            <button 
              className={`px-2 py-1 ${selectedPage === 'AGENTS' ? 'bg-black text-orange-400' : ''}`}
              onClick={() => setSelectedPage('AGENTS')}
            >
              AGENTS
            </button>
          </div>
        </div>
        <div className="flex items-center space-x-6">
          <span className={`${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            {isConnected ? 'LIVE' : 'SIM'}
          </span>
          <span>AI-TRADING-SYS v2.1.3</span>
          <span>{currentTime.toLocaleTimeString()}</span>
          <span>{currentTime.toLocaleDateString()}</span>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-gray-900 text-orange-400 px-4 py-1 text-xs">
        <div className="flex items-center space-x-8">
          <span>STATUS: <span className={`${marketStatus.color} animate-pulse`}>{marketStatus.status}</span></span>
          <span>PORTFOLIO: <span className="text-white">${portfolioValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span></span>
          <span>DAY_PNL: <span className={getChangeColor(dayPnL)}>{formatValue(dayPnL, 'currency')}</span></span>
          <span>POSITIONS: <span className="text-white">{positions.length}</span></span>
          <span>AGENTS: <span className="text-green-400">{agentSignals.filter(s => s.signal !== 'HOLD').length}/6 ACTIVE</span></span>
          <span>LATENCY: <span className="text-yellow-400">{systemStats.find(s => s.label === 'EXEC_LAT')?.value || 0}ms</span></span>
          <span>MARKET: <span className={marketStatus.color}>{marketStatus.message}</span></span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="p-2 grid grid-cols-12 gap-2" style={{ height: 'calc(100vh - 120px)' }}>
        
        {/* Left Panel - Market Data */}
        <div className="col-span-3 space-y-2 overflow-y-auto">
          
          {/* Market Overview */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              MARKET OVERVIEW
            </div>
            <div className="p-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left">SYM</th>
                    <th className="text-right">LAST</th>
                    <th className="text-right">CHG</th>
                    <th className="text-right">VOL</th>
                  </tr>
                </thead>
                <tbody>
                  {marketData.map((item, idx) => (
                    <tr key={idx} className={`${blinkingElements[`market_${idx}`] ? 'bg-yellow-900' : ''}`}>
                      <td className="text-white font-bold">{item.symbol}</td>
                      <td className="text-right text-white">{item.last.toFixed(2)}</td>
                      <td className={`text-right ${getChangeColor(item.chg)}`}>
                        {item.chg >= 0 ? '+' : ''}{item.chg.toFixed(2)}
                      </td>
                      <td className="text-right text-gray-400">{item.vol}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* System Statistics */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              SYSTEM STATS
            </div>
            <div className="p-2 space-y-1">
              {systemStats.map((stat, idx) => (
                <div key={idx} className="flex justify-between">
                  <span className="text-gray-400">{stat.label}</span>
                  <span className={`text-white ${stat.label.includes('PNL') && typeof stat.value === 'number' && stat.value > 0 ? 'text-green-400' : ''}`}>
                    {formatValue(stat.value, stat.format)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Agent Signals */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              AGENT SIGNALS
            </div>
            <div className="p-2 max-h-96 overflow-y-auto">
              {agentSignals.map((signal, idx) => (
                <div key={idx} className={`mb-2 p-1 border border-gray-700 ${
                  signal.signal === 'BUY' ? 'bg-green-900' : 
                  signal.signal === 'SELL' ? 'bg-red-900' : 'bg-gray-800'
                }`}>
                  <div className="flex justify-between">
                    <span className="text-white font-bold">{signal.agent}</span>
                    <span className={`font-bold ${
                      signal.signal === 'BUY' ? 'text-green-400' : 
                      signal.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {signal.signal}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>{signal.symbol}</span>
                    <span>STR: {(signal.strength * 100).toFixed(0)}%</span>
                  </div>
                  <div className="text-xs text-gray-400">
                    SIZE: {signal.size} @ ${signal.price.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">
                    CONF: {(signal.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center Panel - Charts */}
        <div className="col-span-6 space-y-2">
          
          {/* Main Performance Chart */}
          <div className="border border-gray-700 bg-gray-900" style={{ height: '60%' }}>
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              PORTFOLIO PERFORMANCE - LIVE INTRADAY
            </div>
            <div className="p-2" style={{ height: 'calc(100% - 40px)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={portfolioData}>
                  <CartesianGrid strokeDasharray="1 1" stroke="#374151" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#FFA500" 
                    fontSize={10}
                    tickFormatter={(value) => value}
                  />
                  <YAxis 
                    stroke="#FFA500" 
                    fontSize={10}
                    domain={['dataMin - 1000', 'dataMax + 1000']}
                    tickFormatter={(value) => `${(value/1000).toFixed(0)}K`}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#000000', 
                      border: '1px solid #FFA500',
                      color: '#FFA500',
                      fontSize: '11px'
                    }}
                    labelStyle={{ color: '#FFFFFF' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="portfolio" 
                    stroke="#FFA500" 
                    fill="url(#portfolioGradient)" 
                    strokeWidth={2} 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="benchmark" 
                    stroke="#666666" 
                    fill="none" 
                    strokeWidth={1} 
                    strokeDasharray="2 2" 
                  />
                  <defs>
                    <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#FFA500" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#FFA500" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Portfolio Positions */}
          <div className="border border-gray-700 bg-gray-900" style={{ height: '38%' }}>
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              CURRENT POSITIONS - LIVE P&L
            </div>
            <div className="p-2 overflow-y-auto" style={{ height: 'calc(100% - 40px)' }}>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left">SYMBOL</th>
                    <th className="text-right">QTY</th>
                    <th className="text-right">PRICE</th>
                    <th className="text-right">MKT_VAL</th>
                    <th className="text-right">PNL</th>
                    <th className="text-right">PNL%</th>
                    <th className="text-center">SIDE</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos, idx) => (
                    <tr key={idx} className={`${blinkingElements[`pos_${idx}`] ? 'bg-yellow-900' : ''}`}>
                      <td className="text-white font-bold">{pos.symbol}</td>
                      <td className="text-right text-white">{pos.qty.toLocaleString()}</td>
                      <td className="text-right text-white">${pos.price.toFixed(2)}</td>
                      <td className="text-right text-white">${pos.mktVal.toLocaleString()}</td>
                      <td className={`text-right ${getChangeColor(pos.pnl)}`}>
                        {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toLocaleString()}
                      </td>
                      <td className={`text-right ${getChangeColor(pos.pnl)}`}>
                        {pos.pnlPct >= 0 ? '+' : ''}{pos.pnlPct.toFixed(2)}%
                      </td>
                      <td className={`text-center ${pos.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                        {pos.side}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Panel - Risk & Agents */}
        <div className="col-span-3 space-y-2 overflow-y-auto">
          
          {/* Risk Management */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              RISK MANAGEMENT - LIVE
            </div>
            <div className="p-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left">METRIC</th>
                    <th className="text-right">VALUE</th>
                    <th className="text-right">LIMIT</th>
                  </tr>
                </thead>
                <tbody>
                  {riskMetrics.map((metric, idx) => (
                    <tr key={idx}>
                      <td className="text-gray-300 text-xs">{metric.metric}</td>
                      <td className={`text-right ${
                        metric.value < 0 ? 'text-red-400' : 
                        metric.value > 2 ? 'text-green-400' : 'text-white'
                      }`}>
                        {typeof metric.value === 'number' ? 
                          (Math.abs(metric.value) > 100 ? 
                            metric.value.toLocaleString() : 
                            metric.value.toFixed(2)
                          ) : metric.value
                        }
                      </td>
                      <td className="text-right text-gray-400">
                        {metric.limit.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Agent Status */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              AGENT STATUS - LIVE
            </div>
            <div className="p-2 space-y-1">
              {[
                { name: 'MASTER_AGENT', status: 'ACTIVE', conf: 87, lat: 1.2 + Math.random() },
                { name: 'MOMENTUM_01', status: 'ACTIVE', conf: 72 + Math.floor(Math.random() * 20), lat: 2.1 + Math.random() },
                { name: 'SENTIMENT_02', status: 'ACTIVE', conf: 68 + Math.floor(Math.random() * 25), lat: 1.8 + Math.random() },
                { name: 'RISK_MGR', status: 'MONITOR', conf: 95 + Math.floor(Math.random() * 5), lat: 0.9 + Math.random() * 0.5 },
                { name: 'DATA_FEED', status: 'ACTIVE', conf: 99, lat: 0.3 + Math.random() * 0.2 },
                { name: 'NLP_AGENT', status: isConnected ? 'ACTIVE' : 'SIM', conf: 83 + Math.floor(Math.random() * 15), lat: 15.2 + Math.random() * 10 }
              ].map((agent, idx) => (
                <div key={idx} className="flex justify-between items-center text-xs border-b border-gray-700 py-1">
                  <span className="text-white">{agent.name}</span>
                  <div className="flex items-center space-x-2">
                    <span className={`${
                      agent.status === 'ACTIVE' ? 'text-green-400' :
                      agent.status === 'MONITOR' ? 'text-yellow-400' :
                      agent.status === 'SIM' ? 'text-blue-400' : 'text-red-400'
                    }`}>
                      {agent.status}
                    </span>
                    <span className="text-gray-400">{agent.conf}%</span>
                    <span className="text-gray-500">{agent.lat.toFixed(1)}ms</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Live Trading Activity */}
          <div className="border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 px-2 py-1 text-orange-400 font-bold border-b border-gray-700">
              LIVE TRADING ACTIVITY
            </div>
            <div className="p-2">
              <div className="text-center mb-2">
                <div className="text-green-400 font-bold">{isConnected ? 'LIVE TRADING' : 'SIMULATION MODE'}</div>
                <div className="text-xs text-gray-400">MARKET: {marketStatus.message}</div>
              </div>
              <div className="space-y-1">
                {[
                  { activity: 'ORDER_EXEC', status: positions.length > 0 ? 'ACTIVE' : 'IDLE' },
                  { activity: 'RISK_CHECK', status: 'ACTIVE' },
                  { activity: 'DATA_FEED', status: 'ACTIVE' },
                  { activity: 'P&L_CALC', status: 'ACTIVE' },
                  { activity: 'SIGNAL_GEN', status: agentSignals.filter(s => s.signal !== 'HOLD').length > 0 ? 'ACTIVE' : 'IDLE' }
                ].map((item, idx) => (
                  <div key={idx} className="flex justify-between text-xs">
                    <span className="text-gray-300">{item.activity}</span>
                    <span className={`${
                      item.status === 'ACTIVE' ? 'text-green-400' : 'text-gray-400'
                    }`}>
                      {item.status}
                    </span>
                  </div>
                ))}
              </div>
              
              {/* Recent Trades */}
              <div className="mt-3 pt-2 border-t border-gray-700">
                <div className="text-xs text-gray-400 mb-1">RECENT ACTIVITY:</div>
                <div className="text-xs space-y-1">
                  {positions.slice(0, 3).map((pos, idx) => (
                    <div key={idx} className="text-gray-300">
                      {currentTime.toLocaleTimeString().slice(0, 5)} | {pos.side} {pos.qty} {pos.symbol} @ ${pos.price.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div className="bg-orange-500 text-black px-4 py-1 text-xs flex justify-between">
        <div>© 2024 HIVE TRADE | LIVE AI TRADING SYSTEM | {isConnected ? 'CONNECTED TO ALPACA' : 'SIMULATION MODE'}</div>
        <div>INSTITUTIONAL GRADE | PAPER TRADING: $200K BUYING POWER</div>
      </div>
    </div>
  );
};

export default LiveTradingDashboard;