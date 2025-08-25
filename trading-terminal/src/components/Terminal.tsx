import { useEffect } from 'react';
import styled from 'styled-components';
import { useTradingStore } from '../stores/tradingStore';
import Header from './Header';
import MarketDataPanel from './MarketDataPanel';
import PositionsPanel from './PositionsPanel';
import ChartPanel from './ChartPanel';
import NewsPanel from './NewsPanel';
import RiskDashboard from './RiskDashboard';
import AgentMatrix from './AgentMatrix';
import OrderPanel from './OrderPanel';
import CommandLine from './CommandLine';

const TerminalContainer = styled.div`
  background: #000000;
  color: #00ff00;
  font-family: 'Courier New', 'Consolas', monospace;
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const MainGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: repeat(3, 1fr);
  gap: 1px;
  background: #333;
  flex: 1;
  padding: 1px;
`;

const Panel = styled.div`
  background: #0a0a0a;
  border: 1px solid #222;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const Terminal: React.FC = () => {
  const initWebSocket = useTradingStore((state) => state.initWebSocket);
  const setMarketData = useTradingStore((state) => state.setMarketData);

  useEffect(() => {
    initWebSocket();
    
    // Initialize with sample market data
    const sampleData = {
      'SPY': {
        symbol: 'SPY',
        price: 445.32,
        change: 2.15,
        change_percent: 0.48,
        volume: 15234567,
        high: 447.21,
        low: 443.15,
        open: 444.10,
        vwap: 445.67,
        timestamp: new Date().toISOString()
      },
      'AAPL': {
        symbol: 'AAPL',
        price: 178.25,
        change: -1.35,
        change_percent: -0.75,
        volume: 8456789,
        high: 179.85,
        low: 177.90,
        open: 179.30,
        vwap: 178.52,
        timestamp: new Date().toISOString()
      },
      'TSLA': {
        symbol: 'TSLA',
        price: 234.67,
        change: 8.42,
        change_percent: 3.72,
        volume: 12345678,
        high: 236.50,
        low: 228.75,
        open: 230.25,
        vwap: 233.15,
        timestamp: new Date().toISOString()
      },
      'NVDA': {
        symbol: 'NVDA',
        price: 456.89,
        change: 12.34,
        change_percent: 2.78,
        volume: 9876543,
        high: 461.25,
        low: 448.30,
        open: 452.15,
        vwap: 454.72,
        timestamp: new Date().toISOString()
      },
      'MSFT': {
        symbol: 'MSFT',
        price: 378.45,
        change: -2.87,
        change_percent: -0.75,
        volume: 6543210,
        high: 382.10,
        low: 376.25,
        open: 380.75,
        vwap: 379.18,
        timestamp: new Date().toISOString()
      }
    };
    
    setMarketData(sampleData);
  }, [initWebSocket, setMarketData]);

  return (
    <TerminalContainer>
      <Header />
      
      <MainGrid>
        {/* Row 1 */}
        <Panel style={{ gridColumn: 'span 2' }}>
          <MarketDataPanel />
        </Panel>
        <Panel style={{ gridColumn: 'span 2', gridRow: 'span 2' }}>
          <ChartPanel />
        </Panel>
        
        {/* Row 2 */}
        <Panel>
          <PositionsPanel />
        </Panel>
        <Panel>
          <NewsPanel />
        </Panel>
        
        {/* Row 3 */}
        <Panel style={{ gridColumn: 'span 2' }}>
          <RiskDashboard />
        </Panel>
        <Panel>
          <AgentMatrix />
        </Panel>
        <Panel>
          <OrderPanel />
        </Panel>
      </MainGrid>
      
      <CommandLine />
    </TerminalContainer>
  );
};

export default Terminal;