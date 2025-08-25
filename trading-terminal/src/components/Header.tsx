import React from 'react';
import styled from 'styled-components';
import { useTradingStore } from '../stores/tradingStore';

const HeaderContainer = styled.div`
  background: #0a0a0a;
  border-bottom: 1px solid #333;
  padding: 8px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 40px;
  font-size: 11px;
`;

const Logo = styled.div`
  color: #ffa500;
  font-weight: bold;
  font-size: 14px;
  letter-spacing: 2px;
`;

const StatusSection = styled.div`
  display: flex;
  gap: 20px;
  align-items: center;
`;

const StatusItem = styled.div<{ $status: 'active' | 'inactive' | 'warning' }>`
  color: ${props => {
    switch (props.$status) {
      case 'active': return '#00ff00';
      case 'inactive': return '#ff0000';
      case 'warning': return '#ffa500';
      default: return '#ffffff';
    }
  }};
  
  &:before {
    content: '●';
    margin-right: 5px;
  }
`;

const MetricItem = styled.div`
  color: #ffa500;
  font-size: 10px;
`;

const Header: React.FC = () => {
  const { connected, systemStatus, portfolioPnL } = useTradingStore();

  return (
    <HeaderContainer>
      <Logo>HIVE TRADE QUANTUM TERMINAL v2.0</Logo>
      
      <StatusSection>
        <StatusItem $status={connected ? 'active' : 'inactive'}>
          CONNECTED
        </StatusItem>
        
        <StatusItem $status={'active'}>
          MARKET OPEN
        </StatusItem>
        
        <StatusItem $status={systemStatus?.agents_active ? 'active' : 'inactive'}>
          {Object.values(systemStatus?.agents_active || {}).filter(Boolean).length} AGENTS
        </StatusItem>
        
        <MetricItem>
          P&L: ${portfolioPnL.toFixed(2)}
        </MetricItem>
        
        <MetricItem>
          LATENCY: {systemStatus?.system_latency || 0}ms
        </MetricItem>
        
        <MetricItem>
          {new Date().toLocaleString()}
        </MetricItem>
      </StatusSection>
    </HeaderContainer>
  );
};

export default Header;