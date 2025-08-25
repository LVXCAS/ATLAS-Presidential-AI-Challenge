import React from 'react';
import styled from 'styled-components';
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
`;

const PositionsList = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const PositionRow = styled.div`
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  padding: 4px;
  border-bottom: 1px solid #222;
  font-size: 10px;
`;

const Symbol = styled.div`
  color: #ffa500;
  font-weight: bold;
`;

const PnL = styled.div<{ $positive: boolean }>`
  color: ${props => props.$positive ? '#00ff00' : '#ff0000'};
  text-align: right;
`;

const PositionsPanel: React.FC = () => {
  const positions = useTradingStore((state) => state.positions);

  const mockPositions = [
    { symbol: 'AAPL', quantity: 100, unrealized_pnl: 1250.50, unrealized_pnl_percent: 4.2 },
    { symbol: 'TSLA', quantity: -50, unrealized_pnl: -890.25, unrealized_pnl_percent: -2.1 },
    { symbol: 'NVDA', quantity: 25, unrealized_pnl: 3420.75, unrealized_pnl_percent: 8.9 },
    { symbol: 'SPY', quantity: 200, unrealized_pnl: 567.30, unrealized_pnl_percent: 1.3 }
  ];

  const displayPositions = positions.length > 0 ? positions : mockPositions;

  return (
    <PanelContainer>
      <PanelTitle>Positions</PanelTitle>
      <PositionsList>
        {displayPositions.map((position, index) => (
          <PositionRow key={position.symbol || index}>
            <Symbol>{position.symbol} ({position.quantity})</Symbol>
            <PnL $positive={position.unrealized_pnl >= 0}>
              ${position.unrealized_pnl.toFixed(2)}
            </PnL>
            <PnL $positive={position.unrealized_pnl_percent >= 0}>
              {position.unrealized_pnl_percent.toFixed(1)}%
            </PnL>
          </PositionRow>
        ))}
      </PositionsList>
    </PanelContainer>
  );
};

export default PositionsPanel;