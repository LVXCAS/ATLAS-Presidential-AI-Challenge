import React from 'react';
import styled from 'styled-components';

const ChartContainer = styled.div`
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.tertiary};
  font-family: ${props => props.theme.typography.fontFamily};
`;

interface ChartPanelProps {
  symbol?: string;
  timeframe?: string;
  panelId: string;
}

const ChartPanel: React.FC<ChartPanelProps> = ({ symbol = 'SPY', timeframe = '1m' }) => {
  return (
    <ChartContainer>
      CHART: {symbol} ({timeframe})
      <br />
      [TradingView Chart Integration]
    </ChartContainer>
  );
};

export default ChartPanel;