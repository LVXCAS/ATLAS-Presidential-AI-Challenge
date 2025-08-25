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

const RiskGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  flex: 1;
`;

const RiskMetric = styled.div`
  border: 1px solid #333;
  padding: 8px;
  text-align: center;
`;

const MetricLabel = styled.div`
  color: #ffa500;
  font-size: 9px;
  margin-bottom: 4px;
`;

const MetricValue = styled.div<{ $warning?: boolean }>`
  color: ${props => props.$warning ? '#ff0000' : '#00ff00'};
  font-size: 12px;
  font-weight: bold;
`;

const RiskDashboard: React.FC = () => {
  const riskMetrics = useTradingStore((state) => state.riskMetrics);

  const mockRiskMetrics = {
    portfolio_var: 0.025,
    portfolio_cvar: 0.032,
    max_drawdown: 0.085,
    sharpe_ratio: 1.45,
    sortino_ratio: 1.82,
    concentration_risk: 0.15
  };

  const metrics = riskMetrics || mockRiskMetrics;

  return (
    <PanelContainer>
      <PanelTitle>Risk Management Dashboard</PanelTitle>
      <RiskGrid>
        <RiskMetric>
          <MetricLabel>VALUE AT RISK</MetricLabel>
          <MetricValue $warning={metrics.portfolio_var > 0.03}>
            {(metrics.portfolio_var * 100).toFixed(2)}%
          </MetricValue>
        </RiskMetric>
        
        <RiskMetric>
          <MetricLabel>MAX DRAWDOWN</MetricLabel>
          <MetricValue $warning={metrics.max_drawdown > 0.1}>
            {(metrics.max_drawdown * 100).toFixed(2)}%
          </MetricValue>
        </RiskMetric>
        
        <RiskMetric>
          <MetricLabel>SHARPE RATIO</MetricLabel>
          <MetricValue $warning={metrics.sharpe_ratio < 1.0}>
            {metrics.sharpe_ratio.toFixed(2)}
          </MetricValue>
        </RiskMetric>
        
        <RiskMetric>
          <MetricLabel>CVAR (95%)</MetricLabel>
          <MetricValue $warning={metrics.portfolio_cvar > 0.04}>
            {(metrics.portfolio_cvar * 100).toFixed(2)}%
          </MetricValue>
        </RiskMetric>
        
        <RiskMetric>
          <MetricLabel>SORTINO RATIO</MetricLabel>
          <MetricValue>
            {metrics.sortino_ratio.toFixed(2)}
          </MetricValue>
        </RiskMetric>
        
        <RiskMetric>
          <MetricLabel>CONCENTRATION</MetricLabel>
          <MetricValue $warning={metrics.concentration_risk > 0.2}>
            {(metrics.concentration_risk * 100).toFixed(1)}%
          </MetricValue>
        </RiskMetric>
      </RiskGrid>
    </PanelContainer>
  );
};

export default RiskDashboard;