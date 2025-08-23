/**
 * Analytics Panel Component
 * Advanced trading analytics with performance metrics and insights
 */

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import { formatCurrency, formatPercent, formatTime } from '../../utils/formatters';

const AnalyticsContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const AnalyticsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  background-color: ${props => props.theme.colors.surface};
  flex-shrink: 0;
`;

const TabButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
`;

const TabButton = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? props.theme.colors.tertiary : 'transparent'};
  color: ${props => props.$active ? props.theme.colors.background : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 12px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.tertiary};
    color: ${props => props.theme.colors.background};
  }
`;

const AnalyticsContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${props => props.theme.spacing.sm};
  
  &::-webkit-scrollbar {
    width: 4px;
  }
  
  &::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.background};
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.border};
    border-radius: 2px;
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const MetricCard = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.md};
  display: flex;
  flex-direction: column;
`;

const MetricLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xs};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const MetricValue = styled.div<{ $type?: 'positive' | 'negative' | 'neutral' }>`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => {
    switch (props.$type) {
      case 'positive': return props.theme.colors.positive;
      case 'negative': return props.theme.colors.negative;
      default: return props.theme.colors.primary;
    }
  }};
`;

const MetricChange = styled.div<{ $positive: boolean }>`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.$positive ? props.theme.colors.positive : props.theme.colors.negative};
  margin-top: ${props => props.theme.spacing.xs};
`;

const ChartContainer = styled.div`
  height: 200px;
  background-color: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  margin-bottom: ${props => props.theme.spacing.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.theme.colors.textSecondary};
`;

const TableContainer = styled.div`
  border: 1px solid ${props => props.theme.colors.border};
  overflow: hidden;
`;

const TableHeader = styled.div`
  display: grid;
  grid-template-columns: 1fr 80px 80px 80px 100px;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm};
  background-color: ${props => props.theme.colors.surface};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.textSecondary};
  text-transform: uppercase;
`;

const TableRow = styled.div`
  display: grid;
  grid-template-columns: 1fr 80px 80px 80px 100px;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  font-size: ${props => props.theme.typography.fontSize.xs};
  
  &:last-child {
    border-bottom: none;
  }
  
  &:hover {
    background-color: ${props => props.theme.colors.surface};
  }
`;

const StatusIndicator = styled.div<{ $status: 'active' | 'inactive' | 'warning' }>`
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => {
    switch (props.$status) {
      case 'active': return props.theme.colors.positive;
      case 'warning': return props.theme.colors.warning;
      default: return props.theme.colors.textSecondary;
    }
  }};
`;

interface AnalyticsPanelProps {
  panelId: string;
  timeframe?: '1D' | '1W' | '1M' | '3M' | '1Y';
}

interface PerformanceMetrics {
  totalPnL: number;
  totalPnLPercent: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  avgTradeSize: number;
  avgHoldTime: number;
}

interface StrategyPerformance {
  name: string;
  pnl: number;
  pnlPercent: number;
  trades: number;
  winRate: number;
  status: 'active' | 'inactive' | 'warning';
}

const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({
  panelId,
  timeframe = '1D'
}) => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [strategies, setStrategies] = useState<StrategyPerformance[]>([]);

  // Generate mock analytics data
  useEffect(() => {
    const mockMetrics: PerformanceMetrics = {
      totalPnL: 15420.75,
      totalPnLPercent: 3.84,
      dailyPnL: 847.25,
      dailyPnLPercent: 0.21,
      winRate: 68.5,
      sharpeRatio: 1.42,
      maxDrawdown: -2.1,
      totalTrades: 147,
      avgTradeSize: 2875.50,
      avgHoldTime: 2.4
    };

    const mockStrategies: StrategyPerformance[] = [
      {
        name: 'Mean Reversion',
        pnl: 5420.75,
        pnlPercent: 2.85,
        trades: 52,
        winRate: 72.3,
        status: 'active'
      },
      {
        name: 'Momentum Follow',
        pnl: 8750.25,
        pnlPercent: 4.12,
        trades: 38,
        winRate: 65.8,
        status: 'active'
      },
      {
        name: 'Arbitrage',
        pnl: 1249.75,
        pnlPercent: 1.84,
        trades: 57,
        winRate: 89.5,
        status: 'active'
      },
      {
        name: 'News Sentiment',
        pnl: -247.50,
        pnlPercent: -0.68,
        trades: 12,
        winRate: 41.7,
        status: 'warning'
      }
    ];

    setMetrics(mockMetrics);
    setStrategies(mockStrategies);
  }, [timeframe]);

  const tabs = [
    { key: 'overview', label: 'OVERVIEW' },
    { key: 'performance', label: 'PERFORMANCE' },
    { key: 'risk', label: 'RISK' },
    { key: 'strategies', label: 'STRATEGIES' }
  ];

  const renderOverview = () => (
    <>
      <MetricsGrid>
        <MetricCard>
          <MetricLabel>Total P&L</MetricLabel>
          <MetricValue $type={metrics!.totalPnL >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics!.totalPnL)}
          </MetricValue>
          <MetricChange $positive={metrics!.totalPnLPercent >= 0}>
            {formatPercent(metrics!.totalPnLPercent)}
          </MetricChange>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Daily P&L</MetricLabel>
          <MetricValue $type={metrics!.dailyPnL >= 0 ? 'positive' : 'negative'}>
            {formatCurrency(metrics!.dailyPnL)}
          </MetricValue>
          <MetricChange $positive={metrics!.dailyPnLPercent >= 0}>
            {formatPercent(metrics!.dailyPnLPercent)}
          </MetricChange>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Win Rate</MetricLabel>
          <MetricValue $type={metrics!.winRate >= 50 ? 'positive' : 'negative'}>
            {formatPercent(metrics!.winRate)}
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Sharpe Ratio</MetricLabel>
          <MetricValue $type={metrics!.sharpeRatio >= 1 ? 'positive' : 'negative'}>
            {metrics!.sharpeRatio.toFixed(2)}
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Max Drawdown</MetricLabel>
          <MetricValue $type="negative">
            {formatPercent(metrics!.maxDrawdown)}
          </MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Total Trades</MetricLabel>
          <MetricValue>
            {metrics!.totalTrades.toLocaleString()}
          </MetricValue>
        </MetricCard>
      </MetricsGrid>

      <ChartContainer>
        P&L CURVE CHART (TradingView Integration)
      </ChartContainer>
    </>
  );

  const renderStrategies = () => (
    <TableContainer>
      <TableHeader>
        <div>Strategy</div>
        <div>P&L</div>
        <div>P&L %</div>
        <div>Trades</div>
        <div>Win Rate</div>
      </TableHeader>
      
      {strategies.map((strategy, index) => (
        <TableRow key={index}>
          <div>
            <StatusIndicator $status={strategy.status} style={{ marginRight: '8px' }} />
            {strategy.name}
          </div>
          <div style={{ color: strategy.pnl >= 0 ? '#4CAF50' : '#FF5E4D' }}>
            {formatCurrency(strategy.pnl)}
          </div>
          <div style={{ color: strategy.pnlPercent >= 0 ? '#4CAF50' : '#FF5E4D' }}>
            {formatPercent(strategy.pnlPercent)}
          </div>
          <div>{strategy.trades}</div>
          <div>{formatPercent(strategy.winRate)}</div>
        </TableRow>
      ))}
    </TableContainer>
  );

  const renderContent = () => {
    if (!metrics) return <div>Loading analytics...</div>;

    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'strategies':
        return renderStrategies();
      case 'performance':
        return (
          <ChartContainer>
            PERFORMANCE ANALYSIS CHARTS
          </ChartContainer>
        );
      case 'risk':
        return (
          <ChartContainer>
            RISK METRICS AND VAR ANALYSIS
          </ChartContainer>
        );
      default:
        return renderOverview();
    }
  };

  return (
    <AnalyticsContainer>
      <AnalyticsHeader>
        <TabButtons>
          {tabs.map(tab => (
            <TabButton
              key={tab.key}
              $active={activeTab === tab.key}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </TabButton>
          ))}
        </TabButtons>
      </AnalyticsHeader>

      <AnalyticsContent>
        {renderContent()}
      </AnalyticsContent>
    </AnalyticsContainer>
  );
};

export default AnalyticsPanel;