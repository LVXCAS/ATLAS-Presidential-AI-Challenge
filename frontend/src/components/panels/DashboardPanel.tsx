/**
 * Dashboard Panel Component
 * Executive dashboard with key metrics and system overview
 */

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import { formatCurrency, formatPercent, formatTime } from '../../utils/formatters';

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const DashboardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  background-color: ${props => props.theme.colors.surface};
  flex-shrink: 0;
`;

const TimeDisplay = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.tertiary};
`;

const StatusIndicators = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  align-items: center;
`;

const StatusItem = styled.div<{ $status: 'healthy' | 'warning' | 'critical' }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  font-size: ${props => props.theme.typography.fontSize.xs};
  
  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: ${props => {
      switch (props.$status) {
        case 'critical': return props.theme.colors.negative;
        case 'warning': return props.theme.colors.warning;
        default: return props.theme.colors.positive;
      }
    }};
  }
`;

const DashboardContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${props => props.theme.spacing.sm};
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto 1fr;
  gap: ${props => props.theme.spacing.md};
  
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

const MetricsSection = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${props => props.theme.spacing.sm};
`;

const MetricCard = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.md};
  display: flex;
  flex-direction: column;
  justify-content: space-between;
`;

const MetricLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xs};
  text-transform: uppercase;
`;

const MetricValue = styled.div<{ $type?: 'positive' | 'negative' | 'neutral' }>`
  font-size: ${props => props.theme.typography.fontSize.xl};
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

const SystemSection = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.md};
`;

const SectionTitle = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.tertiary};
  margin-bottom: ${props => props.theme.spacing.md};
  text-transform: uppercase;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  padding-bottom: ${props => props.theme.spacing.xs};
`;

const SystemMetric = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.xs} 0;
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const ActivitySection = styled.div`
  grid-column: 1 / -1;
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.md};
  display: flex;
  flex-direction: column;
`;

const ActivityList = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const ActivityItem = styled.div<{ $type: 'trade' | 'alert' | 'system' }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.xs} 0;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  font-size: ${props => props.theme.typography.fontSize.xs};
  
  ${props => {
    const colors = {
      trade: props.theme.colors.tertiary,
      alert: props.theme.colors.warning,
      system: props.theme.colors.positive
    };
    return `border-left: 2px solid ${colors[props.$type]};`;
  }}
  
  padding-left: ${props => props.theme.spacing.sm};
  
  &:last-child {
    border-bottom: none;
  }
`;

const QuickStats = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.md};
  grid-column: 1 / -1;
`;

const StatCard = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.sm};
  text-align: center;
`;

const StatValue = styled.div<{ $color?: string }>`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.$color || props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const StatLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  text-transform: uppercase;
`;

interface DashboardPanelProps {
  panelId: string;
}

interface SystemStatus {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  services: {
    trading: boolean;
    data: boolean;
    risk: boolean;
  };
}

interface TradingMetrics {
  totalPnL: number;
  dailyPnL: number;
  positions: number;
  orders: number;
  winRate: number;
  var: number;
}

const DashboardPanel: React.FC<DashboardPanelProps> = ({ panelId }) => {
  const [currentTime, setCurrentTime] = useState<Date>(new Date());
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [tradingMetrics, setTradingMetrics] = useState<TradingMetrics | null>(null);
  const [recentActivity, setRecentActivity] = useState<any[]>([]);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Initialize dashboard data
  useEffect(() => {
    const mockSystemStatus: SystemStatus = {
      cpu: 34.5,
      memory: 67.2,
      disk: 23.8,
      network: 125.6,
      services: {
        trading: true,
        data: true,
        risk: true
      }
    };

    const mockTradingMetrics: TradingMetrics = {
      totalPnL: 15420.75,
      dailyPnL: 847.25,
      positions: 12,
      orders: 3,
      winRate: 68.5,
      var: 12450.00
    };

    const mockActivity = [
      { type: 'trade', message: 'AAPL: +250 shares @ $150.25', time: Date.now() - 120000 },
      { type: 'alert', message: 'High volatility detected in NVDA', time: Date.now() - 300000 },
      { type: 'system', message: 'Risk monitoring system updated', time: Date.now() - 450000 },
      { type: 'trade', message: 'SPY: -100 shares @ $425.80', time: Date.now() - 600000 },
      { type: 'alert', message: 'Position limit reached for TSLA', time: Date.now() - 900000 }
    ];

    setSystemStatus(mockSystemStatus);
    setTradingMetrics(mockTradingMetrics);
    setRecentActivity(mockActivity);
  }, []);

  const getSystemHealth = (): 'healthy' | 'warning' | 'critical' => {
    if (!systemStatus) return 'healthy';
    
    const { cpu, memory, services } = systemStatus;
    const allServicesUp = Object.values(services).every(s => s);
    
    if (!allServicesUp || cpu > 80 || memory > 90) return 'critical';
    if (cpu > 60 || memory > 75) return 'warning';
    return 'healthy';
  };

  const formatActivityTime = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Now';
  };

  if (!systemStatus || !tradingMetrics) {
    return (
      <DashboardContainer>
        <DashboardContent>
          <div>Loading dashboard...</div>
        </DashboardContent>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <DashboardHeader>
        <TimeDisplay>
          {currentTime.toLocaleTimeString()} EST
        </TimeDisplay>
        
        <StatusIndicators>
          <StatusItem $status={getSystemHealth()}>
            SYSTEM
          </StatusItem>
          <StatusItem $status={systemStatus.services.trading ? 'healthy' : 'critical'}>
            TRADING
          </StatusItem>
          <StatusItem $status={systemStatus.services.data ? 'healthy' : 'critical'}>
            DATA
          </StatusItem>
          <StatusItem $status={systemStatus.services.risk ? 'healthy' : 'critical'}>
            RISK
          </StatusItem>
        </StatusIndicators>
      </DashboardHeader>

      <DashboardContent>
        <QuickStats>
          <StatCard>
            <StatValue $color="#4CAF50">
              {tradingMetrics.positions}
            </StatValue>
            <StatLabel>POSITIONS</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue $color="#FF9800">
              {tradingMetrics.orders}
            </StatValue>
            <StatLabel>ORDERS</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue $color={tradingMetrics.dailyPnL >= 0 ? '#4CAF50' : '#FF5E4D'}>
              {formatCurrency(tradingMetrics.dailyPnL)}
            </StatValue>
            <StatLabel>DAILY P&L</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue $color={tradingMetrics.winRate >= 50 ? '#4CAF50' : '#FF5E4D'}>
              {formatPercent(tradingMetrics.winRate)}
            </StatValue>
            <StatLabel>WIN RATE</StatLabel>
          </StatCard>
        </QuickStats>

        <MetricsSection>
          <MetricCard>
            <MetricLabel>Total P&L</MetricLabel>
            <MetricValue $type={tradingMetrics.totalPnL >= 0 ? 'positive' : 'negative'}>
              {formatCurrency(tradingMetrics.totalPnL)}
            </MetricValue>
            <MetricChange $positive={tradingMetrics.totalPnL >= 0}>
              {formatPercent(3.84)} Today
            </MetricChange>
          </MetricCard>

          <MetricCard>
            <MetricLabel>Value at Risk</MetricLabel>
            <MetricValue $type="negative">
              {formatCurrency(tradingMetrics.var)}
            </MetricValue>
            <MetricChange $positive={false}>
              95% Confidence
            </MetricChange>
          </MetricCard>
        </MetricsSection>

        <SystemSection>
          <SectionTitle>System Resources</SectionTitle>
          <SystemMetric>
            <span>CPU Usage</span>
            <span style={{ color: systemStatus.cpu > 70 ? '#FF5E4D' : '#4CAF50' }}>
              {systemStatus.cpu.toFixed(1)}%
            </span>
          </SystemMetric>
          <SystemMetric>
            <span>Memory Usage</span>
            <span style={{ color: systemStatus.memory > 80 ? '#FF5E4D' : '#4CAF50' }}>
              {systemStatus.memory.toFixed(1)}%
            </span>
          </SystemMetric>
          <SystemMetric>
            <span>Disk Usage</span>
            <span style={{ color: systemStatus.disk > 85 ? '#FF5E4D' : '#4CAF50' }}>
              {systemStatus.disk.toFixed(1)}%
            </span>
          </SystemMetric>
          <SystemMetric>
            <span>Network I/O</span>
            <span style={{ color: '#4CAF50' }}>
              {systemStatus.network.toFixed(1)} MB/s
            </span>
          </SystemMetric>
        </SystemSection>

        <ActivitySection>
          <SectionTitle>Recent Activity</SectionTitle>
          <ActivityList>
            {recentActivity.map((activity, index) => (
              <ActivityItem key={index} $type={activity.type}>
                <span>{activity.message}</span>
                <span>{formatActivityTime(activity.time)}</span>
              </ActivityItem>
            ))}
          </ActivityList>
        </ActivitySection>
      </DashboardContent>
    </DashboardContainer>
  );
};

export default DashboardPanel;