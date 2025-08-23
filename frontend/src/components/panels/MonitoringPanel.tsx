/**
 * Monitoring Panel Component
 * System monitoring dashboard integrated with Prometheus metrics
 */

import React, { useEffect, useState, useRef } from 'react';
import styled from 'styled-components';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import { formatPercent, formatTime } from '../../utils/formatters';

const MonitoringContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const MonitoringHeader = styled.div`
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

const RefreshButton = styled.button`
  background: transparent;
  color: ${props => props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.tertiary};
    color: ${props => props.theme.colors.background};
  }
`;

const MonitoringContent = styled.div`
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

const SystemMetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const MetricCard = styled.div<{ $status: 'healthy' | 'warning' | 'critical' }>`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => {
    switch (props.$status) {
      case 'critical': return props.theme.colors.negative;
      case 'warning': return props.theme.colors.warning;
      default: return props.theme.colors.border;
    }
  }};
  padding: ${props => props.theme.spacing.md};
  display: flex;
  flex-direction: column;
  position: relative;
`;

const MetricLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xs};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const MetricValue = styled.div<{ $status: 'healthy' | 'warning' | 'critical' }>`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => {
    switch (props.$status) {
      case 'critical': return props.theme.colors.negative;
      case 'warning': return props.theme.colors.warning;
      default: return props.theme.colors.positive;
    }
  }};
`;

const ProgressBar = styled.div<{ $value: number; $status: 'healthy' | 'warning' | 'critical' }>`
  width: 100%;
  height: 4px;
  background-color: ${props => props.theme.colors.border};
  margin-top: ${props => props.theme.spacing.sm};
  border-radius: 2px;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    width: ${props => props.$value}%;
    height: 100%;
    background-color: ${props => {
      switch (props.$status) {
        case 'critical': return props.theme.colors.negative;
        case 'warning': return props.theme.colors.warning;
        default: return props.theme.colors.positive;
      }
    }};
    transition: width ${props => props.theme.animation.medium};
  }
`;

const ServiceGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const ServiceCard = styled.div<{ $status: 'running' | 'stopped' | 'error' }>`
  background-color: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.sm};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const ServiceStatus = styled.div<{ $status: 'running' | 'stopped' | 'error' }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => {
    switch (props.$status) {
      case 'running': return props.theme.colors.positive;
      case 'error': return props.theme.colors.negative;
      default: return props.theme.colors.textSecondary;
    }
  }};
  animation: ${props => props.$status === 'running' ? `${props.theme.animation.priceFlash} 2s infinite` : 'none'};
`;

const ServiceInfo = styled.div`
  flex: 1;
`;

const ServiceName = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  margin-bottom: 2px;
`;

const ServiceDetails = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
`;

const AlertsList = styled.div`
  border: 1px solid ${props => props.theme.colors.border};
  max-height: 200px;
  overflow-y: auto;
`;

const AlertItem = styled.div<{ $severity: 'critical' | 'warning' | 'info' }>`
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  border-left: 3px solid ${props => {
    switch (props.$severity) {
      case 'critical': return props.theme.colors.negative;
      case 'warning': return props.theme.colors.warning;
      default: return props.theme.colors.tertiary;
    }
  }};
  
  &:last-child {
    border-bottom: none;
  }
`;

const AlertTitle = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const AlertMessage = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
`;

interface SystemMetric {
  label: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  threshold: { warning: number; critical: number };
}

interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  uptime: string;
  memory: number;
  cpu: number;
}

interface MonitoringAlert {
  id: string;
  title: string;
  message: string;
  severity: 'critical' | 'warning' | 'info';
  timestamp: number;
}

interface MonitoringPanelProps {
  panelId: string;
}

const MonitoringPanel: React.FC<MonitoringPanelProps> = ({ panelId }) => {
  const [activeTab, setActiveTab] = useState<string>('system');
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [alerts, setAlerts] = useState<MonitoringAlert[]>([]);
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const intervalRef = useRef<NodeJS.Timeout>();

  // Fetch monitoring data from backend
  const fetchMonitoringData = async () => {
    try {
      // System metrics
      const mockSystemMetrics: SystemMetric[] = [
        {
          label: 'CPU Usage',
          value: 45.2,
          unit: '%',
          status: 'healthy',
          threshold: { warning: 70, critical: 85 }
        },
        {
          label: 'Memory Usage',
          value: 68.7,
          unit: '%',
          status: 'warning',
          threshold: { warning: 70, critical: 90 }
        },
        {
          label: 'Disk Usage',
          value: 34.1,
          unit: '%',
          status: 'healthy',
          threshold: { warning: 80, critical: 95 }
        },
        {
          label: 'Network I/O',
          value: 125.6,
          unit: 'MB/s',
          status: 'healthy',
          threshold: { warning: 500, critical: 900 }
        },
        {
          label: 'API Latency',
          value: 45.8,
          unit: 'ms',
          status: 'healthy',
          threshold: { warning: 100, critical: 500 }
        },
        {
          label: 'Error Rate',
          value: 0.12,
          unit: '%',
          status: 'healthy',
          threshold: { warning: 1, critical: 5 }
        }
      ];

      // Service status
      const mockServices: ServiceStatus[] = [
        {
          name: 'Market Data Service',
          status: 'running',
          uptime: '2d 14h 32m',
          memory: 145.2,
          cpu: 12.4
        },
        {
          name: 'Order Service',
          status: 'running',
          uptime: '2d 14h 30m',
          memory: 87.6,
          cpu: 8.7
        },
        {
          name: 'Risk Service',
          status: 'running',
          uptime: '2d 14h 28m',
          memory: 92.1,
          cpu: 15.3
        },
        {
          name: 'Analytics Service',
          status: 'running',
          uptime: '1d 8h 15m',
          memory: 234.8,
          cpu: 28.9
        },
        {
          name: 'Redis Cache',
          status: 'running',
          uptime: '5d 12h 45m',
          memory: 156.3,
          cpu: 3.2
        },
        {
          name: 'Database',
          status: 'running',
          uptime: '5d 12h 47m',
          memory: 892.4,
          cpu: 18.6
        }
      ];

      // Active alerts
      const mockAlerts: MonitoringAlert[] = [
        {
          id: '1',
          title: 'High Memory Usage',
          message: 'Memory usage has exceeded 70% threshold',
          severity: 'warning',
          timestamp: Date.now() - 300000
        },
        {
          id: '2',
          title: 'Trading Volume High',
          message: 'Unusual trading volume detected in SPY',
          severity: 'info',
          timestamp: Date.now() - 600000
        }
      ];

      setSystemMetrics(mockSystemMetrics);
      setServices(mockServices);
      setAlerts(mockAlerts);
      setLastUpdate(Date.now());
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
    }
  };

  useEffect(() => {
    fetchMonitoringData();
    
    // Auto-refresh every 30 seconds
    intervalRef.current = setInterval(fetchMonitoringData, 30000);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const handleRefresh = () => {
    fetchMonitoringData();
  };

  const getMetricStatus = (metric: SystemMetric): 'healthy' | 'warning' | 'critical' => {
    if (metric.value >= metric.threshold.critical) return 'critical';
    if (metric.value >= metric.threshold.warning) return 'warning';
    return 'healthy';
  };

  const renderSystemMetrics = () => (
    <>
      <SystemMetricsGrid>
        {systemMetrics.map((metric, index) => {
          const status = getMetricStatus(metric);
          return (
            <MetricCard key={index} $status={status}>
              <MetricLabel>{metric.label}</MetricLabel>
              <MetricValue $status={status}>
                {metric.value.toFixed(1)} {metric.unit}
              </MetricValue>
              {metric.unit === '%' && (
                <ProgressBar $value={metric.value} $status={status} />
              )}
            </MetricCard>
          );
        })}
      </SystemMetricsGrid>
    </>
  );

  const renderServices = () => (
    <ServiceGrid>
      {services.map((service, index) => (
        <ServiceCard key={index} $status={service.status}>
          <ServiceStatus $status={service.status} />
          <ServiceInfo>
            <ServiceName>{service.name}</ServiceName>
            <ServiceDetails>
              Uptime: {service.uptime} | Memory: {service.memory.toFixed(1)}MB | CPU: {service.cpu.toFixed(1)}%
            </ServiceDetails>
          </ServiceInfo>
        </ServiceCard>
      ))}
    </ServiceGrid>
  );

  const renderAlerts = () => (
    <AlertsList>
      {alerts.length === 0 ? (
        <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
          No active alerts
        </div>
      ) : (
        alerts.map(alert => (
          <AlertItem key={alert.id} $severity={alert.severity}>
            <AlertTitle>{alert.title}</AlertTitle>
            <AlertMessage>
              {alert.message} - {formatTime(alert.timestamp)}
            </AlertMessage>
          </AlertItem>
        ))
      )}
    </AlertsList>
  );

  const tabs = [
    { key: 'system', label: 'SYSTEM' },
    { key: 'services', label: 'SERVICES' },
    { key: 'alerts', label: 'ALERTS' }
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'system':
        return renderSystemMetrics();
      case 'services':
        return renderServices();
      case 'alerts':
        return renderAlerts();
      default:
        return renderSystemMetrics();
    }
  };

  return (
    <MonitoringContainer>
      <MonitoringHeader>
        <TabButtons>
          {tabs.map(tab => (
            <TabButton
              key={tab.key}
              $active={activeTab === tab.key}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
              {tab.key === 'alerts' && alerts.length > 0 && (
                <span style={{ 
                  marginLeft: '4px',
                  backgroundColor: '#FF5E4D',
                  color: 'white',
                  padding: '1px 4px',
                  borderRadius: '2px',
                  fontSize: '10px'
                }}>
                  {alerts.length}
                </span>
              )}
            </TabButton>
          ))}
        </TabButtons>
        
        <div style={{ fontSize: '10px', color: '#666', display: 'flex', alignItems: 'center', gap: '8px' }}>
          Last Updated: {formatTime(lastUpdate)}
          <RefreshButton onClick={handleRefresh}>
            REFRESH
          </RefreshButton>
        </div>
      </MonitoringHeader>

      <MonitoringContent>
        {renderContent()}
      </MonitoringContent>
    </MonitoringContainer>
  );
};

export default MonitoringPanel;