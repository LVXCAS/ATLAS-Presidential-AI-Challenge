/**
 * Bloomberg Terminal Main Component
 * Professional trading interface with grid-based layout
 */

import React, { useEffect, useState, useCallback } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import { getWebSocketService } from '../services/WebSocketService';
import { 
  updatePrice, 
  updateQuote, 
  updateIndicators, 
  setConnectionStatus,
  batchUpdate
} from '../store/slices/marketDataSlice';
import { addNotification, setActivePanel } from '../store/slices/uiSlice';

// Import panel components
import MarketDataPanel from './panels/MarketDataPanel';
import ChartPanel from './panels/ChartPanel';
import OrderBookPanel from './panels/OrderBookPanel';
import PositionsPanel from './panels/PositionsPanel';
import OrdersPanel from './panels/OrdersPanel';
import RiskDashboardPanel from './panels/RiskDashboardPanel';
import SystemStatusPanel from './panels/SystemStatusPanel';
import CommandLinePanel from './panels/CommandLinePanel';
import NewsPanel from './panels/NewsPanel';
import AnalyticsPanel from './panels/AnalyticsPanel';
import AlertsPanel from './panels/AlertsPanel';
import MonitoringPanel from './panels/MonitoringPanel';
import DashboardPanel from './panels/DashboardPanel';
import SettingsPanel from './panels/SettingsPanel';
import PanelSelector from './PanelSelector';

const TerminalContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  grid-template-rows: 32px repeat(5, 1fr) 160px;
  gap: 2px;
  height: 100vh;
  width: 100vw;
  padding: 2px;
  background-color: ${props => props.theme.colors.background};
  overflow: hidden;
`;

const PanelContainer = styled.div<{ 
  $x: number; 
  $y: number; 
  $width: number; 
  $height: number;
  $isActive: boolean;
  $isMinimized: boolean;
  $isMaximized: boolean;
}>`
  grid-column: ${props => `${props.$x + 1} / span ${props.$width}`};
  grid-row: ${props => `${props.$y + 1} / span ${props.$height}`};
  background-color: ${props => props.theme.colors.background};
  border: 1px solid ${props => 
    props.$isActive ? props.theme.colors.tertiary : props.theme.colors.border
  };
  display: ${props => props.$isMinimized ? 'none' : 'flex'};
  flex-direction: column;
  overflow: hidden;
  transition: border-color ${props => props.theme.animation.fast};
  
  ${props => props.$isMaximized && `
    position: fixed;
    top: 2px;
    left: 2px;
    right: 2px;
    bottom: 2px;
    z-index: 1000;
    grid-column: unset;
    grid-row: unset;
  `}
`;

const PanelHeader = styled.div<{ $type: string }>`
  background-color: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.tertiary};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
  cursor: default;
`;

const PanelContent = styled.div`
  flex: 1;
  overflow: hidden;
  position: relative;
`;

const PanelControls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
`;

const ControlButton = styled.button`
  background: none;
  border: none;
  color: ${props => props.theme.colors.primary};
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  padding: 2px 4px;
  opacity: 0.7;
  transition: opacity ${props => props.theme.animation.fast};

  &:hover {
    opacity: 1;
    color: ${props => props.theme.colors.tertiary};
  }
`;

const ConnectionIndicator = styled.div<{ $status: 'connected' | 'disconnected' | 'connecting' }>`
  position: fixed;
  top: 4px;
  right: 4px;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.primary};
  z-index: 2000;
  
  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: ${props => {
      switch (props.$status) {
        case 'connected': return props.theme.colors.positive;
        case 'disconnected': return props.theme.colors.negative;
        case 'connecting': return props.theme.colors.warning;
        default: return props.theme.colors.gray;
      }
    }};
    animation: ${props => props.$status === 'connecting' ? `${props.theme.animation.priceFlash} infinite` : 'none'};
  }
`;

const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.theme.colors.tertiary};
  font-family: ${props => props.theme.typography.fontFamily};
  z-index: 100;
  
  &::after {
    content: 'LOADING...';
    animation: ${props => props.theme.animation.priceFlash} infinite;
  }
`;

interface PanelComponent {
  [key: string]: React.ComponentType<any>;
}

const panelComponents: PanelComponent = {
  'system-status': SystemStatusPanel,
  'watchlist': MarketDataPanel,
  'chart': ChartPanel,
  'order-book': OrderBookPanel,
  'positions': PositionsPanel,
  'orders': OrdersPanel,
  'risk-dashboard': RiskDashboardPanel,
  'news': NewsPanel,
  'analytics': AnalyticsPanel,
  'alerts': AlertsPanel,
  'monitoring': MonitoringPanel,
  'dashboard': DashboardPanel,
  'settings': SettingsPanel,
};

const BloombergTerminal: React.FC = () => {
  const dispatch = useDispatch();
  const { layout, activePanel } = useSelector((state: RootState) => state.ui);
  const { connectionStatus } = useSelector((state: RootState) => state.marketData);
  const [isLoading, setIsLoading] = useState(true);

  // WebSocket connection management
  useEffect(() => {
    const wsService = getWebSocketService();

    // Connection handlers
    const handleConnection = (connected: boolean) => {
      dispatch(setConnectionStatus(connected ? 'connected' : 'disconnected'));
      
      if (connected) {
        dispatch(addNotification({
          type: 'success',
          title: 'CONNECTION ESTABLISHED',
          message: 'Real-time data streaming active'
        }));
        
        // Subscribe to default symbols
        const defaultSymbols = ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN'];
        wsService.subscribeToSymbols(defaultSymbols);
      } else {
        dispatch(addNotification({
          type: 'error',
          title: 'CONNECTION LOST',
          message: 'Attempting to reconnect...',
          persistent: true
        }));
      }
    };

    // Market data handlers
    const handleMarketData = (message: any) => {
      if (message.type === 'market_data' && message.data) {
        dispatch(updatePrice(message.data));
      }
    };

    const handleQuoteData = (message: any) => {
      if (message.type === 'quote_data' && message.data) {
        dispatch(updateQuote(message.data));
      }
    };

    const handleIndicatorData = (message: any) => {
      if (message.type === 'indicators' && message.data) {
        dispatch(updateIndicators(message.data));
      }
    };

    const handleSystemStatus = (message: any) => {
      if (message.type === 'system_status') {
        // Handle system status updates
        console.log('System status update:', message.data);
      }
    };

    const handleRiskAlert = (message: any) => {
      if (message.type === 'risk_alert') {
        dispatch(addNotification({
          type: 'error',
          title: 'RISK ALERT',
          message: message.data.description || 'Risk threshold breached',
          persistent: true
        }));
      }
    };

    // Register handlers
    wsService.addConnectionHandler(handleConnection);
    wsService.addMessageHandler('market_data', handleMarketData);
    wsService.addMessageHandler('quote_data', handleQuoteData);
    wsService.addMessageHandler('indicators', handleIndicatorData);
    wsService.addMessageHandler('system_status', handleSystemStatus);
    wsService.addMessageHandler('risk_alert', handleRiskAlert);

    // Set initial connection status
    dispatch(setConnectionStatus(wsService.connected ? 'connected' : 'connecting'));

    // Loading timeout
    const loadingTimer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    // Cleanup
    return () => {
      clearTimeout(loadingTimer);
      wsService.removeConnectionHandler(handleConnection);
      wsService.removeMessageHandler('market_data', handleMarketData);
      wsService.removeMessageHandler('quote_data', handleQuoteData);
      wsService.removeMessageHandler('indicators', handleIndicatorData);
      wsService.removeMessageHandler('system_status', handleSystemStatus);
      wsService.removeMessageHandler('risk_alert', handleRiskAlert);
    };
  }, [dispatch]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const { ctrlKey, key } = event;

      if (ctrlKey) {
        switch (key.toLowerCase()) {
          case 'l':
            event.preventDefault();
            // Toggle command line (handled by CommandLinePanel)
            break;
          case 'm':
            event.preventDefault();
            if (activePanel) {
              // Maximize active panel
            }
            break;
          case 'r':
            event.preventDefault();
            // Refresh data
            window.location.reload();
            break;
        }
      }

      if (key === 'Escape') {
        // Cancel current action or close overlays
        dispatch(setActivePanel(null));
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [activePanel, dispatch]);

  // Panel click handler
  const handlePanelClick = useCallback((panelId: string) => {
    dispatch(setActivePanel(panelId));
  }, [dispatch]);

  // Panel control handlers
  const handleMinimizePanel = useCallback((panelId: string) => {
    // Implementation for minimizing panels
    console.log('Minimize panel:', panelId);
  }, []);

  const handleMaximizePanel = useCallback((panelId: string) => {
    // Implementation for maximizing panels
    console.log('Maximize panel:', panelId);
  }, []);

  const handleClosePanel = useCallback((panelId: string) => {
    // Implementation for closing panels
    console.log('Close panel:', panelId);
  }, []);

  const renderPanel = (panel: any) => {
    const PanelComponent = panelComponents[panel.type];
    
    if (!PanelComponent) {
      return (
        <div style={{ 
          padding: '10px', 
          color: '#FF0000',
          fontFamily: 'monospace',
          fontSize: '11px'
        }}>
          PANEL TYPE '{panel.type}' NOT IMPLEMENTED
        </div>
      );
    }

    return <PanelComponent {...panel.configuration} panelId={panel.id} />;
  };

  const getPanelTitle = (panelType: string): string => {
    const titles: Record<string, string> = {
      'system-status': 'SYSTEM STATUS',
      'watchlist': 'MARKET WATCH',
      'chart': 'TECHNICAL CHART',
      'order-book': 'ORDER BOOK',
      'positions': 'POSITIONS',
      'orders': 'ORDERS',
      'risk-dashboard': 'RISK MONITOR',
      'news': 'NEWS FEED',
      'analytics': 'ANALYTICS',
      'alerts': 'ALERTS',
      'monitoring': 'MONITORING',
      'dashboard': 'DASHBOARD',
      'settings': 'SETTINGS'
    };
    return titles[panelType] || panelType.toUpperCase().replace('-', ' ');
  };

  return (
    <>
      <ConnectionIndicator $status={connectionStatus}>
        {connectionStatus.toUpperCase()}
      </ConnectionIndicator>

      <PanelSelector />

      <TerminalContainer>
        {layout.panels.map((panel) => (
          <PanelContainer
            key={panel.id}
            $x={panel.position.x}
            $y={panel.position.y}
            $width={panel.position.width}
            $height={panel.position.height}
            $isActive={activePanel === panel.id}
            $isMinimized={panel.minimized}
            $isMaximized={panel.maximized}
            onClick={() => handlePanelClick(panel.id)}
          >
            <PanelHeader $type={panel.type}>
              <span>{getPanelTitle(panel.type)}</span>
              <PanelControls>
                <ControlButton 
                  onClick={(e) => {
                    e.stopPropagation();
                    handleMinimizePanel(panel.id);
                  }}
                  title="Minimize"
                >
                  _
                </ControlButton>
                <ControlButton 
                  onClick={(e) => {
                    e.stopPropagation();
                    handleMaximizePanel(panel.id);
                  }}
                  title="Maximize"
                >
                  □
                </ControlButton>
                <ControlButton 
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClosePanel(panel.id);
                  }}
                  title="Close"
                >
                  ×
                </ControlButton>
              </PanelControls>
            </PanelHeader>
            
            <PanelContent>
              {isLoading && <LoadingOverlay />}
              {renderPanel(panel)}
            </PanelContent>
          </PanelContainer>
        ))}

        {/* Command Line at the bottom */}
        <PanelContainer
          $x={0}
          $y={6}
          $width={6}
          $height={1}
          $isActive={false}
          $isMinimized={false}
          $isMaximized={false}
        >
          <CommandLinePanel />
        </PanelContainer>
      </TerminalContainer>
    </>
  );
};

export default BloombergTerminal;