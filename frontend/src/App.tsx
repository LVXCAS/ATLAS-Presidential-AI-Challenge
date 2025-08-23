import React, { useState } from 'react';
import styled, { ThemeProvider, createGlobalStyle } from 'styled-components';
import { Provider } from 'react-redux';
import { store } from './store';
import BloombergTheme, { GlobalStyles } from './themes/bloomberg';
import BloombergTerminal from './components/BloombergTerminal';
import LiveTradingDashboard from './components/LiveTradingDashboard';
import { getWebSocketService } from './services/WebSocketService';

// Global styles injection
const GlobalStylesComponent = createGlobalStyle`
  ${GlobalStyles}
`;

const AppContainer = styled.div`
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background-color: ${props => props.theme.colors.background};
  font-family: ${props => props.theme.typography.fontFamily};
`;

// Initialize WebSocket service
const webSocketService = getWebSocketService({
  url: process.env.REACT_APP_WS_URL || 'ws://localhost:8001',
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectInterval: 5000,
  pingInterval: 30000
});

function App() {
  const [currentView, setCurrentView] = useState('terminal');

  return (
    <Provider store={store}>
      <ThemeProvider theme={BloombergTheme}>
        <GlobalStylesComponent />
        <AppContainer>
          {/* View Toggle */}
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            zIndex: 1000,
            display: 'flex',
            gap: '10px'
          }}>
            <button
              onClick={() => setCurrentView('terminal')}
              style={{
                backgroundColor: currentView === 'terminal' ? '#FFA500' : '#333',
                color: currentView === 'terminal' ? '#000' : '#FFA500',
                border: '1px solid #FFA500',
                padding: '8px 16px',
                fontFamily: 'monospace',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              BLOOMBERG TERMINAL
            </button>
            <button
              onClick={() => setCurrentView('live')}
              style={{
                backgroundColor: currentView === 'live' ? '#FFA500' : '#333',
                color: currentView === 'live' ? '#000' : '#FFA500',
                border: '1px solid #FFA500',
                padding: '8px 16px',
                fontFamily: 'monospace',
                fontSize: '12px',
                cursor: 'pointer'
              }}
            >
              LIVE DASHBOARD
            </button>
          </div>

          {/* Content */}
          {currentView === 'terminal' ? (
            <BloombergTerminal />
          ) : (
            <LiveTradingDashboard />
          )}
        </AppContainer>
      </ThemeProvider>
    </Provider>
  );
}

export default App;