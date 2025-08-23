/**
 * Bloomberg Terminal Theme
 * Exact color scheme and typography matching Bloomberg Terminal
 */

export const BloombergTheme = {
  colors: {
    // Primary colors
    background: '#000000',
    surface: '#001122',
    surfaceLight: '#002244',
    
    // Text colors
    primary: '#00FF00',      // Bright green for primary text
    secondary: '#FFFF00',    // Yellow for secondary text
    tertiary: '#FFA500',     // Orange for headers and important info
    white: '#FFFFFF',
    gray: '#888888',
    darkGray: '#444444',
    
    // Status colors
    positive: '#00FF00',     // Green for gains
    negative: '#FF0000',     // Red for losses
    neutral: '#FFFF00',      // Yellow for neutral
    warning: '#FFA500',      // Orange for warnings
    critical: '#FF0000',     // Red for critical alerts
    
    // Grid and borders
    gridLine: '#333333',
    border: '#444444',
    divider: '#222222',
    
    // Chart colors
    chartBackground: '#000000',
    chartGrid: '#1a1a1a',
    chartText: '#00FF00',
    
    // Technical indicator colors
    sma20: '#00FFFF',        // Cyan
    sma50: '#FF00FF',        // Magenta  
    sma200: '#FFFF00',       // Yellow
    rsi: '#FFA500',          // Orange
    macd: '#00FF00',         // Green
    
    // Order colors
    buy: '#00FF00',          // Green for buy orders
    sell: '#FF0000',         // Red for sell orders
    pending: '#FFFF00',      // Yellow for pending orders
  },
  
  typography: {
    fontFamily: '"Courier New", "Consolas", "Monaco", monospace',
    fontSize: {
      xs: '10px',
      sm: '11px',
      md: '12px',
      lg: '14px',
      xl: '16px',
      xxl: '18px'
    },
    fontWeight: {
      normal: 400,
      bold: 700
    },
    lineHeight: {
      tight: 1.1,
      normal: 1.3,
      loose: 1.5
    }
  },
  
  spacing: {
    xs: '2px',
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    xxl: '24px'
  },
  
  layout: {
    headerHeight: '32px',
    footerHeight: '24px',
    panelMinWidth: '200px',
    panelMinHeight: '150px',
    gridGap: '2px'
  },
  
  animation: {
    // Bloomberg uses minimal animations for performance
    fast: '0.1s ease-in-out',
    normal: '0.2s ease-in-out',
    slow: '0.3s ease-in-out',
    
    // Price flashing effects
    priceFlash: 'flash 0.3s ease-in-out',
    uptick: 'uptick 0.5s ease-out',
    downtick: 'downtick 0.5s ease-out'
  },
  
  shadows: {
    none: 'none',
    sm: '0 1px 2px rgba(0, 255, 0, 0.1)',
    md: '0 2px 4px rgba(0, 255, 0, 0.1)',
    lg: '0 4px 8px rgba(0, 255, 0, 0.1)'
  },
  
  // Panel-specific styling
  panels: {
    marketData: {
      headerBg: '#001122',
      headerText: '#FFA500',
      rowEven: '#000000',
      rowOdd: '#001111'
    },
    
    orderBook: {
      bidColor: '#00FF00',
      askColor: '#FF0000',
      sizeColor: '#FFFF00'
    },
    
    positions: {
      profitColor: '#00FF00',
      lossColor: '#FF0000',
      breakEvenColor: '#FFFF00'
    },
    
    news: {
      titleColor: '#FFA500',
      timestampColor: '#888888',
      bodyColor: '#CCCCCC'
    }
  },
  
  // Command line styling
  commandLine: {
    background: '#000000',
    text: '#00FF00',
    cursor: '#00FF00',
    prompt: '#FFA500',
    error: '#FF0000',
    success: '#00FF00'
  }
};

// CSS-in-JS styled-components theme
export type BloombergThemeType = typeof BloombergTheme;

// Global styles for Bloomberg Terminal
export const GlobalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: ${BloombergTheme.typography.fontFamily};
    font-size: ${BloombergTheme.typography.fontSize.sm};
    background-color: ${BloombergTheme.colors.background};
    color: ${BloombergTheme.colors.primary};
    overflow: hidden;
    user-select: none;
    -webkit-font-smoothing: none;
    -moz-osx-font-smoothing: none;
    text-rendering: optimizeSpeed;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${BloombergTheme.colors.background};
  }

  ::-webkit-scrollbar-thumb {
    background: ${BloombergTheme.colors.gridLine};
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${BloombergTheme.colors.border};
  }

  /* Animations */
  @keyframes flash {
    0% { background-color: rgba(0, 255, 0, 0.3); }
    50% { background-color: rgba(0, 255, 0, 0.1); }
    100% { background-color: transparent; }
  }

  @keyframes uptick {
    0% { color: ${BloombergTheme.colors.positive}; background-color: rgba(0, 255, 0, 0.3); }
    100% { color: ${BloombergTheme.colors.positive}; background-color: transparent; }
  }

  @keyframes downtick {
    0% { color: ${BloombergTheme.colors.negative}; background-color: rgba(255, 0, 0, 0.3); }
    100% { color: ${BloombergTheme.colors.negative}; background-color: transparent; }
  }

  /* Bloomberg-style focus indicators */
  button:focus,
  input:focus,
  select:focus,
  textarea:focus {
    outline: 2px solid ${BloombergTheme.colors.tertiary};
    outline-offset: 1px;
  }

  /* Terminal-style selection */
  ::selection {
    background-color: rgba(0, 255, 0, 0.3);
    color: ${BloombergTheme.colors.background};
  }

  /* Disable text selection on UI elements */
  .no-select {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }

  /* Bloomberg-style table headers */
  .bloomberg-header {
    background-color: ${BloombergTheme.colors.surface};
    color: ${BloombergTheme.colors.tertiary};
    font-weight: ${BloombergTheme.typography.fontWeight.bold};
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: ${BloombergTheme.spacing.sm} ${BloombergTheme.spacing.md};
    border-bottom: 1px solid ${BloombergTheme.colors.border};
  }

  /* Bloomberg-style data rows */
  .bloomberg-row {
    border-bottom: 1px solid ${BloombergTheme.colors.divider};
    font-family: ${BloombergTheme.typography.fontFamily};
    font-size: ${BloombergTheme.typography.fontSize.sm};
  }

  .bloomberg-row:hover {
    background-color: rgba(0, 255, 0, 0.05);
  }

  /* Price change animations */
  .price-up {
    animation: ${BloombergTheme.animation.uptick};
    color: ${BloombergTheme.colors.positive};
  }

  .price-down {
    animation: ${BloombergTheme.animation.downtick};
    color: ${BloombergTheme.colors.negative};
  }

  .price-flash {
    animation: ${BloombergTheme.animation.priceFlash};
  }

  /* Status indicators */
  .status-positive {
    color: ${BloombergTheme.colors.positive};
  }

  .status-negative {
    color: ${BloombergTheme.colors.negative};
  }

  .status-neutral {
    color: ${BloombergTheme.colors.neutral};
  }

  .status-warning {
    color: ${BloombergTheme.colors.warning};
  }

  .status-critical {
    color: ${BloombergTheme.colors.critical};
    animation: ${BloombergTheme.animation.priceFlash} infinite;
  }

  /* Bloomberg-style buttons */
  .bloomberg-button {
    background-color: ${BloombergTheme.colors.surface};
    color: ${BloombergTheme.colors.primary};
    border: 1px solid ${BloombergTheme.colors.border};
    font-family: ${BloombergTheme.typography.fontFamily};
    font-size: ${BloombergTheme.typography.fontSize.sm};
    padding: ${BloombergTheme.spacing.sm} ${BloombergTheme.spacing.md};
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all ${BloombergTheme.animation.fast};
  }

  .bloomberg-button:hover {
    background-color: ${BloombergTheme.colors.surfaceLight};
    border-color: ${BloombergTheme.colors.tertiary};
  }

  .bloomberg-button:active {
    background-color: ${BloombergTheme.colors.tertiary};
    color: ${BloombergTheme.colors.background};
  }

  /* Bloomberg-style inputs */
  .bloomberg-input {
    background-color: ${BloombergTheme.colors.background};
    color: ${BloombergTheme.colors.primary};
    border: 1px solid ${BloombergTheme.colors.border};
    font-family: ${BloombergTheme.typography.fontFamily};
    font-size: ${BloombergTheme.typography.fontSize.sm};
    padding: ${BloombergTheme.spacing.sm} ${BloombergTheme.spacing.md};
    text-transform: uppercase;
  }

  .bloomberg-input:focus {
    border-color: ${BloombergTheme.colors.tertiary};
    background-color: rgba(0, 255, 0, 0.05);
  }

  /* Panel styling */
  .bloomberg-panel {
    background-color: ${BloombergTheme.colors.background};
    border: 1px solid ${BloombergTheme.colors.border};
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .bloomberg-panel-header {
    background-color: ${BloombergTheme.colors.surface};
    color: ${BloombergTheme.colors.tertiary};
    padding: ${BloombergTheme.spacing.sm} ${BloombergTheme.spacing.md};
    font-weight: ${BloombergTheme.typography.fontWeight.bold};
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid ${BloombergTheme.colors.border};
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .bloomberg-panel-content {
    flex: 1;
    overflow: auto;
    padding: ${BloombergTheme.spacing.sm};
  }

  /* Grid system */
  .bloomberg-grid {
    display: grid;
    gap: ${BloombergTheme.layout.gridGap};
    height: 100vh;
    padding: ${BloombergTheme.layout.gridGap};
    background-color: ${BloombergTheme.colors.background};
  }

  /* Responsive grid layouts */
  .grid-4x4 {
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: ${BloombergTheme.layout.headerHeight} repeat(3, 1fr) ${BloombergTheme.layout.footerHeight};
  }

  .grid-3x3 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: ${BloombergTheme.layout.headerHeight} repeat(2, 1fr) ${BloombergTheme.layout.footerHeight};
  }

  /* Loading indicators */
  .bloomberg-loading {
    color: ${BloombergTheme.colors.tertiary};
    font-family: ${BloombergTheme.typography.fontFamily};
    animation: ${BloombergTheme.animation.priceFlash} infinite;
  }

  /* Connection status */
  .connection-status {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: ${BloombergTheme.spacing.sm};
  }

  .connection-connected {
    background-color: ${BloombergTheme.colors.positive};
  }

  .connection-disconnected {
    background-color: ${BloombergTheme.colors.negative};
  }

  .connection-connecting {
    background-color: ${BloombergTheme.colors.warning};
    animation: ${BloombergTheme.animation.priceFlash} infinite;
  }
`;

export default BloombergTheme;