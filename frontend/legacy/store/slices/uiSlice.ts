/**
 * UI State Redux Slice
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { GridLayout, PanelLayout, UserSettings } from '../../types';

interface UIState {
  layout: GridLayout;
  activePanel: string | null;
  commandLineVisible: boolean;
  commandHistory: string[];
  currentCommand: string;
  notifications: Notification[];
  settings: UserSettings;
  fullscreenPanel: string | null;
  sidebarVisible: boolean;
  bottomPanelHeight: number;
  keyboardShortcuts: Record<string, string>;
}

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: number;
  persistent?: boolean;
  actions?: Array<{ label: string; action: string }>;
}

const defaultLayout: GridLayout = {
  columns: 6,
  rows: 6,
  panels: [
    // Top row - System status, alerts, and monitoring
    {
      id: 'system-header',
      type: 'system-status',
      position: { x: 0, y: 0, width: 3, height: 1 },
      minimized: false,
      maximized: false,
      configuration: {}
    },
    {
      id: 'alerts',
      type: 'alerts',
      position: { x: 3, y: 0, width: 2, height: 1 },
      minimized: false,
      maximized: false,
      configuration: {}
    },
    {
      id: 'monitoring',
      type: 'monitoring',
      position: { x: 5, y: 0, width: 1, height: 1 },
      minimized: false,
      maximized: false,
      configuration: {}
    },
    // Second row - Watchlist, chart, and order book
    {
      id: 'watchlist',
      type: 'watchlist',
      position: { x: 0, y: 1, width: 1, height: 2 },
      minimized: false,
      maximized: false,
      configuration: { symbols: ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'] }
    },
    {
      id: 'chart',
      type: 'chart',
      position: { x: 1, y: 1, width: 3, height: 2 },
      minimized: false,
      maximized: false,
      configuration: { symbol: 'SPY', timeframe: '1m' }
    },
    {
      id: 'order-book',
      type: 'order-book',
      position: { x: 4, y: 1, width: 1, height: 2 },
      minimized: false,
      maximized: false,
      configuration: { symbol: 'SPY' }
    },
    {
      id: 'news',
      type: 'news',
      position: { x: 5, y: 1, width: 1, height: 2 },
      minimized: false,
      maximized: false,
      configuration: { symbols: ['SPY', 'QQQ', 'AAPL'], categories: ['market', 'earnings'] }
    },
    // Third row - Analytics and risk
    {
      id: 'analytics',
      type: 'analytics',
      position: { x: 0, y: 3, width: 2, height: 2 },
      minimized: false,
      maximized: false,
      configuration: { timeframe: '1D' }
    },
    {
      id: 'risk-dashboard',
      type: 'risk-dashboard',
      position: { x: 2, y: 3, width: 2, height: 2 },
      minimized: false,
      maximized: false,
      configuration: {}
    },
    // Fourth row - Positions and orders
    {
      id: 'positions',
      type: 'positions',
      position: { x: 4, y: 3, width: 2, height: 1 },
      minimized: false,
      maximized: false,
      configuration: {}
    },
    {
      id: 'orders',
      type: 'orders',
      position: { x: 4, y: 4, width: 2, height: 1 },
      minimized: false,
      maximized: false,
      configuration: {}
    }
  ]
};

const defaultSettings: UserSettings = {
  theme: 'bloomberg',
  layout: defaultLayout,
  watchlists: [
    {
      id: 'default',
      name: 'Main Watchlist',
      symbols: ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN'],
      items: [],
      lastUpdated: new Date().toISOString()
    }
  ],
  notifications: {
    enabled: true,
    riskAlerts: true,
    tradeExecutions: true,
    priceAlerts: true,
    systemAlerts: true,
    sound: false
  },
  trading: {
    confirmOrders: true,
    defaultQuantity: 100,
    defaultTimeInForce: 'DAY',
    riskChecks: true,
    paperTrading: true
  },
  display: {
    fontSize: 11,
    gridSize: 4,
    showMilliseconds: false,
    priceDecimalPlaces: 2,
    percentDecimalPlaces: 2,
    flashUpdates: true
  }
};

const initialState: UIState = {
  layout: defaultLayout,
  activePanel: null,
  commandLineVisible: true,
  commandHistory: [],
  currentCommand: '',
  notifications: [],
  settings: defaultSettings,
  fullscreenPanel: null,
  sidebarVisible: false,
  bottomPanelHeight: 200,
  keyboardShortcuts: {
    'Ctrl+L': 'toggle-command-line',
    'Ctrl+M': 'maximize-panel',
    'Ctrl+N': 'minimize-panel',
    'Ctrl+R': 'refresh-data',
    'Escape': 'cancel-action',
    'F1': 'show-help',
    'F11': 'toggle-fullscreen'
  }
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setLayout: (state, action: PayloadAction<GridLayout>) => {
      state.layout = action.payload;
    },

    updatePanelPosition: (state, action: PayloadAction<{
      panelId: string;
      position: { x: number; y: number; width: number; height: number };
    }>) => {
      const { panelId, position } = action.payload;
      const panel = state.layout.panels.find(p => p.id === panelId);
      if (panel) {
        panel.position = position;
      }
    },

    updatePanelConfiguration: (state, action: PayloadAction<{
      panelId: string;
      configuration: Record<string, any>;
    }>) => {
      const { panelId, configuration } = action.payload;
      const panel = state.layout.panels.find(p => p.id === panelId);
      if (panel) {
        panel.configuration = { ...panel.configuration, ...configuration };
      }
    },

    minimizePanel: (state, action: PayloadAction<string>) => {
      const panel = state.layout.panels.find(p => p.id === action.payload);
      if (panel) {
        panel.minimized = true;
        panel.maximized = false;
      }
    },

    maximizePanel: (state, action: PayloadAction<string>) => {
      const panel = state.layout.panels.find(p => p.id === action.payload);
      if (panel) {
        panel.maximized = true;
        panel.minimized = false;
      }
    },

    restorePanel: (state, action: PayloadAction<string>) => {
      const panel = state.layout.panels.find(p => p.id === action.payload);
      if (panel) {
        panel.minimized = false;
        panel.maximized = false;
      }
    },

    setActivePanel: (state, action: PayloadAction<string | null>) => {
      state.activePanel = action.payload;
    },

    setFullscreenPanel: (state, action: PayloadAction<string | null>) => {
      state.fullscreenPanel = action.payload;
    },

    toggleCommandLine: (state) => {
      state.commandLineVisible = !state.commandLineVisible;
    },

    setCommandLineVisible: (state, action: PayloadAction<boolean>) => {
      state.commandLineVisible = action.payload;
    },

    addCommandToHistory: (state, action: PayloadAction<string>) => {
      state.commandHistory.unshift(action.payload);
      // Keep only last 100 commands
      if (state.commandHistory.length > 100) {
        state.commandHistory = state.commandHistory.slice(0, 100);
      }
    },

    setCurrentCommand: (state, action: PayloadAction<string>) => {
      state.currentCommand = action.payload;
    },

    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now()
      };
      
      state.notifications.unshift(notification);
      
      // Keep only last 50 notifications
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50);
      }
    },

    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },

    clearNotifications: (state) => {
      state.notifications = state.notifications.filter(n => n.persistent);
    },

    updateSettings: (state, action: PayloadAction<Partial<UserSettings>>) => {
      state.settings = { ...state.settings, ...action.payload };
    },

    updateDisplaySettings: (state, action: PayloadAction<Partial<UserSettings['display']>>) => {
      state.settings.display = { ...state.settings.display, ...action.payload };
    },

    updateTradingSettings: (state, action: PayloadAction<Partial<UserSettings['trading']>>) => {
      state.settings.trading = { ...state.settings.trading, ...action.payload };
    },

    updateNotificationSettings: (state, action: PayloadAction<Partial<UserSettings['notifications']>>) => {
      state.settings.notifications = { ...state.settings.notifications, ...action.payload };
    },

    addPanel: (state, action: PayloadAction<PanelLayout>) => {
      state.layout.panels.push(action.payload);
    },

    removePanel: (state, action: PayloadAction<string>) => {
      state.layout.panels = state.layout.panels.filter(p => p.id !== action.payload);
    },

    reorderPanels: (state, action: PayloadAction<string[]>) => {
      const orderedPanels: PanelLayout[] = [];
      action.payload.forEach(panelId => {
        const panel = state.layout.panels.find(p => p.id === panelId);
        if (panel) {
          orderedPanels.push(panel);
        }
      });
      state.layout.panels = orderedPanels;
    },

    resetLayout: (state) => {
      state.layout = defaultLayout;
      state.activePanel = null;
      state.fullscreenPanel = null;
    },

    setSidebarVisible: (state, action: PayloadAction<boolean>) => {
      state.sidebarVisible = action.payload;
    },

    setBottomPanelHeight: (state, action: PayloadAction<number>) => {
      state.bottomPanelHeight = Math.max(100, Math.min(400, action.payload));
    },

    // Keyboard shortcuts
    updateKeyboardShortcuts: (state, action: PayloadAction<Record<string, string>>) => {
      state.keyboardShortcuts = { ...state.keyboardShortcuts, ...action.payload };
    },

    // Quick actions
    focusSymbol: (state, action: PayloadAction<string>) => {
      const symbol = action.payload.toUpperCase();
      
      // Update chart panel
      const chartPanel = state.layout.panels.find(p => p.type === 'chart');
      if (chartPanel) {
        chartPanel.configuration = { ...chartPanel.configuration, symbol };
      }
      
      // Update order book panel
      const orderBookPanel = state.layout.panels.find(p => p.type === 'order-book');
      if (orderBookPanel) {
        orderBookPanel.configuration = { ...orderBookPanel.configuration, symbol };
      }
    },

    // Theme switching (for future expansion)
    setTheme: (state, action: PayloadAction<string>) => {
      state.settings.theme = action.payload;
    },

    // Performance optimization - batch UI updates
    batchUIUpdate: (state, action: PayloadAction<{
      activePanel?: string | null;
      notifications?: Notification[];
      settings?: Partial<UserSettings>;
    }>) => {
      const { activePanel, notifications, settings } = action.payload;
      
      if (activePanel !== undefined) {
        state.activePanel = activePanel;
      }
      
      if (notifications) {
        state.notifications = [...notifications, ...state.notifications].slice(0, 50);
      }
      
      if (settings) {
        state.settings = { ...state.settings, ...settings };
      }
    }
  },
});

export const {
  setLayout,
  updatePanelPosition,
  updatePanelConfiguration,
  minimizePanel,
  maximizePanel,
  restorePanel,
  setActivePanel,
  setFullscreenPanel,
  toggleCommandLine,
  setCommandLineVisible,
  addCommandToHistory,
  setCurrentCommand,
  addNotification,
  removeNotification,
  clearNotifications,
  updateSettings,
  updateDisplaySettings,
  updateTradingSettings,
  updateNotificationSettings,
  addPanel,
  removePanel,
  reorderPanels,
  resetLayout,
  setSidebarVisible,
  setBottomPanelHeight,
  updateKeyboardShortcuts,
  focusSymbol,
  setTheme,
  batchUIUpdate,
} = uiSlice.actions;

export default uiSlice.reducer;

// Selectors
export const selectLayout = (state: { ui: UIState }) => state.ui.layout;
export const selectActivePanel = (state: { ui: UIState }) => state.ui.activePanel;
export const selectSettings = (state: { ui: UIState }) => state.ui.settings;
export const selectNotifications = (state: { ui: UIState }) => state.ui.notifications;
export const selectCommandLineVisible = (state: { ui: UIState }) => state.ui.commandLineVisible;
export const selectFullscreenPanel = (state: { ui: UIState }) => state.ui.fullscreenPanel;

export const selectPanelById = (state: { ui: UIState }, panelId: string) =>
  state.ui.layout.panels.find(p => p.id === panelId);

export const selectPanelsByType = (state: { ui: UIState }, panelType: string) =>
  state.ui.layout.panels.filter(p => p.type === panelType);

export const selectUnreadNotifications = (state: { ui: UIState }) =>
  state.ui.notifications.length;

export const selectCriticalAlerts = (state: { ui: UIState }) =>
  state.ui.notifications.filter(n => n.type === 'error').length;