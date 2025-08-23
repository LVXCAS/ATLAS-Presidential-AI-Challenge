/**
 * Market Data Redux Slice
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { MarketData, Quote, TechnicalIndicators, AsyncState } from '../../types';

interface MarketDataState {
  prices: Record<string, MarketData>;
  quotes: Record<string, Quote>;
  indicators: Record<string, TechnicalIndicators>;
  watchlist: string[];
  selectedSymbol: string | null;
  priceHistory: Record<string, MarketData[]>;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  lastUpdate: number;
  updateCounts: Record<string, number>;
}

const initialState: MarketDataState = {
  prices: {},
  quotes: {},
  indicators: {},
  watchlist: ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN'],
  selectedSymbol: null,
  priceHistory: {},
  connectionStatus: 'disconnected',
  lastUpdate: 0,
  updateCounts: {},
};

const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    updatePrice: (state, action: PayloadAction<MarketData>) => {
      const { symbol } = action.payload;
      
      // Store previous price for change calculation
      const previousPrice = state.prices[symbol]?.price || action.payload.price;
      
      // Update price data
      state.prices[symbol] = {
        ...action.payload,
        change: action.payload.change ?? action.payload.price - previousPrice,
        changePercent: action.payload.changePercent ?? 
          (previousPrice > 0 ? ((action.payload.price - previousPrice) / previousPrice) * 100 : 0)
      };
      
      // Update price history (keep last 1000 points)
      if (!state.priceHistory[symbol]) {
        state.priceHistory[symbol] = [];
      }
      state.priceHistory[symbol].push(action.payload);
      if (state.priceHistory[symbol].length > 1000) {
        state.priceHistory[symbol] = state.priceHistory[symbol].slice(-1000);
      }
      
      // Update counters
      state.updateCounts[symbol] = (state.updateCounts[symbol] || 0) + 1;
      state.lastUpdate = Date.now();
    },

    updateQuote: (state, action: PayloadAction<Quote>) => {
      const { symbol } = action.payload;
      state.quotes[symbol] = action.payload;
      state.lastUpdate = Date.now();
    },

    updateIndicators: (state, action: PayloadAction<TechnicalIndicators>) => {
      const { symbol } = action.payload;
      state.indicators[symbol] = action.payload;
      state.lastUpdate = Date.now();
    },

    updateBatchPrices: (state, action: PayloadAction<MarketData[]>) => {
      action.payload.forEach(priceData => {
        const { symbol } = priceData;
        const previousPrice = state.prices[symbol]?.price || priceData.price;
        
        state.prices[symbol] = {
          ...priceData,
          change: priceData.change ?? priceData.price - previousPrice,
          changePercent: priceData.changePercent ?? 
            (previousPrice > 0 ? ((priceData.price - previousPrice) / previousPrice) * 100 : 0)
        };
        
        state.updateCounts[symbol] = (state.updateCounts[symbol] || 0) + 1;
      });
      state.lastUpdate = Date.now();
    },

    setSelectedSymbol: (state, action: PayloadAction<string | null>) => {
      state.selectedSymbol = action.payload;
    },

    addToWatchlist: (state, action: PayloadAction<string | string[]>) => {
      const symbols = Array.isArray(action.payload) ? action.payload : [action.payload];
      symbols.forEach(symbol => {
        const upperSymbol = symbol.toUpperCase();
        if (!state.watchlist.includes(upperSymbol)) {
          state.watchlist.push(upperSymbol);
        }
      });
    },

    removeFromWatchlist: (state, action: PayloadAction<string>) => {
      const symbol = action.payload.toUpperCase();
      state.watchlist = state.watchlist.filter(s => s !== symbol);
    },

    setConnectionStatus: (state, action: PayloadAction<'connected' | 'disconnected' | 'connecting'>) => {
      state.connectionStatus = action.payload;
    },

    clearMarketData: (state) => {
      state.prices = {};
      state.quotes = {};
      state.indicators = {};
      state.priceHistory = {};
      state.updateCounts = {};
      state.lastUpdate = 0;
    },

    setPriceHistory: (state, action: PayloadAction<{ symbol: string; history: MarketData[] }>) => {
      const { symbol, history } = action.payload;
      state.priceHistory[symbol] = history;
    },

    // Bulk update for initial data load
    initializeMarketData: (state, action: PayloadAction<{
      prices?: Record<string, MarketData>;
      quotes?: Record<string, Quote>;
      indicators?: Record<string, TechnicalIndicators>;
    }>) => {
      const { prices, quotes, indicators } = action.payload;
      
      if (prices) {
        state.prices = { ...state.prices, ...prices };
      }
      
      if (quotes) {
        state.quotes = { ...state.quotes, ...quotes };
      }
      
      if (indicators) {
        state.indicators = { ...state.indicators, ...indicators };
      }
      
      state.lastUpdate = Date.now();
    },

    // Performance optimization - batch updates
    batchUpdate: (state, action: PayloadAction<{
      prices?: MarketData[];
      quotes?: Quote[];
      indicators?: TechnicalIndicators[];
    }>) => {
      const { prices, quotes, indicators } = action.payload;
      
      if (prices) {
        prices.forEach(priceData => {
          const { symbol } = priceData;
          const previousPrice = state.prices[symbol]?.price || priceData.price;
          
          state.prices[symbol] = {
            ...priceData,
            change: priceData.change ?? priceData.price - previousPrice,
            changePercent: priceData.changePercent ?? 
              (previousPrice > 0 ? ((priceData.price - previousPrice) / previousPrice) * 100 : 0)
          };
          
          state.updateCounts[symbol] = (state.updateCounts[symbol] || 0) + 1;
        });
      }
      
      if (quotes) {
        quotes.forEach(quote => {
          state.quotes[quote.symbol] = quote;
        });
      }
      
      if (indicators) {
        indicators.forEach(indicator => {
          state.indicators[indicator.symbol] = indicator;
        });
      }
      
      state.lastUpdate = Date.now();
    }
  },
});

export const {
  updatePrice,
  updateQuote,
  updateIndicators,
  updateBatchPrices,
  setSelectedSymbol,
  addToWatchlist,
  removeFromWatchlist,
  setConnectionStatus,
  clearMarketData,
  setPriceHistory,
  initializeMarketData,
  batchUpdate,
} = marketDataSlice.actions;

export default marketDataSlice.reducer;

// Selectors
export const selectPriceData = (state: { marketData: MarketDataState }, symbol: string) => 
  state.marketData.prices[symbol];

export const selectQuoteData = (state: { marketData: MarketDataState }, symbol: string) => 
  state.marketData.quotes[symbol];

export const selectIndicatorData = (state: { marketData: MarketDataState }, symbol: string) => 
  state.marketData.indicators[symbol];

export const selectWatchlistData = (state: { marketData: MarketDataState }) => {
  return state.marketData.watchlist.map(symbol => ({
    symbol,
    ...state.marketData.prices[symbol],
    quote: state.marketData.quotes[symbol],
    indicators: state.marketData.indicators[symbol]
  })).filter(item => item.price !== undefined);
};

export const selectMarketDataStats = (state: { marketData: MarketDataState }) => ({
  totalSymbols: Object.keys(state.marketData.prices).length,
  totalUpdates: Object.values(state.marketData.updateCounts).reduce((sum, count) => sum + count, 0),
  lastUpdate: state.marketData.lastUpdate,
  connectionStatus: state.marketData.connectionStatus,
  watchlistSize: state.marketData.watchlist.length
});