import React, { useState, useCallback, useRef, useEffect } from 'react';
import { StockChart } from './StockChart';

interface StockBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
}

interface SearchResult {
  symbol: string;
  name: string;
  market: string;
  type: string;
}

export const StockSearch: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [stockData, setStockData] = useState<StockBar[]>([]);
  const [stockQuote, setStockQuote] = useState<StockQuote | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout>();

  const searchStocks = useCallback(async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    try {
      const response = await fetch(`http://localhost:8001/api/stocks/search?query=${encodeURIComponent(query)}`);
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
        setShowResults(true);
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    }
  }, []);

  const fetchStockData = async (symbol: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Fetch historical bars (using daily data for better availability)
      const barsResponse = await fetch(`http://localhost:8001/api/stocks/bars/${symbol}?timespan=day&limit=50`);
      if (barsResponse.ok) {
        const barsData = await barsResponse.json();
        setStockData(barsData.bars || []);
      } else {
        throw new Error('Failed to fetch stock data');
      }

      // Fetch current quote
      const quoteResponse = await fetch(`http://localhost:8001/api/stocks/quote/${symbol}`);
      if (quoteResponse.ok) {
        const quoteData = await quoteResponse.json();
        setStockQuote(quoteData);
      }

    } catch (error) {
      console.error('Error fetching stock data:', error);
      setError('Failed to fetch stock data. Please try again.');
      setStockData([]);
      setStockQuote(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);

    // Clear previous timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Debounce search
    searchTimeoutRef.current = setTimeout(() => {
      searchStocks(value);
    }, 300);
  };

  const handleSelectStock = (result: SearchResult) => {
    setSearchQuery(`${result.symbol} - ${result.name}`);
    setSelectedSymbol(result.symbol);
    setShowResults(false);
    fetchStockData(result.symbol);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const symbol = searchQuery.split(' ')[0].toUpperCase();
    if (symbol) {
      setSelectedSymbol(symbol);
      setShowResults(false);
      fetchStockData(symbol);
    }
  };

  return (
    <div style={{ 
      width: '100%', 
      maxWidth: '1200px', 
      margin: '0 auto',
      padding: '20px',
      color: '#fff'
    }}>
      {/* Search Bar */}
      <div style={{ marginBottom: '20px', position: 'relative' }}>
        <form onSubmit={handleSearchSubmit}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={handleSearchInput}
              placeholder="Search stocks (e.g., AAPL, Microsoft, TSLA...)"
              style={{
                flex: 1,
                padding: '12px 16px',
                fontSize: '16px',
                backgroundColor: '#1a1a1a',
                border: '2px solid #333',
                borderRadius: '6px',
                color: '#fff',
                outline: 'none'
              }}
              onFocus={() => searchQuery.length >= 2 && setShowResults(true)}
              onBlur={() => setTimeout(() => setShowResults(false), 200)}
            />
            <button
              type="submit"
              style={{
                padding: '12px 20px',
                backgroundColor: '#ffa500',
                color: '#000',
                border: 'none',
                borderRadius: '6px',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
            >
              SEARCH
            </button>
          </div>
        </form>

        {/* Search Results Dropdown */}
        {showResults && searchResults.length > 0 && (
          <div style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            backgroundColor: '#1a1a1a',
            border: '2px solid #333',
            borderTop: 'none',
            borderRadius: '0 0 6px 6px',
            maxHeight: '300px',
            overflowY: 'auto',
            zIndex: 1000
          }}>
            {searchResults.map((result, index) => (
              <div
                key={`${result.symbol}-${index}`}
                onClick={() => handleSelectStock(result)}
                style={{
                  padding: '12px 16px',
                  borderBottom: '1px solid #333',
                  cursor: 'pointer',
                  ':hover': { backgroundColor: '#333' }
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#333'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <div style={{ fontWeight: 'bold', color: '#ffa500' }}>
                  {result.symbol}
                </div>
                <div style={{ fontSize: '14px', color: '#ccc' }}>
                  {result.name}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  {result.market} • {result.type}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Current Quote */}
      {stockQuote && (
        <div style={{
          padding: '15px',
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '6px',
          marginBottom: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ margin: 0, color: '#ffa500' }}>{stockQuote.symbol}</h2>
              <div style={{ fontSize: '24px', fontWeight: 'bold', margin: '5px 0' }}>
                ${stockQuote.price.toFixed(2)}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontSize: '18px',
                fontWeight: 'bold',
                color: stockQuote.change >= 0 ? '#00ff00' : '#ff0000'
              }}>
                {stockQuote.change >= 0 ? '+' : ''}${stockQuote.change.toFixed(2)}
              </div>
              <div style={{
                fontSize: '16px',
                color: stockQuote.change >= 0 ? '#00ff00' : '#ff0000'
              }}>
                ({stockQuote.change_percent >= 0 ? '+' : ''}{stockQuote.change_percent.toFixed(2)}%)
              </div>
            </div>
          </div>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
            Volume: {stockQuote.volume.toLocaleString()} • {new Date(stockQuote.timestamp).toLocaleString()}
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          fontSize: '18px',
          color: '#ffa500'
        }}>
          Loading stock data...
        </div>
      )}

      {/* Error State */}
      {error && (
        <div style={{
          padding: '15px',
          backgroundColor: '#ff0000',
          color: '#fff',
          borderRadius: '6px',
          marginBottom: '20px'
        }}>
          {error}
        </div>
      )}

      {/* Stock Chart */}
      {selectedSymbol && stockData.length > 0 && !isLoading && (
        <StockChart
          symbol={selectedSymbol}
          data={stockData}
          width={1160}
          height={500}
        />
      )}
    </div>
  );
};