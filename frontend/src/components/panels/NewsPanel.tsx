/**
 * News Panel Component
 * Real-time financial news feed with filtering and alerts
 */

import React, { useEffect, useState, useRef } from 'react';
import styled from 'styled-components';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import { formatCurrency, formatTime } from '../../utils/formatters';

const NewsContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const NewsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  background-color: ${props => props.theme.colors.surface};
  flex-shrink: 0;
`;

const FilterButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
`;

const FilterButton = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? props.theme.colors.tertiary : 'transparent'};
  color: ${props => props.$active ? props.theme.colors.background : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 2px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.tertiary};
    color: ${props => props.theme.colors.background};
  }
`;

const NewsContent = styled.div`
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

const NewsItem = styled.div<{ $priority: 'high' | 'medium' | 'low' }>`
  border-bottom: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.sm} 0;
  cursor: pointer;
  transition: background-color ${props => props.theme.animation.fast};
  
  ${props => props.$priority === 'high' && `
    border-left: 3px solid ${props.theme.colors.negative};
    padding-left: ${props.theme.spacing.sm};
    background-color: rgba(255, 94, 77, 0.1);
  `}
  
  ${props => props.$priority === 'medium' && `
    border-left: 3px solid ${props.theme.colors.warning};
    padding-left: ${props.theme.spacing.sm};
    background-color: rgba(255, 206, 84, 0.1);
  `}
  
  &:hover {
    background-color: ${props => props.theme.colors.surface};
  }
  
  &:last-child {
    border-bottom: none;
  }
`;

const NewsTitle = styled.div`
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin-bottom: ${props => props.theme.spacing.xs};
  line-height: 1.2;
`;

const NewsMetadata = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const NewsSource = styled.span`
  color: ${props => props.theme.colors.tertiary};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
`;

const NewsTime = styled.span``;

const NewsSymbols = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.xs};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const SymbolTag = styled.span`
  background-color: ${props => props.theme.colors.tertiary};
  color: ${props => props.theme.colors.background};
  padding: 1px 4px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  border-radius: 2px;
  font-weight: ${props => props.theme.typography.fontWeight.medium};
`;

const NewsSummary = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  line-height: 1.3;
  color: ${props => props.theme.colors.primary};
`;

const LoadingIndicator = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${props => props.theme.spacing.md};
  color: ${props => props.theme.colors.textSecondary};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const ConnectionStatus = styled.div<{ $connected: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.$connected ? props.theme.colors.positive : props.theme.colors.negative};
  
  &::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: ${props => props.$connected ? props.theme.colors.positive : props.theme.colors.negative};
  }
`;

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  timestamp: number;
  symbols: string[];
  priority: 'high' | 'medium' | 'low';
  category: string;
  url?: string;
}

interface NewsPanelProps {
  panelId: string;
  symbols?: string[];
  categories?: string[];
  maxItems?: number;
}

const NewsPanel: React.FC<NewsPanelProps> = ({
  panelId,
  symbols = [],
  categories = ['market', 'earnings', 'crypto'],
  maxItems = 50
}) => {
  const { connectionStatus } = useSelector((state: RootState) => state.marketData);
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const newsContentRef = useRef<HTMLDivElement>(null);

  // Mock news data generator
  const generateMockNews = (): NewsItem[] => {
    const sources = ['Bloomberg', 'Reuters', 'MarketWatch', 'WSJ', 'CNBC', 'Financial Times'];
    const mockSymbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ'];
    const categories = ['earnings', 'market', 'crypto', 'fed', 'tech', 'energy'];
    
    const mockTitles = [
      'Federal Reserve Signals Rate Cut Considerations',
      'Tech Giants Report Strong Q4 Earnings',
      'Market Volatility Increases Amid Inflation Concerns',
      'Cryptocurrency Adoption Accelerates in Enterprise',
      'Energy Sector Sees Major Infrastructure Investment',
      'AI Revolution Drives Technology Stock Rally',
      'Supply Chain Disruptions Impact Manufacturing',
      'Central Bank Digital Currency Development Updates'
    ];
    
    return Array.from({ length: 30 }, (_, i) => ({
      id: `news_${Date.now()}_${i}`,
      title: mockTitles[i % mockTitles.length],
      summary: 'Breaking news development affecting market sentiment and trading activity across multiple sectors.',
      source: sources[Math.floor(Math.random() * sources.length)],
      timestamp: Date.now() - (i * 300000), // 5 minutes apart
      symbols: mockSymbols.slice(0, Math.floor(Math.random() * 3) + 1),
      priority: (['high', 'medium', 'low'] as const)[Math.floor(Math.random() * 3)],
      category: categories[Math.floor(Math.random() * categories.length)]
    }));
  };

  // Initialize news feed
  useEffect(() => {
    const mockNews = generateMockNews();
    setNewsItems(mockNews);
    setIsLoading(false);

    // Simulate real-time news updates
    const newsInterval = setInterval(() => {
      const newItem: NewsItem = {
        id: `news_${Date.now()}_${Math.random()}`,
        title: 'Breaking: Market Update',
        summary: 'Latest market development affecting trading activity.',
        source: 'Reuters',
        timestamp: Date.now(),
        symbols: ['SPY'],
        priority: 'medium',
        category: 'market'
      };
      
      setNewsItems(prev => [newItem, ...prev.slice(0, maxItems - 1)]);
    }, 30000); // New item every 30 seconds

    return () => clearInterval(newsInterval);
  }, [maxItems]);

  const handleNewsClick = (item: NewsItem) => {
    if (item.url) {
      window.open(item.url, '_blank');
    }
  };

  const filteredNews = newsItems.filter(item => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'high') return item.priority === 'high';
    if (activeFilter === 'watchlist') {
      return symbols.some(symbol => item.symbols.includes(symbol));
    }
    return item.category === activeFilter;
  });

  const filters = [
    { key: 'all', label: 'ALL' },
    { key: 'high', label: 'HIGH PRIORITY' },
    { key: 'market', label: 'MARKET' },
    { key: 'earnings', label: 'EARNINGS' },
    { key: 'watchlist', label: 'WATCHLIST' }
  ];

  return (
    <NewsContainer>
      <NewsHeader>
        <FilterButtons>
          {filters.map(filter => (
            <FilterButton
              key={filter.key}
              $active={activeFilter === filter.key}
              onClick={() => setActiveFilter(filter.key)}
            >
              {filter.label}
            </FilterButton>
          ))}
        </FilterButtons>
        
        <ConnectionStatus $connected={connectionStatus === 'connected'}>
          NEWS FEED
        </ConnectionStatus>
      </NewsHeader>

      <NewsContent ref={newsContentRef}>
        {isLoading ? (
          <LoadingIndicator>LOADING NEWS...</LoadingIndicator>
        ) : (
          filteredNews.map(item => (
            <NewsItem
              key={item.id}
              $priority={item.priority}
              onClick={() => handleNewsClick(item)}
            >
              <NewsMetadata>
                <NewsSource>{item.source}</NewsSource>
                <NewsTime>{formatTime(item.timestamp)}</NewsTime>
              </NewsMetadata>
              
              <NewsTitle>{item.title}</NewsTitle>
              
              {item.symbols.length > 0 && (
                <NewsSymbols>
                  {item.symbols.map(symbol => (
                    <SymbolTag key={symbol}>{symbol}</SymbolTag>
                  ))}
                </NewsSymbols>
              )}
              
              <NewsSummary>{item.summary}</NewsSummary>
            </NewsItem>
          ))
        )}
        
        {filteredNews.length === 0 && !isLoading && (
          <LoadingIndicator>NO NEWS AVAILABLE</LoadingIndicator>
        )}
      </NewsContent>
    </NewsContainer>
  );
};

export default NewsPanel;