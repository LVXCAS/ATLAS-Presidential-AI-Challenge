import React from 'react';
import styled from 'styled-components';
import { useTradingStore } from '../stores/tradingStore';

const PanelContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 10px;
`;

const PanelTitle = styled.div`
  color: #ffa500;
  font-size: 11px;
  font-weight: bold;
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
`;

const NewsList = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const NewsItem = styled.div`
  padding: 8px;
  border-bottom: 1px solid #222;
  margin-bottom: 8px;
`;

const NewsTitle = styled.div`
  color: #00ff00;
  font-size: 10px;
  font-weight: bold;
  margin-bottom: 4px;
`;

const NewsMetadata = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: #666;
  margin-bottom: 4px;
`;

const SentimentBadge = styled.span<{ $sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' }>`
  color: ${props => {
    switch (props.$sentiment) {
      case 'POSITIVE': return '#00ff00';
      case 'NEGATIVE': return '#ff0000';
      default: return '#ffa500';
    }
  }};
  font-size: 8px;
  font-weight: bold;
`;

const NewsContent = styled.div`
  color: #ccc;
  font-size: 9px;
  line-height: 1.2;
`;

const NewsPanel: React.FC = () => {
  const newsItems = useTradingStore((state) => state.newsItems);

  const mockNews = [
    {
      title: 'Federal Reserve Hints at Rate Cuts Amid Economic Slowdown',
      content: 'Fed officials suggest potential monetary policy shifts...',
      sentiment_score: 0.3,
      impact_prediction: 'POSITIVE' as const,
      source: 'Reuters',
      timestamp: '2024-08-24T10:30:00Z',
      symbols_mentioned: ['SPY', 'QQQ']
    },
    {
      title: 'Tesla Q2 Earnings Beat Expectations, Stock Surges',
      content: 'Electric vehicle maker reports strong quarterly results...',
      sentiment_score: 0.7,
      impact_prediction: 'POSITIVE' as const,
      source: 'Bloomberg',
      timestamp: '2024-08-24T09:15:00Z',
      symbols_mentioned: ['TSLA']
    },
    {
      title: 'Oil Prices Drop on Supply Concerns',
      content: 'Crude futures fall as supply disruption fears ease...',
      sentiment_score: -0.4,
      impact_prediction: 'NEGATIVE' as const,
      source: 'MarketWatch',
      timestamp: '2024-08-24T08:45:00Z',
      symbols_mentioned: ['USO', 'XOM']
    }
  ];

  const displayNews = newsItems.length > 0 ? newsItems : mockNews;

  return (
    <PanelContainer>
      <PanelTitle>Market News & Sentiment</PanelTitle>
      <NewsList>
        {displayNews.map((item, index) => (
          <NewsItem key={index}>
            <NewsTitle>{item.title}</NewsTitle>
            <NewsMetadata>
              <span>{item.source}</span>
              <SentimentBadge $sentiment={item.impact_prediction}>
                {item.impact_prediction}
              </SentimentBadge>
            </NewsMetadata>
            <NewsContent>{item.content}</NewsContent>
          </NewsItem>
        ))}
      </NewsList>
    </PanelContainer>
  );
};

export default NewsPanel;