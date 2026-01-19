import React from 'react';
import styled from 'styled-components';

const OrderBookContainer = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: ${props => props.theme.colors.background};
`;

const BookSection = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.sm};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  color: ${props => props.theme.colors.primary};
`;

const BookRow = styled.div<{ $side: 'bid' | 'ask' }>`
  display: flex;
  justify-content: space-between;
  color: ${props => props.$side === 'bid' ? 
    props.theme.colors.positive : 
    props.theme.colors.negative
  };
  margin: 2px 0;
`;

interface OrderBookPanelProps {
  symbol?: string;
  panelId: string;
}

const OrderBookPanel: React.FC<OrderBookPanelProps> = ({ symbol = 'SPY' }) => {
  return (
    <OrderBookContainer>
      <BookSection>
        <div style={{ color: '#FFA500', marginBottom: '8px' }}>ASKS ({symbol})</div>
        {[...Array(5)].map((_, i) => (
          <BookRow key={i} $side="ask">
            <span>{(450.50 + i * 0.01).toFixed(2)}</span>
            <span>{(1000 - i * 100).toLocaleString()}</span>
          </BookRow>
        ))}
      </BookSection>
      
      <BookSection>
        <div style={{ color: '#FFA500', marginBottom: '8px' }}>BIDS ({symbol})</div>
        {[...Array(5)].map((_, i) => (
          <BookRow key={i} $side="bid">
            <span>{(450.45 - i * 0.01).toFixed(2)}</span>
            <span>{(1200 + i * 150).toLocaleString()}</span>
          </BookRow>
        ))}
      </BookSection>
    </OrderBookContainer>
  );
};

export default OrderBookPanel;