import React, { useState } from 'react';
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

const OrderForm = styled.form`
  display: grid;
  gap: 8px;
  margin-bottom: 15px;
`;

const InputGroup = styled.div`
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 8px;
  align-items: center;
`;

const Label = styled.label`
  color: #ffa500;
  font-size: 9px;
  width: 50px;
`;

const Input = styled.input`
  background: #000;
  border: 1px solid #333;
  color: #00ff00;
  padding: 4px 8px;
  font-family: 'Courier New', monospace;
  font-size: 10px;
`;

const Select = styled.select`
  background: #000;
  border: 1px solid #333;
  color: #00ff00;
  padding: 4px;
  font-family: 'Courier New', monospace;
  font-size: 10px;
`;

const ButtonGroup = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
`;

const ActionButton = styled.button<{ $type: 'buy' | 'sell' }>`
  background: ${props => props.$type === 'buy' ? '#004400' : '#440000'};
  color: ${props => props.$type === 'buy' ? '#00ff00' : '#ff0000'};
  border: 1px solid ${props => props.$type === 'buy' ? '#00ff00' : '#ff0000'};
  padding: 6px;
  cursor: pointer;
  font-family: 'Courier New', monospace;
  font-size: 10px;
  font-weight: bold;
`;

const OrdersList = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const OrderRow = styled.div`
  display: grid;
  grid-template-columns: auto auto auto auto auto;
  gap: 8px;
  padding: 4px;
  border-bottom: 1px solid #222;
  font-size: 9px;
  align-items: center;
`;

const OrderPanel: React.FC = () => {
  const { placeOrder, orders } = useTradingStore();
  const [symbol, setSymbol] = useState('');
  const [quantity, setQuantity] = useState('100');
  const [orderType, setOrderType] = useState('MARKET');
  const [price, setPrice] = useState('');

  const mockOrders = [
    { id: '1', symbol: 'AAPL', side: 'BUY' as const, quantity: 100, status: 'FILLED' as const, timestamp: '10:30:15' },
    { id: '2', symbol: 'TSLA', side: 'SELL' as const, quantity: 50, status: 'PENDING' as const, timestamp: '10:28:42' },
    { id: '3', symbol: 'NVDA', side: 'BUY' as const, quantity: 25, status: 'FILLED' as const, timestamp: '10:25:31' }
  ];

  const displayOrders = orders.length > 0 ? orders : mockOrders;

  const handleSubmit = (e: React.FormEvent, side: 'BUY' | 'SELL') => {
    e.preventDefault();
    if (!symbol || !quantity) return;
    
    placeOrder(symbol.toUpperCase(), side, parseInt(quantity));
    setSymbol('');
    setQuantity('100');
  };

  return (
    <PanelContainer>
      <PanelTitle>Order Entry</PanelTitle>
      
      <OrderForm>
        <InputGroup>
          <Label>SYMBOL:</Label>
          <Input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="AAPL"
          />
        </InputGroup>
        
        <InputGroup>
          <Label>QTY:</Label>
          <Input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
          />
        </InputGroup>
        
        <InputGroup>
          <Label>TYPE:</Label>
          <Select value={orderType} onChange={(e) => setOrderType(e.target.value)}>
            <option value="MARKET">MARKET</option>
            <option value="LIMIT">LIMIT</option>
          </Select>
        </InputGroup>
        
        {orderType === 'LIMIT' && (
          <InputGroup>
            <Label>PRICE:</Label>
            <Input
              type="number"
              step="0.01"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              placeholder="0.00"
            />
          </InputGroup>
        )}
        
        <ButtonGroup>
          <ActionButton 
            $type="buy" 
            type="button"
            onClick={(e) => handleSubmit(e, 'BUY')}
          >
            BUY
          </ActionButton>
          <ActionButton 
            $type="sell" 
            type="button"
            onClick={(e) => handleSubmit(e, 'SELL')}
          >
            SELL
          </ActionButton>
        </ButtonGroup>
      </OrderForm>
      
      <PanelTitle style={{ marginBottom: '5px', fontSize: '10px' }}>Recent Orders</PanelTitle>
      <OrdersList>
        {displayOrders.map((order) => (
          <OrderRow key={order.id}>
            <span style={{ color: order.side === 'BUY' ? '#00ff00' : '#ff0000' }}>
              {order.side}
            </span>
            <span style={{ color: '#ffa500' }}>{order.symbol}</span>
            <span>{order.quantity}</span>
            <span style={{ 
              color: order.status === 'FILLED' ? '#00ff00' : 
                     order.status === 'PENDING' ? '#ffa500' : '#ff0000' 
            }}>
              {order.status}
            </span>
            <span style={{ color: '#666' }}>{order.timestamp}</span>
          </OrderRow>
        ))}
      </OrdersList>
    </PanelContainer>
  );
};

export default OrderPanel;