import React from 'react';
import styled from 'styled-components';

const OrdersContainer = styled.div`
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const OrdersPanel: React.FC<{ panelId: string }> = () => {
  return (
    <OrdersContainer>
      ORDERS: 0 ACTIVE
      <br />
      0 FILLED TODAY
    </OrdersContainer>
  );
};

export default OrdersPanel;