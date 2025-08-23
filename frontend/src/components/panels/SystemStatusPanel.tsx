import React from 'react';
import styled from 'styled-components';

const StatusContainer = styled.div`
  padding: ${props => props.theme.spacing.md};
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
`;

const StatusGroup = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.xl};
`;

const StatusItem = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.primary};
`;

const SystemStatusPanel: React.FC = () => {
  return (
    <StatusContainer>
      <StatusGroup>
        <StatusItem>BLOOMBERG TERMINAL v1.0</StatusItem>
        <StatusItem>MARKET: OPEN</StatusItem>
        <StatusItem>DATA: REAL-TIME</StatusItem>
      </StatusGroup>
      <StatusGroup>
        <StatusItem>NY 09:30:15</StatusItem>
        <StatusItem>LON 14:30:15</StatusItem>
        <StatusItem>TKY 22:30:15</StatusItem>
      </StatusGroup>
    </StatusContainer>
  );
};

export default SystemStatusPanel;