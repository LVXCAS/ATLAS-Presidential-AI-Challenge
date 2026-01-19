import React from 'react';
import styled from 'styled-components';

const RiskContainer = styled.div`
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.warning};
  font-family: ${props => props.theme.typography.fontFamily};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const RiskDashboardPanel: React.FC<{ panelId: string }> = () => {
  return (
    <RiskContainer>
      RISK: NORMAL
      <br />
      VAR: $0.00
    </RiskContainer>
  );
};

export default RiskDashboardPanel;