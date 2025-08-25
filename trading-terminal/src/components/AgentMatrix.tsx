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

const AgentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  flex: 1;
  overflow-y: auto;
`;

const AgentRow = styled.div`
  display: grid;
  grid-template-columns: 1fr auto auto auto;
  gap: 8px;
  padding: 4px;
  border: 1px solid #333;
  font-size: 9px;
  align-items: center;
`;

const AgentName = styled.div`
  color: #ffa500;
  font-weight: bold;
`;

const AgentMetric = styled.div<{ $positive?: boolean }>`
  color: ${props => props.$positive ? '#00ff00' : '#ff0000'};
  text-align: right;
`;

const AgentToggle = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? '#004400' : '#440000'};
  color: ${props => props.$active ? '#00ff00' : '#ff0000'};
  border: 1px solid ${props => props.$active ? '#00ff00' : '#ff0000'};
  padding: 2px 6px;
  cursor: pointer;
  font-size: 8px;
  font-family: 'Courier New', monospace;
`;

const TrainingStatus = styled.div`
  margin-top: 10px;
  padding: 8px;
  border: 1px solid #333;
  font-size: 9px;
`;

const AgentMatrix: React.FC = () => {
  const { trainingStatus, toggleAgent } = useTradingStore();

  const agents = [
    { name: 'MOMENTUM', pnl: 1250.50, winRate: 67.3, active: true },
    { name: 'MEAN_REV', pnl: -234.20, winRate: 45.1, active: true },
    { name: 'SENTIMENT', pnl: 890.75, winRate: 71.8, active: true },
    { name: 'CRYPTO_ARB', pnl: 3420.10, winRate: 82.4, active: false },
    { name: 'OPTIONS', pnl: 567.30, winRate: 58.9, active: true },
    { name: 'META_LEARN', pnl: 0.00, winRate: 0.0, active: false }
  ];

  return (
    <PanelContainer>
      <PanelTitle>Agent Performance Matrix</PanelTitle>
      
      <AgentGrid>
        {agents.map((agent) => (
          <AgentRow key={agent.name}>
            <AgentName>{agent.name}</AgentName>
            <AgentMetric $positive={agent.pnl >= 0}>
              ${agent.pnl.toFixed(2)}
            </AgentMetric>
            <AgentMetric $positive={agent.winRate >= 50}>
              {agent.winRate.toFixed(1)}%
            </AgentMetric>
            <AgentToggle 
              $active={agent.active}
              onClick={() => toggleAgent(agent.name)}
            >
              {agent.active ? 'ON' : 'OFF'}
            </AgentToggle>
          </AgentRow>
        ))}
      </AgentGrid>
      
      {trainingStatus && (
        <TrainingStatus>
          <div style={{ color: '#ffa500', marginBottom: '4px' }}>TRAINING STATUS</div>
          <div>Mode: {trainingStatus.is_training ? 'ACTIVE' : 'IDLE'}</div>
          <div>Epoch: {trainingStatus.current_epoch}/{trainingStatus.total_epochs}</div>
          <div>Accuracy: {(trainingStatus.model_accuracy * 100).toFixed(1)}%</div>
          <div>ETA: {trainingStatus.eta_completion}</div>
        </TrainingStatus>
      )}
    </PanelContainer>
  );
};

export default AgentMatrix;