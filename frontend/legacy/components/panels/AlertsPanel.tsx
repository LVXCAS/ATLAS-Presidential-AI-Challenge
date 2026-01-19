/**
 * Alerts Panel Component
 * Real-time system and trading alerts with priority management
 */

import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { removeNotification, clearNotifications } from '../../store/slices/uiSlice';
import { formatTime } from '../../utils/formatters';

const AlertsContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const AlertsHeader = styled.div`
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
  gap: ${props => props.theme.spacing.xs};
`;

const FilterButton = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? props.theme.colors.tertiary : 'transparent'};
  color: ${props => props.$active ? props.theme.colors.background : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 2px 6px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.tertiary};
    color: ${props => props.theme.colors.background};
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.xs};
`;

const ActionButton = styled.button`
  background: transparent;
  color: ${props => props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 2px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.negative};
    color: ${props => props.theme.colors.background};
  }
`;

const AlertsContent = styled.div`
  flex: 1;
  overflow-y: auto;
  
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

const AlertItem = styled.div<{ $type: 'info' | 'warning' | 'error' | 'success' }>`
  border-bottom: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.sm};
  cursor: pointer;
  transition: background-color ${props => props.theme.animation.fast};
  position: relative;
  
  ${props => {
    const colors = {
      info: props.theme.colors.tertiary,
      warning: props.theme.colors.warning,
      error: props.theme.colors.negative,
      success: props.theme.colors.positive
    };
    
    return `
      border-left: 3px solid ${colors[props.$type]};
      background-color: ${colors[props.$type]}10;
    `;
  }}
  
  &:hover {
    background-color: ${props => props.theme.colors.surface};
  }
  
  &:last-child {
    border-bottom: none;
  }
`;

const AlertHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const AlertTitle = styled.div`
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-size: ${props => props.theme.typography.fontSize.sm};
  flex: 1;
`;

const AlertTime = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-left: ${props => props.theme.spacing.sm};
`;

const AlertMessage = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  line-height: 1.3;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const AlertActions = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.sm};
`;

const AlertActionButton = styled.button<{ $type?: 'primary' | 'secondary' }>`
  background: ${props => props.$type === 'primary' ? props.theme.colors.tertiary : 'transparent'};
  color: ${props => props.$type === 'primary' ? props.theme.colors.background : props.theme.colors.primary};
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

const DismissButton = styled.button`
  position: absolute;
  top: ${props => props.theme.spacing.xs};
  right: ${props => props.theme.spacing.xs};
  background: none;
  border: none;
  color: ${props => props.theme.colors.textSecondary};
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  padding: 2px;
  line-height: 1;
  
  &:hover {
    color: ${props => props.theme.colors.negative};
  }
`;

const EmptyState = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: ${props => props.theme.colors.textSecondary};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const AlertBadge = styled.div<{ $count: number }>`
  display: ${props => props.$count > 0 ? 'flex' : 'none'};
  align-items: center;
  justify-content: center;
  background-color: ${props => props.theme.colors.negative};
  color: ${props => props.theme.colors.background};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  min-width: 16px;
  height: 16px;
  border-radius: 8px;
  padding: 0 4px;
  margin-left: ${props => props.theme.spacing.xs};
`;

interface AlertsPanelProps {
  panelId: string;
  maxAlerts?: number;
}

const AlertsPanel: React.FC<AlertsPanelProps> = ({
  panelId,
  maxAlerts = 100
}) => {
  const dispatch = useDispatch();
  const { notifications } = useSelector((state: RootState) => state.ui);
  const [filter, setFilter] = useState<string>('all');

  const handleDismiss = (alertId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    dispatch(removeNotification(alertId));
  };

  const handleClearAll = () => {
    dispatch(clearNotifications());
  };

  const handleAlertClick = (alert: any) => {
    // Handle alert click actions
    console.log('Alert clicked:', alert);
  };

  const handleAlertAction = (alert: any, action: string) => {
    console.log('Alert action:', action, alert);
    // Implement specific alert actions
  };

  const filteredAlerts = notifications.filter(alert => {
    if (filter === 'all') return true;
    return alert.type === filter;
  }).slice(0, maxAlerts);

  const alertCounts = {
    all: notifications.length,
    error: notifications.filter(n => n.type === 'error').length,
    warning: notifications.filter(n => n.type === 'warning').length,
    info: notifications.filter(n => n.type === 'info').length,
    success: notifications.filter(n => n.type === 'success').length
  };

  const filters = [
    { key: 'all', label: 'ALL', count: alertCounts.all },
    { key: 'error', label: 'ERROR', count: alertCounts.error },
    { key: 'warning', label: 'WARN', count: alertCounts.warning },
    { key: 'info', label: 'INFO', count: alertCounts.info },
    { key: 'success', label: 'SUCCESS', count: alertCounts.success }
  ];

  return (
    <AlertsContainer>
      <AlertsHeader>
        <FilterButtons>
          {filters.map(filterItem => (
            <FilterButton
              key={filterItem.key}
              $active={filter === filterItem.key}
              onClick={() => setFilter(filterItem.key)}
            >
              {filterItem.label}
              <AlertBadge $count={filterItem.count}>
                {filterItem.count}
              </AlertBadge>
            </FilterButton>
          ))}
        </FilterButtons>
        
        <ActionButtons>
          <ActionButton onClick={handleClearAll}>
            CLEAR ALL
          </ActionButton>
        </ActionButtons>
      </AlertsHeader>

      <AlertsContent>
        {filteredAlerts.length === 0 ? (
          <EmptyState>
            {notifications.length === 0 ? 'NO ALERTS' : 'NO ALERTS MATCH FILTER'}
          </EmptyState>
        ) : (
          filteredAlerts.map(alert => (
            <AlertItem
              key={alert.id}
              $type={alert.type}
              onClick={() => handleAlertClick(alert)}
            >
              <DismissButton
                onClick={(e) => handleDismiss(alert.id, e)}
                title="Dismiss"
              >
                ×
              </DismissButton>
              
              <AlertHeader>
                <AlertTitle>{alert.title}</AlertTitle>
                <AlertTime>{formatTime(alert.timestamp)}</AlertTime>
              </AlertHeader>
              
              <AlertMessage>{alert.message}</AlertMessage>
              
              {alert.actions && alert.actions.length > 0 && (
                <AlertActions>
                  {alert.actions.map((action, index) => (
                    <AlertActionButton
                      key={index}
                      $type={index === 0 ? 'primary' : 'secondary'}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAlertAction(alert, action.action);
                      }}
                    >
                      {action.label}
                    </AlertActionButton>
                  ))}
                </AlertActions>
              )}
            </AlertItem>
          ))
        )}
      </AlertsContent>
    </AlertsContainer>
  );
};

export default AlertsPanel;