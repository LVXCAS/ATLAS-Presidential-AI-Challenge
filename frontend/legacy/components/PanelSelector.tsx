/**
 * Panel Selector Component
 * Quick panel switching and layout management
 */

import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { addPanel, setActivePanel, focusSymbol } from '../store/slices/uiSlice';

const SelectorContainer = styled.div`
  position: fixed;
  top: 10px;
  left: 10px;
  z-index: 3000;
`;

const SelectorButton = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? props.theme.colors.tertiary : props.theme.colors.surface};
  color: ${props => props.$active ? props.theme.colors.background : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 6px 12px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.theme.colors.tertiary};
    color: ${props => props.theme.colors.background};
  }
`;

const DropdownMenu = styled.div<{ $visible: boolean }>`
  position: absolute;
  top: 100%;
  left: 0;
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-top: none;
  min-width: 200px;
  max-height: 400px;
  overflow-y: auto;
  display: ${props => props.$visible ? 'block' : 'none'};
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  
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

const MenuSection = styled.div`
  border-bottom: 1px solid ${props => props.theme.colors.border};
  
  &:last-child {
    border-bottom: none;
  }
`;

const SectionHeader = styled.div`
  padding: 8px 12px;
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.tertiary};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const MenuItem = styled.button<{ $available: boolean }>`
  width: 100%;
  background: transparent;
  color: ${props => props.$available ? props.theme.colors.primary : props.theme.colors.textSecondary};
  border: none;
  padding: 8px 16px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  text-align: left;
  cursor: ${props => props.$available ? 'pointer' : 'not-allowed'};
  transition: background-color ${props => props.theme.animation.fast};
  
  &:hover {
    background-color: ${props => props.$available ? props.theme.colors.tertiary + '20' : 'transparent'};
  }
  
  &:disabled {
    opacity: 0.5;
  }
`;

const QuickAction = styled.div`
  padding: 4px 12px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const QuickInput = styled.input`
  flex: 1;
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 2px 6px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.tertiary};
  }
`;

const QuickButton = styled.button`
  background: ${props => props.theme.colors.tertiary};
  color: ${props => props.theme.colors.background};
  border: none;
  padding: 2px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  
  &:hover {
    opacity: 0.8;
  }
`;

interface PanelSelectorProps {}

const PanelSelector: React.FC<PanelSelectorProps> = () => {
  const dispatch = useDispatch();
  const { layout } = useSelector((state: RootState) => state.ui);
  const [isOpen, setIsOpen] = useState(false);
  const [quickSymbol, setQuickSymbol] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const availablePanels = [
    { type: 'dashboard', label: 'Dashboard', category: 'overview' },
    { type: 'system-status', label: 'System Status', category: 'overview' },
    { type: 'monitoring', label: 'Monitoring', category: 'overview' },
    { type: 'alerts', label: 'Alerts', category: 'overview' },
    
    { type: 'watchlist', label: 'Market Watch', category: 'market' },
    { type: 'chart', label: 'Technical Chart', category: 'market' },
    { type: 'order-book', label: 'Order Book', category: 'market' },
    { type: 'news', label: 'News Feed', category: 'market' },
    
    { type: 'positions', label: 'Positions', category: 'trading' },
    { type: 'orders', label: 'Orders', category: 'trading' },
    { type: 'analytics', label: 'Analytics', category: 'trading' },
    { type: 'risk-dashboard', label: 'Risk Monitor', category: 'trading' },
    
    { type: 'settings', label: 'Settings', category: 'system' }
  ];

  const existingPanels = layout.panels.map(p => p.type);

  const handlePanelSelect = (panelType: string) => {
    const existingPanel = layout.panels.find(p => p.type === panelType);
    
    if (existingPanel) {
      // Focus existing panel
      dispatch(setActivePanel(existingPanel.id));
    } else {
      // Add new panel
      const newPanel = {
        id: `${panelType}_${Date.now()}`,
        type: panelType,
        position: { x: 0, y: 1, width: 2, height: 2 }, // Default position
        minimized: false,
        maximized: false,
        configuration: {}
      };
      
      dispatch(addPanel(newPanel));
      dispatch(setActivePanel(newPanel.id));
    }
    
    setIsOpen(false);
  };

  const handleQuickSymbol = () => {
    if (quickSymbol.trim()) {
      dispatch(focusSymbol(quickSymbol.trim().toUpperCase()));
      setQuickSymbol('');
      setIsOpen(false);
    }
  };

  const handleQuickSymbolKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleQuickSymbol();
    }
  };

  const groupedPanels = availablePanels.reduce((groups, panel) => {
    if (!groups[panel.category]) {
      groups[panel.category] = [];
    }
    groups[panel.category].push(panel);
    return groups;
  }, {} as Record<string, typeof availablePanels>);

  const categoryLabels = {
    overview: 'Overview',
    market: 'Market Data',
    trading: 'Trading',
    system: 'System'
  };

  return (
    <SelectorContainer ref={containerRef}>
      <SelectorButton 
        $active={isOpen}
        onClick={() => setIsOpen(!isOpen)}
      >
        PANELS {isOpen ? '▼' : '▶'}
      </SelectorButton>
      
      <DropdownMenu $visible={isOpen}>
        <MenuSection>
          <SectionHeader>Quick Symbol</SectionHeader>
          <QuickAction>
            <QuickInput
              type="text"
              placeholder="Enter symbol..."
              value={quickSymbol}
              onChange={(e) => setQuickSymbol(e.target.value)}
              onKeyPress={handleQuickSymbolKeyPress}
            />
            <QuickButton onClick={handleQuickSymbol}>
              GO
            </QuickButton>
          </QuickAction>
        </MenuSection>

        {Object.entries(groupedPanels).map(([category, panels]) => (
          <MenuSection key={category}>
            <SectionHeader>{categoryLabels[category as keyof typeof categoryLabels]}</SectionHeader>
            {panels.map(panel => {
              const isAvailable = !existingPanels.includes(panel.type) || panel.type === 'chart' || panel.type === 'watchlist';
              return (
                <MenuItem
                  key={panel.type}
                  $available={isAvailable}
                  onClick={() => isAvailable && handlePanelSelect(panel.type)}
                  disabled={!isAvailable}
                >
                  {panel.label} {existingPanels.includes(panel.type) && '●'}
                </MenuItem>
              );
            })}
          </MenuSection>
        ))}
      </DropdownMenu>
    </SelectorContainer>
  );
};

export default PanelSelector;