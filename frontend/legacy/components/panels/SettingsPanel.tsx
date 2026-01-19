/**
 * Settings Panel Component
 * Terminal configuration and user preferences
 */

import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { 
  updateDisplaySettings, 
  updateTradingSettings, 
  updateNotificationSettings,
  resetLayout,
  setTheme 
} from '../../store/slices/uiSlice';

const SettingsContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.typography.fontFamily};
  overflow: hidden;
`;

const SettingsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  background-color: ${props => props.theme.colors.surface};
  flex-shrink: 0;
`;

const CategoryTabs = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
`;

const CategoryTab = styled.button<{ $active: boolean }>`
  background: ${props => props.$active ? props.theme.colors.tertiary : 'transparent'};
  color: ${props => props.$active ? props.theme.colors.background : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 12px;
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
  gap: ${props => props.theme.spacing.sm};
`;

const ActionButton = styled.button<{ $type?: 'primary' | 'danger' }>`
  background: ${props => {
    switch (props.$type) {
      case 'primary': return props.theme.colors.tertiary;
      case 'danger': return props.theme.colors.negative;
      default: return 'transparent';
    }
  }};
  color: ${props => {
    switch (props.$type) {
      case 'primary': return props.theme.colors.background;
      case 'danger': return props.theme.colors.background;
      default: return props.theme.colors.primary;
    }
  }};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 12px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all ${props => props.theme.animation.fast};
  
  &:hover {
    opacity: 0.8;
  }
`;

const SettingsContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${props => props.theme.spacing.md};
  
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

const SettingSection = styled.div`
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const SectionTitle = styled.div`
  font-size: ${props => props.theme.typography.fontSize.md};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.tertiary};
  margin-bottom: ${props => props.theme.spacing.md};
  text-transform: uppercase;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  padding-bottom: ${props => props.theme.spacing.xs};
`;

const SettingRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.sm} 0;
  border-bottom: 1px solid ${props => props.theme.colors.border}40;
  
  &:last-child {
    border-bottom: none;
  }
`;

const SettingLabel = styled.div`
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
`;

const SettingDescription = styled.div`
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.textSecondary};
  margin-top: 2px;
`;

const SettingControl = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  min-width: 150px;
  justify-content: flex-end;
`;

const Input = styled.input`
  background: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  width: 80px;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.tertiary};
  }
`;

const Select = styled.select`
  background: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.border};
  padding: 4px 8px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
  min-width: 100px;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.tertiary};
  }
`;

const Checkbox = styled.input.attrs({ type: 'checkbox' })`
  width: 16px;
  height: 16px;
  accent-color: ${props => props.theme.colors.tertiary};
`;

const ColorSwatch = styled.div<{ $color: string }>`
  width: 20px;
  height: 20px;
  background-color: ${props => props.$color};
  border: 1px solid ${props => props.theme.colors.border};
  cursor: pointer;
`;

interface SettingsPanelProps {
  panelId: string;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ panelId }) => {
  const dispatch = useDispatch();
  const { settings } = useSelector((state: RootState) => state.ui);
  const [activeCategory, setActiveCategory] = useState<string>('display');

  const handleDisplaySettingChange = useCallback((key: string, value: any) => {
    dispatch(updateDisplaySettings({ [key]: value }));
  }, [dispatch]);

  const handleTradingSettingChange = useCallback((key: string, value: any) => {
    dispatch(updateTradingSettings({ [key]: value }));
  }, [dispatch]);

  const handleNotificationSettingChange = useCallback((key: string, value: any) => {
    dispatch(updateNotificationSettings({ [key]: value }));
  }, [dispatch]);

  const handleResetLayout = useCallback(() => {
    if (window.confirm('Are you sure you want to reset the layout? This cannot be undone.')) {
      dispatch(resetLayout());
    }
  }, [dispatch]);

  const handleThemeChange = useCallback((theme: string) => {
    dispatch(setTheme(theme));
  }, [dispatch]);

  const categories = [
    { key: 'display', label: 'DISPLAY' },
    { key: 'trading', label: 'TRADING' },
    { key: 'notifications', label: 'NOTIFICATIONS' },
    { key: 'layout', label: 'LAYOUT' },
    { key: 'system', label: 'SYSTEM' }
  ];

  const renderDisplaySettings = () => (
    <SettingSection>
      <SectionTitle>Display Preferences</SectionTitle>
      
      <SettingRow>
        <div>
          <SettingLabel>Font Size</SettingLabel>
          <SettingDescription>Base font size for all panels</SettingDescription>
        </div>
        <SettingControl>
          <Input
            type="number"
            value={settings.display.fontSize}
            min={8}
            max={16}
            onChange={(e) => handleDisplaySettingChange('fontSize', parseInt(e.target.value))}
          />
          <span>px</span>
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Price Decimal Places</SettingLabel>
          <SettingDescription>Number of decimal places for prices</SettingDescription>
        </div>
        <SettingControl>
          <Input
            type="number"
            value={settings.display.priceDecimalPlaces}
            min={0}
            max={8}
            onChange={(e) => handleDisplaySettingChange('priceDecimalPlaces', parseInt(e.target.value))}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Percent Decimal Places</SettingLabel>
          <SettingDescription>Number of decimal places for percentages</SettingDescription>
        </div>
        <SettingControl>
          <Input
            type="number"
            value={settings.display.percentDecimalPlaces}
            min={0}
            max={4}
            onChange={(e) => handleDisplaySettingChange('percentDecimalPlaces', parseInt(e.target.value))}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Flash Updates</SettingLabel>
          <SettingDescription>Highlight price changes with color flash</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.display.flashUpdates}
            onChange={(e) => handleDisplaySettingChange('flashUpdates', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Show Milliseconds</SettingLabel>
          <SettingDescription>Display milliseconds in timestamps</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.display.showMilliseconds}
            onChange={(e) => handleDisplaySettingChange('showMilliseconds', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>
    </SettingSection>
  );

  const renderTradingSettings = () => (
    <SettingSection>
      <SectionTitle>Trading Preferences</SectionTitle>
      
      <SettingRow>
        <div>
          <SettingLabel>Confirm Orders</SettingLabel>
          <SettingDescription>Require confirmation before sending orders</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.trading.confirmOrders}
            onChange={(e) => handleTradingSettingChange('confirmOrders', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Default Quantity</SettingLabel>
          <SettingDescription>Default order quantity</SettingDescription>
        </div>
        <SettingControl>
          <Input
            type="number"
            value={settings.trading.defaultQuantity}
            min={1}
            onChange={(e) => handleTradingSettingChange('defaultQuantity', parseInt(e.target.value))}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Default Time In Force</SettingLabel>
          <SettingDescription>Default order time in force</SettingDescription>
        </div>
        <SettingControl>
          <Select
            value={settings.trading.defaultTimeInForce}
            onChange={(e) => handleTradingSettingChange('defaultTimeInForce', e.target.value)}
          >
            <option value="DAY">DAY</option>
            <option value="GTC">GTC</option>
            <option value="IOC">IOC</option>
            <option value="FOK">FOK</option>
          </Select>
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Risk Checks</SettingLabel>
          <SettingDescription>Enable pre-trade risk checks</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.trading.riskChecks}
            onChange={(e) => handleTradingSettingChange('riskChecks', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Paper Trading</SettingLabel>
          <SettingDescription>Enable paper trading mode</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.trading.paperTrading}
            onChange={(e) => handleTradingSettingChange('paperTrading', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>
    </SettingSection>
  );

  const renderNotificationSettings = () => (
    <SettingSection>
      <SectionTitle>Notification Preferences</SectionTitle>
      
      <SettingRow>
        <div>
          <SettingLabel>Enable Notifications</SettingLabel>
          <SettingDescription>Show system notifications</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.enabled}
            onChange={(e) => handleNotificationSettingChange('enabled', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Risk Alerts</SettingLabel>
          <SettingDescription>Show risk management alerts</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.riskAlerts}
            onChange={(e) => handleNotificationSettingChange('riskAlerts', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Trade Executions</SettingLabel>
          <SettingDescription>Show trade execution notifications</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.tradeExecutions}
            onChange={(e) => handleNotificationSettingChange('tradeExecutions', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Price Alerts</SettingLabel>
          <SettingDescription>Show price alerts</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.priceAlerts}
            onChange={(e) => handleNotificationSettingChange('priceAlerts', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>System Alerts</SettingLabel>
          <SettingDescription>Show system status alerts</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.systemAlerts}
            onChange={(e) => handleNotificationSettingChange('systemAlerts', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>

      <SettingRow>
        <div>
          <SettingLabel>Sound Notifications</SettingLabel>
          <SettingDescription>Play sounds for notifications</SettingDescription>
        </div>
        <SettingControl>
          <Checkbox
            checked={settings.notifications.sound}
            onChange={(e) => handleNotificationSettingChange('sound', e.target.checked)}
          />
        </SettingControl>
      </SettingRow>
    </SettingSection>
  );

  const renderLayoutSettings = () => (
    <SettingSection>
      <SectionTitle>Layout Preferences</SectionTitle>
      
      <SettingRow>
        <div>
          <SettingLabel>Reset Layout</SettingLabel>
          <SettingDescription>Reset to default panel layout</SettingDescription>
        </div>
        <SettingControl>
          <ActionButton $type="danger" onClick={handleResetLayout}>
            RESET
          </ActionButton>
        </SettingControl>
      </SettingRow>
    </SettingSection>
  );

  const renderSystemSettings = () => (
    <SettingSection>
      <SectionTitle>System Preferences</SectionTitle>
      
      <SettingRow>
        <div>
          <SettingLabel>Theme</SettingLabel>
          <SettingDescription>Terminal color scheme</SettingDescription>
        </div>
        <SettingControl>
          <Select
            value={settings.theme}
            onChange={(e) => handleThemeChange(e.target.value)}
          >
            <option value="bloomberg">Bloomberg</option>
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </Select>
        </SettingControl>
      </SettingRow>
    </SettingSection>
  );

  const renderContent = () => {
    switch (activeCategory) {
      case 'display':
        return renderDisplaySettings();
      case 'trading':
        return renderTradingSettings();
      case 'notifications':
        return renderNotificationSettings();
      case 'layout':
        return renderLayoutSettings();
      case 'system':
        return renderSystemSettings();
      default:
        return renderDisplaySettings();
    }
  };

  return (
    <SettingsContainer>
      <SettingsHeader>
        <CategoryTabs>
          {categories.map(category => (
            <CategoryTab
              key={category.key}
              $active={activeCategory === category.key}
              onClick={() => setActiveCategory(category.key)}
            >
              {category.label}
            </CategoryTab>
          ))}
        </CategoryTabs>
        
        <ActionButtons>
          <ActionButton onClick={() => window.location.reload()}>
            REFRESH
          </ActionButton>
        </ActionButtons>
      </SettingsHeader>

      <SettingsContent>
        {renderContent()}
      </SettingsContent>
    </SettingsContainer>
  );
};

export default SettingsPanel;