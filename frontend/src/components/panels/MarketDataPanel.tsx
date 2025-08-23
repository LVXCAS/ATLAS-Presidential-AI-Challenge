/**
 * Market Data Panel Component
 * Real-time watchlist with price updates and performance indicators
 */

import React, { useState, useEffect, useMemo } from 'react';
import styled from 'styled-components';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import { selectWatchlistData } from '../../store/slices/marketDataSlice';
import { formatPrice, formatPercent, formatVolume } from '../../utils/formatters';

const PanelContainer = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const TableContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-family: ${props => props.theme.typography.fontFamily};
`;

const TableHeader = styled.th`
  background-color: ${props => props.theme.colors.surface};
  color: ${props => props.theme.colors.tertiary};
  padding: ${props => props.theme.spacing.sm};
  text-align: left;
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-size: ${props => props.theme.typography.fontSize.xs};
  letter-spacing: 0.5px;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  position: sticky;
  top: 0;
  z-index: 10;
`;

const TableRow = styled.tr<{ $isSelected?: boolean }>`
  cursor: pointer;
  transition: background-color ${props => props.theme.animation.fast};
  background-color: ${props => props.$isSelected ? 
    'rgba(0, 255, 0, 0.1)' : 'transparent'};

  &:hover {
    background-color: rgba(0, 255, 0, 0.05);
  }

  &:nth-child(even) {
    background-color: rgba(0, 17, 34, 0.3);
  }
`;

const TableCell = styled.td<{ 
  $align?: 'left' | 'right' | 'center';
  $color?: 'positive' | 'negative' | 'neutral';
  $flash?: boolean;
}>`
  padding: ${props => props.theme.spacing.sm};
  text-align: ${props => props.$align || 'left'};
  border-bottom: 1px solid ${props => props.theme.colors.divider};
  color: ${props => {
    if (props.$color === 'positive') return props.theme.colors.positive;
    if (props.$color === 'negative') return props.theme.colors.negative;
    if (props.$color === 'neutral') return props.theme.colors.neutral;
    return props.theme.colors.primary;
  }};
  
  ${props => props.$flash && `
    animation: ${props.theme.animation.priceFlash};
  `}
`;

const SymbolCell = styled(TableCell)`
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => props.theme.colors.tertiary};
`;

const StatusBar = styled.div`
  background-color: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.sm};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondary};
  border-top: 1px solid ${props => props.theme.colors.border};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const UpdateIndicator = styled.span<{ $updating: boolean }>`
  color: ${props => props.$updating ? 
    props.theme.colors.warning : 
    props.theme.colors.gray
  };
  
  ${props => props.$updating && `
    animation: ${props.theme.animation.priceFlash} infinite;
  `}
`;

interface MarketDataPanelProps {
  symbols?: string[];
  panelId: string;
}

const MarketDataPanel: React.FC<MarketDataPanelProps> = ({ 
  symbols: propSymbols,
  panelId 
}) => {
  const watchlistData = useSelector(selectWatchlistData);
  const { lastUpdate, updateCounts } = useSelector((state: RootState) => state.marketData);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [flashingCells, setFlashingCells] = useState<Set<string>>(new Set());
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);

  // Use provided symbols or fallback to watchlist
  const displaySymbols = propSymbols || watchlistData.map(item => item.symbol);

  // Filtered and sorted data
  const sortedData = useMemo(() => {
    let data = watchlistData.filter(item => displaySymbols.includes(item.symbol));

    if (sortConfig) {
      data.sort((a, b) => {
        let aValue = a[sortConfig.key as keyof typeof a];
        let bValue = b[sortConfig.key as keyof typeof b];

        if (typeof aValue === 'string' && typeof bValue === 'string') {
          aValue = aValue.toLowerCase();
          bValue = bValue.toLowerCase();
        }

        if (aValue < bValue) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }

    return data;
  }, [watchlistData, displaySymbols, sortConfig]);

  // Handle price flashing effect
  useEffect(() => {
    const newFlashingCells = new Set<string>();
    
    sortedData.forEach(item => {
      if (item.timestamp && Date.now() - item.timestamp < 1000) {
        newFlashingCells.add(`${item.symbol}-price`);
        newFlashingCells.add(`${item.symbol}-change`);
      }
    });

    if (newFlashingCells.size > 0) {
      setFlashingCells(newFlashingCells);
      
      const timer = setTimeout(() => {
        setFlashingCells(new Set());
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [lastUpdate, sortedData]);

  // Handle sorting
  const handleSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'desc';
    
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }

    setSortConfig({ key, direction });
  };

  // Handle row selection
  const handleRowClick = (symbol: string) => {
    setSelectedSymbol(symbol === selectedSymbol ? null : symbol);
    // Could dispatch action to focus symbol across panels
  };

  const getChangeColor = (change: number | undefined): 'positive' | 'negative' | 'neutral' => {
    if (!change || change === 0) return 'neutral';
    return change > 0 ? 'positive' : 'negative';
  };

  const formatChange = (change: number | undefined): string => {
    if (!change) return '0.00';
    return change > 0 ? `+${change.toFixed(2)}` : change.toFixed(2);
  };

  const totalUpdates = Object.values(updateCounts).reduce((sum, count) => sum + count, 0);

  return (
    <PanelContainer>
      <TableContainer>
        <Table>
          <thead>
            <tr>
              <TableHeader onClick={() => handleSort('symbol')}>
                SYMBOL {sortConfig?.key === 'symbol' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </TableHeader>
              <TableHeader onClick={() => handleSort('price')} style={{ textAlign: 'right' }}>
                PRICE {sortConfig?.key === 'price' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </TableHeader>
              <TableHeader onClick={() => handleSort('change')} style={{ textAlign: 'right' }}>
                CHG {sortConfig?.key === 'change' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </TableHeader>
              <TableHeader onClick={() => handleSort('changePercent')} style={{ textAlign: 'right' }}>
                CHG% {sortConfig?.key === 'changePercent' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </TableHeader>
              <TableHeader onClick={() => handleSort('volume')} style={{ textAlign: 'right' }}>
                VOL {sortConfig?.key === 'volume' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </TableHeader>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((item) => (
              <TableRow
                key={item.symbol}
                $isSelected={selectedSymbol === item.symbol}
                onClick={() => handleRowClick(item.symbol)}
              >
                <SymbolCell>
                  {item.symbol}
                </SymbolCell>
                <TableCell 
                  $align="right"
                  $flash={flashingCells.has(`${item.symbol}-price`)}
                  $color={getChangeColor(item.change)}
                >
                  {formatPrice(item.price)}
                </TableCell>
                <TableCell 
                  $align="right"
                  $color={getChangeColor(item.change)}
                  $flash={flashingCells.has(`${item.symbol}-change`)}
                >
                  {formatChange(item.change)}
                </TableCell>
                <TableCell 
                  $align="right"
                  $color={getChangeColor(item.change)}
                  $flash={flashingCells.has(`${item.symbol}-change`)}
                >
                  {formatPercent(item.changePercent)}
                </TableCell>
                <TableCell $align="right">
                  {formatVolume(item.volume)}
                </TableCell>
              </TableRow>
            ))}
          </tbody>
        </Table>
      </TableContainer>

      <StatusBar>
        <span>
          {sortedData.length} SYMBOLS | {totalUpdates} UPDATES
        </span>
        <UpdateIndicator $updating={Date.now() - lastUpdate < 5000}>
          {Date.now() - lastUpdate < 5000 ? 'UPDATING' : 'IDLE'}
        </UpdateIndicator>
      </StatusBar>
    </PanelContainer>
  );
};

export default MarketDataPanel;