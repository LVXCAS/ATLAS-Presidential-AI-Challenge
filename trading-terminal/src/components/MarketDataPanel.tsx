import React from 'react';
import styled from 'styled-components';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-balham.css';
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

const GridContainer = styled.div`
  flex: 1;
  
  .ag-theme-balham {
    --ag-background-color: #0a0a0a;
    --ag-header-background-color: #000000;
    --ag-odd-row-background-color: #0a0a0a;
    --ag-header-foreground-color: #ffa500;
    --ag-foreground-color: #00ff00;
    --ag-border-color: #333;
    font-family: 'Courier New', monospace;
    font-size: 10px;
  }
`;

const MarketDataPanel: React.FC = () => {
  const marketData = useTradingStore((state) => state.marketData);
  const placeOrder = useTradingStore((state) => state.placeOrder);

  const rowData = Object.values(marketData);

  const columnDefs: any[] = [
    { 
      field: 'symbol', 
      headerName: 'SYMBOL',
      width: 80,
      cellStyle: { color: '#ffa500', fontWeight: 'bold' }
    },
    { 
      field: 'price', 
      headerName: 'PRICE',
      width: 80,
      valueFormatter: (params: any) => `${params.value?.toFixed(2) || '0.00'}`,
      cellStyle: (params: any) => ({
        color: params.data.change >= 0 ? '#00ff00' : '#ff0000'
      })
    },
    { 
      field: 'change', 
      headerName: 'CHANGE',
      width: 70,
      valueFormatter: (params: any) => `${params.value >= 0 ? '+' : ''}${params.value?.toFixed(2) || '0.00'}`,
      cellStyle: (params: any) => ({
        color: params.value >= 0 ? '#00ff00' : '#ff0000'
      })
    },
    { 
      field: 'change_percent', 
      headerName: 'CHANGE%',
      width: 80,
      valueFormatter: (params: any) => `${params.value >= 0 ? '+' : ''}${params.value?.toFixed(2) || '0.00'}%`,
      cellStyle: (params: any) => ({
        color: params.value >= 0 ? '#00ff00' : '#ff0000'
      })
    },
    { 
      field: 'volume', 
      headerName: 'VOLUME',
      width: 90,
      valueFormatter: (params: any) => {
        const vol = params.value || 0;
        if (vol >= 1000000) return `${(vol / 1000000).toFixed(1)}M`;
        if (vol >= 1000) return `${(vol / 1000).toFixed(1)}K`;
        return vol.toString();
      }
    },
    { 
      field: 'vwap', 
      headerName: 'VWAP',
      width: 80,
      valueFormatter: (params: any) => `${params.value?.toFixed(2) || '0.00'}`
    },
    {
      headerName: 'ACTIONS',
      width: 120,
      cellRenderer: (params: any) => {
        return (
          <div style={{ display: 'flex', gap: '5px', height: '100%', alignItems: 'center' }}>
            <button
              onClick={() => placeOrder(params.data.symbol, 'BUY', 100)}
              style={{
                backgroundColor: '#004400',
                color: '#00ff00',
                border: '1px solid #00ff00',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '9px'
              }}
            >
              BUY
            </button>
            <button
              onClick={() => placeOrder(params.data.symbol, 'SELL', 100)}
              style={{
                backgroundColor: '#440000',
                color: '#ff0000',
                border: '1px solid #ff0000',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '9px'
              }}
            >
              SELL
            </button>
          </div>
        );
      }
    }
  ];

  const defaultColDef = {
    sortable: true,
    filter: true,
    resizable: true,
    suppressSizeToFit: false
  };

  return (
    <PanelContainer>
      <PanelTitle>Market Data</PanelTitle>
      <GridContainer className="ag-theme-balham">
        <AgGridReact
          rowData={rowData}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          headerHeight={25}
          rowHeight={22}
          suppressRowClickSelection={true}
          enableCellTextSelection={false}
          animateRows={false}
        />
      </GridContainer>
    </PanelContainer>
  );
};

export default MarketDataPanel;