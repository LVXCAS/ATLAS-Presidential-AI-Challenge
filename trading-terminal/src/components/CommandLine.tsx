import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useTradingStore } from '../stores/tradingStore';

const CommandLineContainer = styled.div`
  background: #0a0a0a;
  border-top: 1px solid #333;
  padding: 8px;
  font-size: 11px;
  height: 40px;
  display: flex;
  align-items: center;
`;

const Prompt = styled.span`
  color: #ffa500;
  margin-right: 8px;
`;

const Input = styled.input`
  background: transparent;
  border: none;
  outline: none;
  color: #00ff00;
  font-family: 'Courier New', monospace;
  font-size: 11px;
  flex: 1;
  
  &::placeholder {
    color: #666;
  }
`;

const CommandLine: React.FC = () => {
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const { placeOrder, toggleAgent, marketData } = useTradingStore();

  const executeCommand = async (cmd: string) => {
    const parts = cmd.trim().toUpperCase().split(' ');
    const action = parts[0];

    try {
      switch (action) {
        case 'BUY':
        case 'SELL':
          if (parts.length >= 3) {
            const symbol = parts[1];
            const quantity = parseInt(parts[2]) || 100;
            await placeOrder(symbol, action as 'BUY' | 'SELL', quantity);
            console.log(`${action} ${quantity} ${symbol} executed`);
          }
          break;
          
        case 'AGENT':
          if (parts[1] === 'START' || parts[1] === 'STOP') {
            const agentName = parts[2];
            await toggleAgent(agentName);
            console.log(`Agent ${agentName} ${parts[1].toLowerCase()}ed`);
          }
          break;
          
        case 'STATUS':
          console.log('System Status:', Object.keys(marketData).length, 'symbols tracked');
          break;
          
        case 'HELP':
          console.log('Commands: BUY <SYMBOL> <QTY>, SELL <SYMBOL> <QTY>, AGENT START/STOP <NAME>, STATUS, CLEAR');
          break;
          
        case 'CLEAR':
          console.clear();
          break;
          
        default:
          console.log(`Unknown command: ${cmd}`);
      }
    } catch (error) {
      console.error('Command execution error:', error);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!command.trim()) return;
    
    setHistory(prev => [...prev, command]);
    setHistoryIndex(-1);
    executeCommand(command);
    setCommand('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < history.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setCommand(history[history.length - 1 - newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setCommand(history[history.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setCommand('');
      }
    }
  };

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <CommandLineContainer>
      <Prompt>HTQT&gt;</Prompt>
      <form onSubmit={handleSubmit} style={{ flex: 1, display: 'flex' }}>
        <Input
          ref={inputRef}
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter command (type HELP for commands)"
          autoComplete="off"
        />
      </form>
    </CommandLineContainer>
  );
};

export default CommandLine;