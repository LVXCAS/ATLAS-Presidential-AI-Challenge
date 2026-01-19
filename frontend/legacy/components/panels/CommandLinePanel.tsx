import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const CommandContainer = styled.div`
  height: 100%;
  background-color: ${props => props.theme.colors.background};
  display: flex;
  flex-direction: column;
  font-family: ${props => props.theme.typography.fontFamily};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const CommandLine = styled.div`
  display: flex;
  align-items: center;
  padding: ${props => props.theme.spacing.md};
  border-top: 1px solid ${props => props.theme.colors.border};
  background-color: ${props => props.theme.colors.surface};
`;

const Prompt = styled.span`
  color: ${props => props.theme.colors.tertiary};
  margin-right: ${props => props.theme.spacing.md};
`;

const Input = styled.input`
  flex: 1;
  background: transparent;
  border: none;
  color: ${props => props.theme.colors.primary};
  font-family: inherit;
  font-size: inherit;
  outline: none;
  text-transform: uppercase;

  &::placeholder {
    color: ${props => props.theme.colors.gray};
    opacity: 0.7;
  }
`;

const Output = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.md};
  color: ${props => props.theme.colors.primary};
  overflow-y: auto;
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const CommandLinePanel: React.FC = () => {
  const [command, setCommand] = useState('');
  const [output, setOutput] = useState<string[]>([
    'BLOOMBERG TERMINAL COMMAND LINE',
    'TYPE HELP FOR AVAILABLE COMMANDS',
    ''
  ]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Auto-focus the input
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!command.trim()) return;

    const newOutput = [...output, `> ${command}`];
    
    // Simple command processing
    const cmd = command.toUpperCase().trim();
    
    switch (cmd) {
      case 'HELP':
        newOutput.push(
          'AVAILABLE COMMANDS:',
          '  HELP - SHOW THIS HELP',
          '  CLEAR - CLEAR SCREEN',
          '  STATUS - SYSTEM STATUS',
          '  TIME - CURRENT TIME',
          '  [SYMBOL] - QUOTE FOR SYMBOL',
          ''
        );
        break;
      
      case 'CLEAR':
        setOutput(['']);
        setCommand('');
        return;
      
      case 'STATUS':
        newOutput.push(
          'SYSTEM STATUS: OPERATIONAL',
          'CONNECTIONS: ACTIVE',
          'MARKET DATA: STREAMING',
          ''
        );
        break;
      
      case 'TIME':
        newOutput.push(
          new Date().toLocaleString(),
          ''
        );
        break;
      
      default:
        if (cmd.match(/^[A-Z]{1,5}$/)) {
          // Looks like a symbol
          newOutput.push(
            `QUOTE: ${cmd}`,
            `PRICE: 450.25 (+2.15 +0.48%)`,
            `VOLUME: 1.2M`,
            ''
          );
        } else {
          newOutput.push(`UNKNOWN COMMAND: ${cmd}`, '');
        }
    }
    
    setOutput(newOutput);
    setCommand('');
  };

  return (
    <CommandContainer>
      <Output>
        {output.map((line, index) => (
          <div key={index}>{line || '\u00A0'}</div>
        ))}
      </Output>
      
      <CommandLine>
        <Prompt>BLOOMBERG&gt;</Prompt>
        <form onSubmit={handleSubmit} style={{ flex: 1 }}>
          <Input
            ref={inputRef}
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="ENTER COMMAND"
          />
        </form>
      </CommandLine>
    </CommandContainer>
  );
};

export default CommandLinePanel;