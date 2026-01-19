/**
 * Bloomberg Terminal Formatting Utilities
 * Consistent number and data formatting across the application
 */

// Price formatting
export const formatPrice = (price: number | undefined, decimals = 2): string => {
  if (price === undefined || price === null || isNaN(price)) {
    return '-.--';
  }
  
  return price.toFixed(decimals);
};

// Percentage formatting
export const formatPercent = (percent: number | undefined, decimals = 2): string => {
  if (percent === undefined || percent === null || isNaN(percent)) {
    return '-.--';
  }
  
  const sign = percent >= 0 ? '+' : '';
  return `${sign}${percent.toFixed(decimals)}%`;
};

// Volume formatting (Bloomberg style)
export const formatVolume = (volume: number | undefined): string => {
  if (volume === undefined || volume === null || isNaN(volume) || volume === 0) {
    return '--';
  }
  
  if (volume >= 1_000_000_000) {
    return `${(volume / 1_000_000_000).toFixed(1)}B`;
  }
  
  if (volume >= 1_000_000) {
    return `${(volume / 1_000_000).toFixed(1)}M`;
  }
  
  if (volume >= 1_000) {
    return `${(volume / 1_000).toFixed(1)}K`;
  }
  
  return volume.toString();
};

// Currency formatting
export const formatCurrency = (value: number | undefined, decimals = 2): string => {
  if (value === undefined || value === null || isNaN(value)) {
    return '$-.--';
  }
  
  const sign = value < 0 ? '-' : '';
  const absValue = Math.abs(value);
  
  if (absValue >= 1_000_000_000) {
    return `${sign}$${(absValue / 1_000_000_000).toFixed(1)}B`;
  }
  
  if (absValue >= 1_000_000) {
    return `${sign}$${(absValue / 1_000_000).toFixed(1)}M`;
  }
  
  if (absValue >= 1_000) {
    return `${sign}$${(absValue / 1_000).toFixed(1)}K`;
  }
  
  return `${sign}$${absValue.toFixed(decimals)}`;
};

// Time formatting for Bloomberg style timestamps
export const formatTime = (timestamp: number | string | Date, includeSeconds = true): string => {
  try {
    const date = new Date(timestamp);
    
    if (isNaN(date.getTime())) {
      return '--:--';
    }
    
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    
    if (includeSeconds) {
      return `${hours}:${minutes}:${seconds}`;
    }
    
    return `${hours}:${minutes}`;
  } catch {
    return '--:--';
  }
};

// Date formatting
export const formatDate = (timestamp: number | string | Date, includeTime = false): string => {
  try {
    const date = new Date(timestamp);
    
    if (isNaN(date.getTime())) {
      return '--/--/--';
    }
    
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const year = date.getFullYear().toString().slice(-2);
    
    const dateStr = `${month}/${day}/${year}`;
    
    if (includeTime) {
      return `${dateStr} ${formatTime(timestamp)}`;
    }
    
    return dateStr;
  } catch {
    return '--/--/--';
  }
};

// Position formatting
export const formatPosition = (quantity: number | undefined): string => {
  if (quantity === undefined || quantity === null || isNaN(quantity)) {
    return '--';
  }
  
  if (quantity === 0) {
    return '0';
  }
  
  const sign = quantity > 0 ? '+' : '';
  return `${sign}${Math.abs(quantity).toLocaleString()}`;
};

// P&L formatting with color indication
export const formatPnL = (pnl: number | undefined, includeParentheses = false): {
  text: string;
  color: 'positive' | 'negative' | 'neutral';
} => {
  if (pnl === undefined || pnl === null || isNaN(pnl)) {
    return { text: '-.--', color: 'neutral' };
  }
  
  const color = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';
  
  let text = formatCurrency(Math.abs(pnl));
  
  if (pnl > 0) {
    text = `+${text}`;
  } else if (pnl < 0) {
    text = includeParentheses ? `(${text})` : `-${text}`;
  }
  
  return { text, color };
};

// Basis points formatting
export const formatBasisPoints = (value: number | undefined): string => {
  if (value === undefined || value === null || isNaN(value)) {
    return '--bp';
  }
  
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(0)}bp`;
};

// Duration formatting (for time intervals)
export const formatDuration = (milliseconds: number): string => {
  if (isNaN(milliseconds) || milliseconds < 0) {
    return '--s';
  }
  
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  }
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  
  return `${seconds}s`;
};

// Large number formatting
export const formatLargeNumber = (value: number | undefined, decimals = 1): string => {
  if (value === undefined || value === null || isNaN(value)) {
    return '--';
  }
  
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (absValue >= 1_000_000_000_000) {
    return `${sign}${(absValue / 1_000_000_000_000).toFixed(decimals)}T`;
  }
  
  if (absValue >= 1_000_000_000) {
    return `${sign}${(absValue / 1_000_000_000).toFixed(decimals)}B`;
  }
  
  if (absValue >= 1_000_000) {
    return `${sign}${(absValue / 1_000_000).toFixed(decimals)}M`;
  }
  
  if (absValue >= 1_000) {
    return `${sign}${(absValue / 1_000).toFixed(decimals)}K`;
  }
  
  return `${sign}${absValue.toFixed(decimals)}`;
};

// Ratio formatting (for financial ratios)
export const formatRatio = (ratio: number | undefined, decimals = 2): string => {
  if (ratio === undefined || ratio === null || isNaN(ratio)) {
    return '--';
  }
  
  return ratio.toFixed(decimals);
};

// Greeks formatting (for options)
export const formatGreek = (value: number | undefined, decimals = 4): string => {
  if (value === undefined || value === null || isNaN(value)) {
    return '--';
  }
  
  return value.toFixed(decimals);
};

// Volatility formatting
export const formatVolatility = (vol: number | undefined, asPercent = true): string => {
  if (vol === undefined || vol === null || isNaN(vol)) {
    return '--';
  }
  
  if (asPercent) {
    return `${(vol * 100).toFixed(1)}%`;
  }
  
  return vol.toFixed(4);
};

// Order status formatting
export const formatOrderStatus = (status: string): {
  text: string;
  color: 'positive' | 'negative' | 'neutral' | 'warning';
} => {
  const statusUpper = status.toUpperCase();
  
  switch (statusUpper) {
    case 'FILLED':
      return { text: 'FILLED', color: 'positive' };
    case 'CANCELLED':
    case 'REJECTED':
      return { text: statusUpper, color: 'negative' };
    case 'PENDING_NEW':
    case 'NEW':
      return { text: statusUpper.replace('_', ' '), color: 'neutral' };
    case 'PARTIALLY_FILLED':
      return { text: 'PARTIAL', color: 'warning' };
    default:
      return { text: statusUpper, color: 'neutral' };
  }
};

// Symbol formatting (ensure uppercase)
export const formatSymbol = (symbol: string | undefined): string => {
  if (!symbol || typeof symbol !== 'string') {
    return '--';
  }
  
  return symbol.toUpperCase().trim();
};

// Side formatting
export const formatSide = (side: string): {
  text: string;
  color: 'positive' | 'negative' | 'neutral';
} => {
  const sideUpper = side.toUpperCase();
  
  switch (sideUpper) {
    case 'BUY':
      return { text: 'BUY', color: 'positive' };
    case 'SELL':
      return { text: 'SELL', color: 'negative' };
    default:
      return { text: sideUpper, color: 'neutral' };
  }
};

// Utility to add Bloomberg-style padding to numbers
export const padNumber = (num: number | string, width: number): string => {
  const str = num.toString();
  return str.length >= width ? str : new Array(width - str.length + 1).join(' ') + str;
};

// Utility to truncate text with ellipsis
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) {
    return text;
  }
  
  return text.substring(0, maxLength - 3) + '...';
};

// Bloomberg-style number alignment
export const alignNumber = (value: string, width: number): string => {
  if (value.length >= width) {
    return value;
  }
  
  const padding = width - value.length;
  return ' '.repeat(padding) + value;
};

export default {
  formatPrice,
  formatPercent,
  formatVolume,
  formatCurrency,
  formatTime,
  formatDate,
  formatPosition,
  formatPnL,
  formatBasisPoints,
  formatDuration,
  formatLargeNumber,
  formatRatio,
  formatGreek,
  formatVolatility,
  formatOrderStatus,
  formatSymbol,
  formatSide,
  padNumber,
  truncateText,
  alignNumber,
};