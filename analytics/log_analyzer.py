"""
Hive Trade Trading Log Analyzer and Optimizer
Comprehensive analysis and optimization of trading system logs
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from collections import Counter, defaultdict
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogMetrics:
    """Trading log analysis metrics"""
    total_entries: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    error_rate_pct: float
    avg_entries_per_hour: float
    peak_activity_hour: int
    most_common_errors: List[Tuple[str, int]]
    performance_issues: List[str]
    trading_patterns: Dict[str, Any]

@dataclass
class PerformanceIssue:
    """Performance issue identification"""
    timestamp: datetime
    component: str
    issue_type: str
    severity: str
    description: str
    impact: str
    recommendation: str

class TradingLogAnalyzer:
    """
    Advanced trading log analysis and optimization system
    """
    
    def __init__(self, db_path: str = "trading_optimized.db"):
        self.db_path = db_path
        self.connection = None
        self.log_patterns = self._initialize_patterns()
        
    def connect_db(self) -> bool:
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for log analysis"""
        return {
            'error_patterns': re.compile(r'ERROR|CRITICAL|FATAL|Exception|Error:|Failed|Timeout', re.IGNORECASE),
            'warning_patterns': re.compile(r'WARNING|WARN|Deprecated|Slow|Retry', re.IGNORECASE),
            'performance_patterns': re.compile(r'slow|timeout|memory|cpu|latency|delay|bottleneck', re.IGNORECASE),
            'trading_patterns': re.compile(r'BUY|SELL|TRADE|ORDER|POSITION|FILL|CANCEL', re.IGNORECASE),
            'network_patterns': re.compile(r'connection|socket|network|http|api|websocket', re.IGNORECASE),
            'database_patterns': re.compile(r'database|sql|query|connection pool|deadlock', re.IGNORECASE)
        }
    
    def analyze_database_logs(self, days_back: int = 7) -> LogMetrics:
        """Analyze logs stored in the database"""
        try:
            query = """
            SELECT level, message, component, timestamp
            FROM trading_logs 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            """
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            df = pd.read_sql_query(query, self.connection, 
                                 params=[cutoff_time.timestamp()])
            
            if df.empty:
                logger.warning("No logs found in database")
                return self._create_empty_metrics()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Basic metrics
            total_entries = len(df)
            error_count = len(df[df['level'] == 'ERROR'])
            warning_count = len(df[df['level'] == 'WARNING'])
            info_count = len(df[df['level'] == 'INFO'])
            debug_count = len(df[df['level'] == 'DEBUG'])
            error_rate_pct = (error_count / total_entries * 100) if total_entries > 0 else 0
            
            # Time-based analysis
            df['hour'] = df['timestamp'].dt.hour
            entries_per_hour = df.groupby('hour').size()
            avg_entries_per_hour = entries_per_hour.mean()
            peak_activity_hour = entries_per_hour.idxmax() if not entries_per_hour.empty else 0
            
            # Error analysis
            error_messages = df[df['level'] == 'ERROR']['message'].tolist()
            most_common_errors = Counter(error_messages).most_common(10)
            
            # Performance issues
            performance_issues = self._identify_performance_issues(df)
            
            # Trading patterns
            trading_patterns = self._analyze_trading_patterns(df)
            
            logger.info(f"Analyzed {total_entries} log entries from database")
            
            return LogMetrics(
                total_entries=total_entries,
                error_count=error_count,
                warning_count=warning_count,
                info_count=info_count,
                debug_count=debug_count,
                error_rate_pct=error_rate_pct,
                avg_entries_per_hour=avg_entries_per_hour,
                peak_activity_hour=peak_activity_hour,
                most_common_errors=most_common_errors,
                performance_issues=performance_issues,
                trading_patterns=trading_patterns
            )
            
        except Exception as e:
            logger.error(f"Database log analysis failed: {e}")
            return self._create_empty_metrics()
    
    def analyze_file_logs(self, log_directory: str = "logs") -> LogMetrics:
        """Analyze logs from files"""
        try:
            log_files = []
            
            # Find log files
            for pattern in ['*.log', '*.txt', '*.out']:
                log_files.extend(glob.glob(os.path.join(log_directory, pattern)))
                log_files.extend(glob.glob(os.path.join(log_directory, '**', pattern), recursive=True))
            
            if not log_files:
                logger.warning(f"No log files found in {log_directory}")
                return self._create_empty_metrics()
            
            logger.info(f"Found {len(log_files)} log files")
            
            all_entries = []
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            entry = self._parse_log_entry(line.strip(), log_file, line_num)
                            if entry:
                                all_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Could not read {log_file}: {e}")
            
            if not all_entries:
                logger.warning("No valid log entries found")
                return self._create_empty_metrics()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(all_entries)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # Apply same analysis as database logs
            return self._analyze_log_dataframe(df)
            
        except Exception as e:
            logger.error(f"File log analysis failed: {e}")
            return self._create_empty_metrics()
    
    def _parse_log_entry(self, line: str, filename: str, line_num: int) -> Optional[Dict]:
        """Parse a single log entry"""
        if not line.strip():
            return None
        
        # Try to extract timestamp, level, and message
        # Common formats: [TIMESTAMP] LEVEL: MESSAGE or TIMESTAMP LEVEL MESSAGE
        
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
            r'\[([^\]]+)\]',
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'
        ]
        
        timestamp = None
        for pattern in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp = pd.to_datetime(match.group(1))
                    break
                except:
                    continue
        
        if not timestamp:
            timestamp = datetime.now()  # Fallback
        
        # Extract log level
        level = 'INFO'  # Default
        level_match = re.search(r'\b(ERROR|CRITICAL|FATAL|WARNING|WARN|INFO|DEBUG)\b', line, re.IGNORECASE)
        if level_match:
            level = level_match.group(1).upper()
        
        # Extract component (if available)
        component = 'unknown'
        component_patterns = [
            r'(\w+Agent)',
            r'(\w+Manager)',
            r'(\w+Service)',
            r'\[(\w+)\]'
        ]
        
        for pattern in component_patterns:
            match = re.search(pattern, line)
            if match:
                component = match.group(1)
                break
        
        return {
            'timestamp': timestamp,
            'level': level,
            'message': line,
            'component': component,
            'file': filename,
            'line_number': line_num
        }
    
    def _analyze_log_dataframe(self, df: pd.DataFrame) -> LogMetrics:
        """Analyze log DataFrame and return metrics"""
        total_entries = len(df)
        error_count = len(df[df['level'] == 'ERROR'])
        warning_count = len(df[df['level'] == 'WARNING'])
        info_count = len(df[df['level'] == 'INFO'])
        debug_count = len(df[df['level'] == 'DEBUG'])
        error_rate_pct = (error_count / total_entries * 100) if total_entries > 0 else 0
        
        # Time-based analysis
        df['hour'] = df['timestamp'].dt.hour
        entries_per_hour = df.groupby('hour').size()
        avg_entries_per_hour = entries_per_hour.mean()
        peak_activity_hour = entries_per_hour.idxmax() if not entries_per_hour.empty else 0
        
        # Error analysis
        error_messages = df[df['level'] == 'ERROR']['message'].tolist()
        most_common_errors = Counter(error_messages).most_common(10)
        
        # Performance issues
        performance_issues = self._identify_performance_issues(df)
        
        # Trading patterns
        trading_patterns = self._analyze_trading_patterns(df)
        
        return LogMetrics(
            total_entries=total_entries,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            debug_count=debug_count,
            error_rate_pct=error_rate_pct,
            avg_entries_per_hour=avg_entries_per_hour,
            peak_activity_hour=peak_activity_hour,
            most_common_errors=most_common_errors,
            performance_issues=performance_issues,
            trading_patterns=trading_patterns
        )
    
    def _identify_performance_issues(self, df: pd.DataFrame) -> List[str]:
        """Identify performance issues from log entries"""
        issues = []
        
        # High error rate
        error_rate = len(df[df['level'] == 'ERROR']) / len(df) * 100 if len(df) > 0 else 0
        if error_rate > 5:
            issues.append(f"High error rate: {error_rate:.1f}% (threshold: 5%)")
        
        # Memory issues
        memory_issues = df[df['message'].str.contains('memory|Memory|OutOfMemory', case=False, na=False)]
        if len(memory_issues) > 0:
            issues.append(f"Memory issues detected: {len(memory_issues)} instances")
        
        # Timeout issues
        timeout_issues = df[df['message'].str.contains('timeout|Timeout|timed out', case=False, na=False)]
        if len(timeout_issues) > 0:
            issues.append(f"Timeout issues detected: {len(timeout_issues)} instances")
        
        # Connection issues
        connection_issues = df[df['message'].str.contains('connection|Connection|disconnect', case=False, na=False)]
        if len(connection_issues) > 10:
            issues.append(f"Frequent connection issues: {len(connection_issues)} instances")
        
        # Database issues
        db_issues = df[df['message'].str.contains('database|Database|SQL|sql', case=False, na=False)]
        db_errors = db_issues[db_issues['level'] == 'ERROR']
        if len(db_errors) > 0:
            issues.append(f"Database errors detected: {len(db_errors)} instances")
        
        return issues
    
    def _analyze_trading_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading-related log patterns"""
        trading_logs = df[df['message'].str.contains('BUY|SELL|TRADE|ORDER|POSITION', case=False, na=False)]
        
        patterns = {}
        
        if len(trading_logs) > 0:
            # Trading frequency by hour
            trading_by_hour = trading_logs.groupby(trading_logs['timestamp'].dt.hour).size()
            patterns['peak_trading_hour'] = trading_by_hour.idxmax() if not trading_by_hour.empty else None
            patterns['trading_entries_count'] = len(trading_logs)
            
            # Order types
            buy_orders = len(trading_logs[trading_logs['message'].str.contains('BUY', case=False, na=False)])
            sell_orders = len(trading_logs[trading_logs['message'].str.contains('SELL', case=False, na=False)])
            patterns['buy_sell_ratio'] = buy_orders / sell_orders if sell_orders > 0 else float('inf')
            
            # Agent activity
            agent_activity = trading_logs['component'].value_counts().to_dict()
            patterns['most_active_agent'] = max(agent_activity.items(), key=lambda x: x[1])[0] if agent_activity else None
        
        return patterns
    
    def _create_empty_metrics(self) -> LogMetrics:
        """Create empty metrics when no data is available"""
        return LogMetrics(
            total_entries=0,
            error_count=0,
            warning_count=0,
            info_count=0,
            debug_count=0,
            error_rate_pct=0.0,
            avg_entries_per_hour=0.0,
            peak_activity_hour=0,
            most_common_errors=[],
            performance_issues=[],
            trading_patterns={}
        )
    
    def optimize_logging_configuration(self, current_metrics: LogMetrics) -> Dict[str, Any]:
        """Generate optimized logging configuration"""
        config = {
            'log_level': 'INFO',
            'max_file_size': '100MB',
            'backup_count': 7,
            'rotation': 'daily',
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'level': 'INFO',
                    'filename': 'trading.log',
                    'max_bytes': 104857600,  # 100MB
                    'backup_count': 7
                },
                'error_file': {
                    'level': 'ERROR',
                    'filename': 'trading_errors.log',
                    'max_bytes': 52428800,   # 50MB
                    'backup_count': 14
                }
            }
        }
        
        # Adjust based on current metrics
        if current_metrics.error_rate_pct > 10:
            config['log_level'] = 'DEBUG'  # More verbose for troubleshooting
            config['handlers']['debug_file'] = {
                'level': 'DEBUG',
                'filename': 'trading_debug.log',
                'max_bytes': 52428800,
                'backup_count': 3
            }
        
        if current_metrics.avg_entries_per_hour > 1000:
            config['max_file_size'] = '200MB'  # Larger files for high volume
            config['handlers']['file']['max_bytes'] = 209715200
        
        return config
    
    def generate_optimization_recommendations(self, metrics: LogMetrics) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Error rate recommendations
        if metrics.error_rate_pct > 5:
            recommendations.append("CRITICAL: Error rate exceeds 5% - immediate investigation required")
            recommendations.append("Implement automated error alerting")
            recommendations.append("Add error rate monitoring dashboard")
        elif metrics.error_rate_pct > 2:
            recommendations.append("WARNING: Error rate above 2% - monitor closely")
        
        # Performance recommendations
        if 'memory' in str(metrics.performance_issues).lower():
            recommendations.append("Optimize memory usage - consider garbage collection tuning")
            recommendations.append("Add memory usage monitoring")
        
        if 'timeout' in str(metrics.performance_issues).lower():
            recommendations.append("Investigate timeout issues - may need connection pool tuning")
            recommendations.append("Implement retry mechanisms with exponential backoff")
        
        # Volume recommendations
        if metrics.avg_entries_per_hour > 2000:
            recommendations.append("High log volume detected - consider log level optimization")
            recommendations.append("Implement log sampling for non-critical messages")
            recommendations.append("Use structured logging for better analysis")
        
        # Trading pattern recommendations
        if metrics.trading_patterns.get('buy_sell_ratio', 1) > 2:
            recommendations.append("Buy/sell ratio imbalanced - review trading strategy")
        
        # General recommendations
        recommendations.extend([
            "Implement centralized logging with ELK stack",
            "Add log rotation and archiving policies",
            "Create log-based alerting rules",
            "Implement real-time log monitoring dashboard",
            "Add structured logging with consistent formats",
            "Implement log correlation IDs for request tracing"
        ])
        
        return recommendations
    
    def generate_analysis_report(self, db_metrics: LogMetrics, 
                                file_metrics: LogMetrics) -> str:
        """Generate comprehensive log analysis report"""
        
        total_entries = db_metrics.total_entries + file_metrics.total_entries
        total_errors = db_metrics.error_count + file_metrics.error_count
        avg_error_rate = ((db_metrics.error_rate_pct + file_metrics.error_rate_pct) / 2) if total_entries > 0 else 0
        
        recommendations = self.generate_optimization_recommendations(db_metrics)
        optimized_config = self.optimize_logging_configuration(db_metrics)
        
        report = f"""
HIVE TRADE TRADING LOG ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

EXECUTIVE SUMMARY:
{'*'*30}

Total Log Entries:        {total_entries:,}
Total Error Count:        {total_errors:,}
Average Error Rate:       {avg_error_rate:.2f}%
Performance Issues:       {len(db_metrics.performance_issues + file_metrics.performance_issues)}

System Health: {'+++ Excellent' if avg_error_rate < 1 else '++ Good' if avg_error_rate < 3 else '+/- Needs Attention' if avg_error_rate < 10 else '--- Critical'}

DATABASE LOG ANALYSIS:
{'*'*30}

Total Entries:            {db_metrics.total_entries:,}
Error Count:              {db_metrics.error_count:,} ({db_metrics.error_rate_pct:.2f}%)
Warning Count:            {db_metrics.warning_count:,}
Info Count:               {db_metrics.info_count:,}
Debug Count:              {db_metrics.debug_count:,}

Activity Patterns:
  Avg Entries/Hour:       {db_metrics.avg_entries_per_hour:.1f}
  Peak Activity Hour:     {db_metrics.peak_activity_hour:02d}:00

FILE LOG ANALYSIS:
{'*'*30}

Total Entries:            {file_metrics.total_entries:,}
Error Count:              {file_metrics.error_count:,} ({file_metrics.error_rate_pct:.2f}%)
Warning Count:            {file_metrics.warning_count:,}
Info Count:               {file_metrics.info_count:,}
Debug Count:              {file_metrics.debug_count:,}

MOST COMMON ERRORS:
{'*'*30}

Database Errors:
"""
        
        for i, (error, count) in enumerate(db_metrics.most_common_errors[:5], 1):
            report += f"{i:2d}. ({count:3d}x) {error[:80]}{'...' if len(error) > 80 else ''}\n"
        
        if not db_metrics.most_common_errors:
            report += "    No errors found in database logs\n"
        
        report += f"""
File Errors:
"""
        
        for i, (error, count) in enumerate(file_metrics.most_common_errors[:5], 1):
            report += f"{i:2d}. ({count:3d}x) {error[:80]}{'...' if len(error) > 80 else ''}\n"
        
        if not file_metrics.most_common_errors:
            report += "    No errors found in file logs\n"
        
        report += f"""

PERFORMANCE ISSUES:
{'*'*30}

Database Issues:
"""
        for issue in db_metrics.performance_issues:
            report += f"  - {issue}\n"
        
        if not db_metrics.performance_issues:
            report += "  - No performance issues detected in database logs\n"
        
        report += f"""
File Issues:
"""
        for issue in file_metrics.performance_issues:
            report += f"  - {issue}\n"
        
        if not file_metrics.performance_issues:
            report += "  - No performance issues detected in file logs\n"
        
        report += f"""

TRADING PATTERNS ANALYSIS:
{'*'*30}

Database Trading Patterns:
"""
        
        if db_metrics.trading_patterns:
            for key, value in db_metrics.trading_patterns.items():
                report += f"  {key.replace('_', ' ').title()}: {value}\n"
        else:
            report += "  No trading patterns detected in database logs\n"
        
        report += f"""
File Trading Patterns:
"""
        
        if file_metrics.trading_patterns:
            for key, value in file_metrics.trading_patterns.items():
                report += f"  {key.replace('_', ' ').title()}: {value}\n"
        else:
            report += "  No trading patterns detected in file logs\n"
        
        report += f"""

OPTIMIZATION RECOMMENDATIONS:
{'*'*30}
"""
        
        for i, recommendation in enumerate(recommendations[:10], 1):
            report += f"{i:2d}. {recommendation}\n"
        
        report += f"""

OPTIMIZED LOGGING CONFIGURATION:
{'*'*30}

Recommended Settings:
  Log Level:              {optimized_config['log_level']}
  Max File Size:          {optimized_config['max_file_size']}
  Backup Count:           {optimized_config['backup_count']}
  Rotation:               {optimized_config['rotation']}

Handler Configuration:
"""
        
        for handler_name, handler_config in optimized_config['handlers'].items():
            report += f"  {handler_name.title()}:\n"
            for key, value in handler_config.items():
                report += f"    {key}: {value}\n"
        
        report += f"""

MONITORING THRESHOLDS:
{'*'*30}

Error Rates:
  Excellent:              < 1%
  Good:                   1-3%
  Warning:                3-5%
  Critical:               > 5%

Log Volume:
  Low:                    < 500 entries/hour
  Normal:                 500-1500 entries/hour
  High:                   1500-3000 entries/hour
  Very High:              > 3000 entries/hour

IMMEDIATE ACTION ITEMS:
{'*'*30}

Priority 1 (Critical):
"""
        
        critical_actions = [action for action in recommendations if 'CRITICAL' in action]
        for action in critical_actions:
            report += f"  - {action.replace('CRITICAL: ', '')}\n"
        
        if not critical_actions:
            report += "  - No critical actions required\n"
        
        report += f"""
Priority 2 (High):
  - Implement automated log monitoring
  - Set up error rate alerting
  - Create log analysis dashboard
  - Optimize high-volume log sources

Priority 3 (Medium):
  - Archive old log files
  - Implement log compression
  - Set up log rotation policies
  - Create log retention policies

NEXT STEPS:
{'*'*30}

1. Immediate (Today):
   - Address any critical error rates
   - Implement basic log rotation
   - Set up error monitoring

2. Short-term (This Week):
   - Deploy optimized logging configuration
   - Create log monitoring dashboard
   - Implement automated alerting

3. Long-term (This Month):
   - Deploy centralized logging system
   - Implement advanced log analysis
   - Create comprehensive monitoring

{'='*70}
Log Analysis Complete - Recommendations Generated
"""
        
        return report
    
    def create_sample_logs(self) -> bool:
        """Create sample log entries for testing"""
        try:
            cursor = self.connection.cursor()
            
            sample_logs = [
                ('INFO', 'Trading system started successfully', 'main', int(datetime.now().timestamp())),
                ('INFO', 'Connected to Alpaca API', 'api_manager', int(datetime.now().timestamp())),
                ('DEBUG', 'Market data received for AAPL', 'market_data_feed', int(datetime.now().timestamp())),
                ('INFO', 'BUY order placed for 100 shares of AAPL at $175.00', 'trading_agent', int(datetime.now().timestamp())),
                ('WARNING', 'High volatility detected for TSLA', 'risk_manager', int(datetime.now().timestamp())),
                ('ERROR', 'Connection timeout to market data feed', 'api_manager', int(datetime.now().timestamp())),
                ('INFO', 'SELL order executed for 50 shares of GOOGL at $2800.00', 'trading_agent', int(datetime.now().timestamp())),
                ('DEBUG', 'Portfolio rebalancing completed', 'portfolio_manager', int(datetime.now().timestamp())),
                ('WARNING', 'Memory usage above 80%', 'system_monitor', int(datetime.now().timestamp())),
                ('INFO', 'Daily P&L: +$1,250.50', 'portfolio_manager', int(datetime.now().timestamp())),
            ]
            
            cursor.executemany("""
                INSERT INTO trading_logs (level, message, component, timestamp)
                VALUES (?, ?, ?, ?)
            """, sample_logs)
            
            self.connection.commit()
            logger.info(f"Created {len(sample_logs)} sample log entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create sample logs: {e}")
            return False

def main():
    """Main log analysis workflow"""
    
    print("HIVE TRADE TRADING LOG ANALYZER")
    print("="*45)
    
    # Initialize analyzer
    analyzer = TradingLogAnalyzer("trading_optimized.db")
    
    if not analyzer.connect_db():
        print("ERROR: Could not connect to database")
        return
    
    print("Connected to database successfully")
    
    # Create sample logs for testing
    print("\nCreating sample log entries...")
    analyzer.create_sample_logs()
    
    # Analyze database logs
    print("1. Analyzing database logs...")
    db_metrics = analyzer.analyze_database_logs(days_back=7)
    print(f"   Database: {db_metrics.total_entries} entries, {db_metrics.error_rate_pct:.1f}% error rate")
    
    # Analyze file logs
    print("2. Analyzing file logs...")
    file_metrics = analyzer.analyze_file_logs("logs")
    print(f"   Files: {file_metrics.total_entries} entries, {file_metrics.error_rate_pct:.1f}% error rate")
    
    # Generate report
    print("3. Generating analysis report...")
    report = analyzer.generate_analysis_report(db_metrics, file_metrics)
    
    # Save report
    report_filename = f"log_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Summary
    total_entries = db_metrics.total_entries + file_metrics.total_entries
    total_errors = db_metrics.error_count + file_metrics.error_count
    avg_error_rate = ((db_metrics.error_rate_pct + file_metrics.error_rate_pct) / 2) if total_entries > 0 else 0
    
    print(f"\nANALYSIS COMPLETE:")
    print(f"- Total Log Entries: {total_entries:,}")
    print(f"- Total Errors: {total_errors:,}")
    print(f"- Average Error Rate: {avg_error_rate:.2f}%")
    print(f"- Performance Issues: {len(db_metrics.performance_issues + file_metrics.performance_issues)}")
    print(f"- Report saved: {report_filename}")
    
    # Health assessment
    if avg_error_rate < 1:
        print("\nSYSTEM HEALTH: Excellent - logs show optimal performance")
    elif avg_error_rate < 3:
        print("\nSYSTEM HEALTH: Good - minor issues detected")
    elif avg_error_rate < 10:
        print("\nSYSTEM HEALTH: Needs attention - review error patterns")
    else:
        print("\nSYSTEM HEALTH: Critical - immediate investigation required")
    
    # Save metrics
    metrics_data = {
        'analysis_date': datetime.now().isoformat(),
        'database_metrics': {
            'total_entries': int(db_metrics.total_entries),
            'error_count': int(db_metrics.error_count),
            'error_rate_pct': float(db_metrics.error_rate_pct),
            'peak_activity_hour': int(db_metrics.peak_activity_hour)
        },
        'file_metrics': {
            'total_entries': int(file_metrics.total_entries),
            'error_count': int(file_metrics.error_count),
            'error_rate_pct': float(file_metrics.error_rate_pct),
            'peak_activity_hour': int(file_metrics.peak_activity_hour)
        },
        'summary': {
            'total_entries': total_entries,
            'total_errors': total_errors,
            'avg_error_rate': avg_error_rate,
            'health_status': 'excellent' if avg_error_rate < 1 else 'good' if avg_error_rate < 3 else 'warning' if avg_error_rate < 10 else 'critical'
        }
    }
    
    json_filename = f"log_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"- Metrics data saved: {json_filename}")

if __name__ == "__main__":
    main()