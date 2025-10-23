#!/usr/bin/env python3
"""
Trade Logging and Audit Trail Agent - Task 8.2 Implementation

This agent implements comprehensive trade logging and audit trail capabilities:
- Comprehensive trade logging with complete metadata
- Audit trail for all system decisions and actions
- Trade reconciliation and reporting
- Data backup and recovery procedures
- Regulatory compliance and data retention

Requirements: Requirement 14 (Regulatory Compliance and Reporting)
Task: 8.2 Trade Logging and Audit Trail
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import shutil
import gzip
import pickle
from collections import defaultdict, deque

# Database imports
import asyncpg
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Float, Integer, Text, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configuration
from config.settings import settings
from config.secure_config import get_api_keys

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for audit trail"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of system actions for audit trail"""
    # Trading actions
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    
    # Signal actions
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_FUSED = "signal_fused"
    SIGNAL_EXECUTED = "signal_executed"
    
    # Risk actions
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    POSITION_LIMIT_ENFORCED = "position_limit_enforced"
    
    # System actions
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    CONFIGURATION_CHANGED = "configuration_changed"
    
    # Data actions
    MARKET_DATA_RECEIVED = "market_data_received"
    NEWS_DATA_RECEIVED = "news_data_received"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    
    # Compliance actions
    WASH_SALE_DETECTED = "wash_sale_detected"
    PDT_VIOLATION_DETECTED = "pdt_violation_detected"
    REGULATORY_REPORT_GENERATED = "regulatory_report_generated"


class EntityType(Enum):
    """Types of entities involved in actions"""
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    SIGNAL = "signal"
    AGENT = "agent"
    SYSTEM = "system"
    USER = "user"
    CONFIGURATION = "configuration"
    MARKET_DATA = "market_data"
    NEWS = "news"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    action_type: ActionType
    entity_type: EntityType
    entity_id: str
    user_id: Optional[str]
    agent_id: Optional[str]
    description: str
    details: Dict[str, Any]
    log_level: LogLevel
    ip_address: Optional[str]
    session_id: Optional[str]
    correlation_id: Optional[str]
    parent_event_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeLog:
    """Trade log entry with complete metadata"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    strategy: str
    agent_id: str
    signal_strength: float
    market_conditions: Dict[str, Any]
    execution_quality: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    compliance_flags: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationReport:
    """Trade reconciliation report"""
    report_id: str
    timestamp: datetime
    broker_positions: Dict[str, Any]
    system_positions: Dict[str, Any]
    discrepancies: List[Dict[str, Any]]
    reconciliation_status: str
    total_market_value: float
    total_unrealized_pnl: float
    summary: Dict[str, Any]


@dataclass
class BackupMetadata:
    """Backup metadata for tracking"""
    backup_id: str
    timestamp: datetime
    backup_type: str
    source_path: str
    destination_path: str
    size_bytes: int
    checksum: str
    compression_ratio: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeReconciler:
    """Trade reconciliation system"""
    
    def __init__(self, trade_logger):
        self.trade_logger = trade_logger
        self.reconciliation_counter = 0
        
    def reconcile_positions(self, broker_positions: Dict[str, Any], 
                          system_positions: Dict[str, Any]) -> ReconciliationReport:
        """Reconcile broker positions with system positions"""
        try:
            self.reconciliation_counter += 1
            report_id = f"RECON_{self.reconciliation_counter:06d}_{int(time.time())}"
            
            discrepancies = []
            total_market_value = 0.0
            total_unrealized_pnl = 0.0
            
            # Compare positions
            all_symbols = set(broker_positions.keys()) | set(system_positions.keys())
            
            for symbol in all_symbols:
                broker_pos = broker_positions.get(symbol, {})
                system_pos = system_positions.get(symbol, {})
                
                # Check for discrepancies
                discrepancy = self._check_position_discrepancy(symbol, broker_pos, system_pos)
                if discrepancy:
                    discrepancies.append(discrepancy)
                
                # Accumulate totals from broker (source of truth)
                if broker_pos:
                    total_market_value += broker_pos.get('market_value', 0)
                    total_unrealized_pnl += broker_pos.get('unrealized_pl', 0)
            
            # Determine reconciliation status
            if not discrepancies:
                status = "RECONCILED"
            elif len(discrepancies) <= 3:
                status = "MINOR_DISCREPANCIES"
            else:
                status = "MAJOR_DISCREPANCIES"
            
            # Create reconciliation report
            report = ReconciliationReport(
                report_id=report_id,
                timestamp=datetime.now(),
                broker_positions=broker_positions,
                system_positions=system_positions,
                discrepancies=discrepancies,
                reconciliation_status=status,
                total_market_value=total_market_value,
                total_unrealized_pnl=total_unrealized_pnl,
                summary={
                    'total_positions': len(broker_positions),
                    'discrepancy_count': len(discrepancies),
                    'reconciliation_time': datetime.now().isoformat()
                }
            )
            
            # Log reconciliation event
            self.trade_logger.log_audit_event(
                action_type=ActionType.RISK_CHECK_PASSED if status == "RECONCILED" else ActionType.RISK_CHECK_FAILED,
                entity_type=EntityType.SYSTEM,
                entity_id=report_id,
                description=f"Position reconciliation completed: {status}",
                details={
                    'discrepancy_count': len(discrepancies),
                    'total_positions': len(broker_positions),
                    'status': status
                },
                log_level=LogLevel.INFO if status == "RECONCILED" else LogLevel.WARNING
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to reconcile positions: {e}")
            raise
    
    def _check_position_discrepancy(self, symbol: str, broker_pos: Dict, system_pos: Dict) -> Optional[Dict]:
        """Check for position discrepancies between broker and system"""
        if not broker_pos and not system_pos:
            return None
        
        if not broker_pos:
            return {
                'symbol': symbol,
                'type': 'MISSING_BROKER_POSITION',
                'system_position': system_pos,
                'severity': 'HIGH'
            }
        
        if not system_pos:
            return {
                'symbol': symbol,
                'type': 'MISSING_SYSTEM_POSITION',
                'broker_position': broker_pos,
                'severity': 'HIGH'
            }
        
        # Check quantity discrepancy
        broker_qty = broker_pos.get('quantity', 0)
        system_qty = system_pos.get('quantity', 0)
        
        if abs(broker_qty - system_qty) > 0.01:  # Allow for small rounding differences
            return {
                'symbol': symbol,
                'type': 'QUANTITY_MISMATCH',
                'broker_quantity': broker_qty,
                'system_quantity': system_qty,
                'difference': broker_qty - system_qty,
                'severity': 'MEDIUM'
            }
        
        # Check other discrepancies
        discrepancies = []
        
        # Check average price
        broker_avg_price = broker_pos.get('avg_entry_price', 0)
        system_avg_price = system_pos.get('avg_entry_price', 0)
        
        if abs(broker_avg_price - system_avg_price) > 0.01:
            discrepancies.append({
                'field': 'avg_entry_price',
                'broker_value': broker_avg_price,
                'system_value': system_avg_price,
                'difference': broker_avg_price - system_avg_price
            })
        
        # Check market value
        broker_market_value = broker_pos.get('market_value', 0)
        system_market_value = system_pos.get('market_value', 0)
        
        if abs(broker_market_value - system_market_value) > 1.0:  # $1 tolerance
            discrepancies.append({
                'field': 'market_value',
                'broker_value': broker_market_value,
                'system_value': system_market_value,
                'difference': broker_market_value - system_market_value
            })
        
        if discrepancies:
            return {
                'symbol': symbol,
                'type': 'DATA_MISMATCH',
                'discrepancies': discrepancies,
                'severity': 'LOW'
            }
        
        return None


class BackupRecoveryManager:
    """Data backup and recovery management"""
    
    def __init__(self, backup_directory: str = "backups"):
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(exist_ok=True)
        self.backup_counter = 0
        
        # Create backup metadata file
        self.metadata_file = self.backup_directory / "backup_metadata.json"
        self.backup_metadata = self._load_backup_metadata()
        
        logger.info(f"Backup Recovery Manager initialized: {self.backup_directory}")
    
    def _load_backup_metadata(self) -> Dict[str, BackupMetadata]:
        """Load existing backup metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {
                        backup_id: BackupMetadata(**metadata_data)
                        for backup_id, metadata_data in data.items()
                    }
            return {}
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return {}
    
    def _save_backup_metadata(self):
        """Save backup metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    backup_id: asdict(metadata)
                    for backup_id, metadata in self.backup_metadata.items()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    def create_backup(self, source_path: str, backup_type: str = "manual") -> str:
        """Create a backup of the specified source"""
        try:
            self.backup_counter += 1
            backup_id = f"BACKUP_{self.backup_counter:06d}_{int(time.time())}"
            
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {source_path}")
            
            # Create backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{source_path.name}_{timestamp}.backup"
            backup_path = self.backup_directory / backup_filename
            
            # Create backup
            if source_path.is_file():
                backup_path = self._backup_file(source_path, backup_path)
            elif source_path.is_dir():
                backup_path = self._backup_directory(source_path, backup_path)
            else:
                raise ValueError(f"Unsupported source type: {source_path}")
            
            # Calculate backup metadata
            size_bytes = backup_path.stat().st_size
            checksum = self._calculate_checksum(backup_path)
            compression_ratio = self._calculate_compression_ratio(source_path, backup_path)
            
            # Create backup metadata
            backup_metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type=backup_type,
                source_path=str(source_path),
                destination_path=str(backup_path),
                size_bytes=size_bytes,
                checksum=checksum,
                compression_ratio=compression_ratio,
                status="COMPLETED",
                metadata={
                    'source_size': source_path.stat().st_size if source_path.exists() else 0,
                    'backup_method': 'gzip' if backup_path.suffix == '.gz' else 'copy'
                }
            )
            
            # Store metadata
            self.backup_metadata[backup_id] = backup_metadata
            self._save_backup_metadata()
            
            logger.info(f"Backup created successfully: {backup_id} -> {backup_path}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _backup_file(self, source_path: Path, backup_path: Path) -> Path:
        """Backup a single file with compression"""
        try:
            # Use gzip compression for better storage efficiency
            compressed_path = backup_path.with_suffix('.gz')
            
            with open(source_path, 'rb') as source_file:
                with gzip.open(compressed_path, 'wb') as backup_file:
                    shutil.copyfileobj(source_file, backup_file)
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"Failed to backup file: {e}")
            raise
    
    def _backup_directory(self, source_path: Path, backup_path: Path) -> Path:
        """Backup a directory with compression"""
        try:
            # Create tar.gz archive
            archive_path = backup_path.with_suffix('.tar.gz')
            
            shutil.make_archive(
                str(archive_path.with_suffix('')),  # Remove .tar.gz extension
                'gztar',
                source_path
            )
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to backup directory: {e}")
            raise
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    def _calculate_compression_ratio(self, source_path: Path, backup_path: Path) -> float:
        """Calculate compression ratio"""
        try:
            source_size = source_path.stat().st_size
            backup_size = backup_path.stat().st_size
            
            if source_size > 0:
                return backup_size / source_size
            return 1.0
        except Exception as e:
            logger.error(f"Failed to calculate compression ratio: {e}")
            return 1.0
    
    def restore_backup(self, backup_id: str, destination_path: str) -> bool:
        """Restore a backup to the specified destination"""
        try:
            if backup_id not in self.backup_metadata:
                raise ValueError(f"Backup ID not found: {backup_id}")
            
            backup_metadata = self.backup_metadata[backup_id]
            backup_path = Path(backup_metadata.destination_path)
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            destination_path = Path(destination_path)
            
            # Restore based on backup type
            if backup_path.suffix == '.gz':
                self._restore_compressed_file(backup_path, destination_path)
            elif backup_path.suffix == '.tar.gz':
                self._restore_compressed_directory(backup_path, destination_path)
            else:
                shutil.copy2(backup_path, destination_path)
            
            # Verify restoration
            if not destination_path.exists():
                raise RuntimeError("Restoration failed: destination does not exist")
            
            # Update metadata
            backup_metadata.status = "RESTORED"
            backup_metadata.metadata['restored_at'] = datetime.now().isoformat()
            backup_metadata.metadata['restored_to'] = str(destination_path)
            self._save_backup_metadata()
            
            logger.info(f"Backup restored successfully: {backup_id} -> {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _restore_compressed_file(self, backup_path: Path, destination_path: Path):
        """Restore a compressed file"""
        with gzip.open(backup_path, 'rb') as backup_file:
            with open(destination_path, 'wb') as dest_file:
                shutil.copyfileobj(backup_file, dest_file)
    
    def _restore_compressed_directory(self, backup_path: Path, destination_path: Path):
        """Restore a compressed directory"""
        shutil.unpack_archive(backup_path, destination_path, 'gztar')
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        return list(self.backup_metadata.values())
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get information about a specific backup"""
        return self.backup_metadata.get(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        try:
            if backup_id not in self.backup_metadata:
                raise ValueError(f"Backup ID not found: {backup_id}")
            
            backup_metadata = self.backup_metadata[backup_id]
            backup_path = Path(backup_metadata.destination_path)
            
            # Delete backup file
            if backup_path.exists():
                backup_path.unlink()
            
            # Remove from metadata
            del self.backup_metadata[backup_id]
            self._save_backup_metadata()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Clean up backups older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0
            
            backup_ids_to_delete = []
            for backup_id, metadata in self.backup_metadata.items():
                if metadata.timestamp < cutoff_date:
                    backup_ids_to_delete.append(backup_id)
            
            for backup_id in backup_ids_to_delete:
                if self.delete_backup(backup_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0


class TradeLogger:
    """Comprehensive trade logging system"""
    
    def __init__(self, db_connection_string: str = None):
        self.db_connection_string = db_connection_string or settings.DATABASE_URL
        self.engine = None
        self.session_factory = None
        self.trade_counter = 0
        self.log_buffer = deque(maxlen=10000)
        
        # Initialize database
        self._init_database()
        
        logger.info("Trade Logger initialized")
    
    def _init_database(self):
        """Initialize database connection and tables"""
        try:
            # Create engine
            self.engine = create_engine(self.db_connection_string)
            
            # Create session factory
            Session = sessionmaker(bind=self.engine)
            self.session_factory = Session
            
            # Create tables if they don't exist
            self._create_tables()
            
            logger.info("Database connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fallback to SQLite for local development
            self._init_sqlite_fallback()
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite fallback for local development"""
        try:
            self.db_path = Path("trade_logs.db")
            self.engine = None
            self.session_factory = None
            
            # Create SQLite connection
            self.conn = sqlite3.connect(str(self.db_path))
            self._create_sqlite_tables()
            
            logger.info("SQLite fallback database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite fallback: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables for trade logging"""
        try:
            metadata = MetaData()
            
            # Trade logs table
            Table('trade_logs', metadata,
                Column('trade_id', String, primary_key=True),
                Column('order_id', String, nullable=False),
                Column('symbol', String, nullable=False),
                Column('side', String, nullable=False),
                Column('quantity', Float, nullable=False),
                Column('price', Float, nullable=False),
                Column('timestamp', DateTime, nullable=False),
                Column('strategy', String, nullable=False),
                Column('agent_id', String, nullable=False),
                Column('signal_strength', Float),
                Column('market_conditions', Text),
                Column('execution_quality', Text),
                Column('risk_metrics', Text),
                Column('compliance_flags', Text),
                Column('metadata', Text)
            )
            
            # Audit trail table
            Table('audit_trail', metadata,
                Column('event_id', String, primary_key=True),
                Column('timestamp', DateTime, nullable=False),
                Column('action_type', String, nullable=False),
                Column('entity_type', String, nullable=False),
                Column('entity_id', String, nullable=False),
                Column('user_id', String),
                Column('agent_id', String),
                Column('description', Text, nullable=False),
                Column('details', Text),
                Column('log_level', String, nullable=False),
                Column('ip_address', String),
                Column('session_id', String),
                Column('correlation_id', String),
                Column('parent_event_id', String),
                Column('metadata', Text)
            )
            
            # Create tables
            metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            if hasattr(self, 'conn'):
                self._create_sqlite_tables()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables for trade logging"""
        try:
            cursor = self.conn.cursor()
            
            # Trade logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_logs (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    signal_strength REAL,
                    market_conditions TEXT,
                    execution_quality TEXT,
                    risk_metrics TEXT,
                    compliance_flags TEXT,
                    metadata TEXT
                )
            ''')
            
            # Audit trail table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    description TEXT NOT NULL,
                    details TEXT,
                    log_level TEXT NOT NULL,
                    ip_address TEXT,
                    session_id TEXT,
                    correlation_id TEXT,
                    parent_event_id TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_logs_timestamp ON trade_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol ON trade_logs(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_trail_action_type ON audit_trail(action_type)')
            
            self.conn.commit()
            logger.info("SQLite tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create SQLite tables: {e}")
            raise
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution with complete metadata"""
        try:
            # Create trade log entry
            trade_log = TradeLog(
                trade_id=trade_data.get('trade_id', f"TRADE_{int(time.time())}"),
                order_id=trade_data.get('order_id', ''),
                symbol=trade_data.get('symbol', ''),
                side=trade_data.get('side', ''),
                quantity=float(trade_data.get('quantity', 0)),
                price=float(trade_data.get('price', 0)),
                timestamp=trade_data.get('timestamp', datetime.now()),
                strategy=trade_data.get('strategy', ''),
                agent_id=trade_data.get('agent_id', ''),
                signal_strength=float(trade_data.get('signal_strength', 0)),
                market_conditions=trade_data.get('market_conditions', {}),
                execution_quality=trade_data.get('execution_quality', {}),
                risk_metrics=trade_data.get('risk_metrics', {}),
                compliance_flags=trade_data.get('compliance_flags', []),
                metadata=trade_data.get('metadata', {})
            )
            
            # Serialize for storage
            serialized_trade = {
                'trade_id': trade_log.trade_id,
                'order_id': trade_log.order_id,
                'symbol': trade_log.symbol,
                'side': trade_log.side,
                'quantity': trade_log.quantity,
                'price': trade_log.price,
                'timestamp': self._serialize_for_json(trade_log.timestamp),
                'strategy': trade_log.strategy,
                'signal_strength': trade_log.signal_strength,
                'market_conditions': self._serialize_for_json(trade_log.market_conditions),
                'execution_quality': self._serialize_for_json(trade_log.execution_quality),
                'risk_metrics': self._serialize_for_json(trade_log.risk_metrics),
                'compliance_flags': trade_log.compliance_flags,
                'metadata': self._serialize_for_json(trade_log.metadata)
            }
            
            # Store in database
            if self.conn and not self.conn.closed:
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO trade_logs (
                        trade_id, order_id, symbol, side, quantity, price, timestamp,
                        strategy, signal_strength, market_conditions, execution_quality,
                        risk_metrics, compliance_flags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    serialized_trade['trade_id'],
                    serialized_trade['order_id'],
                    serialized_trade['symbol'],
                    serialized_trade['side'],
                    serialized_trade['quantity'],
                    serialized_trade['price'],
                    serialized_trade['timestamp'],
                    serialized_trade['strategy'],
                    serialized_trade['signal_strength'],
                    json.dumps(serialized_trade['market_conditions']),
                    json.dumps(serialized_trade['execution_quality']),
                    json.dumps(serialized_trade['risk_metrics']),
                    json.dumps(serialized_trade['compliance_flags']),
                    json.dumps(serialized_trade['metadata'])
                ))
                self.conn.commit()
                cursor.close()
            
            # Store in memory
            self.log_buffer.append(trade_log)
            
            # Keep only recent history
            if len(self.log_buffer) > self.max_history_size:
                self.log_buffer = self.log_buffer[-self.max_history_size:]
            
            logger.info(f"Trade logged: {trade_log.trade_id} - {trade_log.side} {trade_log.quantity} {trade_log.symbol} @ {trade_log.price}")
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            # Don't raise the error to prevent system crashes
    
    def _store_trade_log(self, trade_log: TradeLog):
        """Store trade log in database"""
        try:
            if hasattr(self, 'conn'):
                # SQLite storage
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_log.trade_id,
                    trade_log.order_id,
                    trade_log.symbol,
                    trade_log.side,
                    trade_log.quantity,
                    trade_log.price,
                    trade_log.timestamp.isoformat(),
                    trade_log.strategy,
                    trade_log.agent_id,
                    trade_log.signal_strength,
                    json.dumps(trade_log.market_conditions),
                    json.dumps(trade_log.execution_quality),
                    json.dumps(trade_log.risk_metrics),
                    json.dumps(trade_log.compliance_flags),
                    json.dumps(trade_log.metadata)
                ))
                self.conn.commit()
            else:
                # PostgreSQL storage
                with self.session_factory() as session:
                    # Convert to dict and store
                    trade_dict = asdict(trade_log)
                    # Convert datetime to string for JSON serialization
                    trade_dict['timestamp'] = trade_log.timestamp.isoformat()
                    
                    # Insert using raw SQL for simplicity
                    session.execute(text('''
                        INSERT INTO trade_logs VALUES (
                            :trade_id, :order_id, :symbol, :side, :quantity, :price,
                            :timestamp, :strategy, :agent_id, :signal_strength,
                            :market_conditions, :execution_quality, :risk_metrics,
                            :compliance_flags, :metadata
                        )
                    '''), trade_dict)
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store trade log: {e}")
            raise
    
    def log_audit_event(self, action_type: ActionType, entity_type: EntityType, 
                       entity_id: str, description: str, details: Dict[str, Any] = None,
                       log_level: LogLevel = LogLevel.INFO, user_id: str = None,
                       agent_id: str = None, correlation_id: str = None,
                       parent_event_id: str = None) -> str:
        """Log an audit event"""
        try:
            # Generate event ID
            event_id = f"EVENT_{int(time.time())}_{hash(description) % 10000:04d}"
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=user_id,
                agent_id=agent_id,
                description=description,
                details=details or {},
                log_level=log_level,
                ip_address=None,  # Would be set in web context
                session_id=None,   # Would be set in web context
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
                metadata={}
            )
            
            # Store in database
            self._store_audit_event(audit_event)
            
            # Log to standard logger
            log_method = getattr(logger, log_level.value)
            log_method(f"AUDIT: {action_type.value} - {description}")
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    def _serialize_for_json(self, obj):
        """Serialize objects for JSON storage, handling datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return {k: self._serialize_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        else:
            return obj
    
    def _store_audit_event(self, audit_event: AuditEvent):
        """Store audit event in database"""
        try:
            # Check if SQLite connection is available
            if not hasattr(self, 'conn') or self.conn is None:
                logger.warning("Database connection not available, skipping audit event storage")
                return
            
            # For SQLite, check if connection is valid
            try:
                cursor = self.conn.cursor()
            except Exception as e:
                logger.warning(f"Database connection not available: {e}, skipping audit event storage")
                return
            
            # Serialize details and metadata for JSON storage
            serialized_details = self._serialize_for_json(audit_event.details)
            serialized_metadata = self._serialize_for_json(audit_event.metadata)
            
            cursor.execute("""
                INSERT INTO audit_trail VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_event.event_id,
                audit_event.timestamp.isoformat(),
                audit_event.action_type.value,
                audit_event.entity_type.value,
                audit_event.entity_id,
                audit_event.user_id,
                audit_event.agent_id,
                audit_event.description,
                json.dumps(serialized_details),
                audit_event.log_level.value,
                audit_event.ip_address,
                audit_event.session_id,
                audit_event.correlation_id,
                audit_event.parent_event_id,
                json.dumps(serialized_metadata)
            ))
            
            self.conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            # Don't raise the error to prevent system crashes
    
    def get_trade_history(self, symbol: str = None, start_date: datetime = None, 
                         end_date: datetime = None, limit: int = 1000) -> List[TradeLog]:
        """Get trade history with filters"""
        try:
            if hasattr(self, 'conn'):
                # SQLite query
                cursor = self.conn.cursor()
                
                query = "SELECT * FROM trade_logs WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to TradeLog objects
                trade_logs = []
                for row in rows:
                    trade_log = TradeLog(
                        trade_id=row[0],
                        order_id=row[1],
                        symbol=row[2],
                        side=row[3],
                        quantity=row[4],
                        price=row[5],
                        timestamp=datetime.fromisoformat(row[6]),
                        strategy=row[7],
                        agent_id=row[8],
                        signal_strength=row[9],
                        market_conditions=json.loads(row[10]) if row[10] else {},
                        execution_quality=json.loads(row[11]) if row[11] else {},
                        risk_metrics=json.loads(row[12]) if row[12] else {},
                        compliance_flags=json.loads(row[13]) if row[13] else [],
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    trade_logs.append(trade_log)
                
                return trade_logs
            else:
                # PostgreSQL query
                with self.session_factory() as session:
                    query = "SELECT * FROM trade_logs WHERE 1=1"
                    params = {}
                    
                    if symbol:
                        query += " AND symbol = :symbol"
                        params['symbol'] = symbol
                    
                    if start_date:
                        query += " AND timestamp >= :start_date"
                        params['start_date'] = start_date
                    
                    if end_date:
                        query += " AND timestamp <= :end_date"
                        params['end_date'] = end_date
                    
                    query += " ORDER BY timestamp DESC LIMIT :limit"
                    params['limit'] = limit
                    
                    result = session.execute(text(query), params)
                    rows = result.fetchall()
                    
                    # Convert to TradeLog objects (simplified for now)
                    trade_logs = []
                    for row in rows:
                        # This would need proper column mapping
                        pass
                    
                    return trade_logs
                    
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def get_audit_trail(self, action_type: ActionType = None, entity_type: EntityType = None,
                        entity_id: str = None, start_date: datetime = None,
                        end_date: datetime = None, limit: int = 1000) -> List[AuditEvent]:
        """Get audit trail with filters"""
        try:
            if hasattr(self, 'conn'):
                # SQLite query
                cursor = self.conn.cursor()
                
                query = "SELECT * FROM audit_trail WHERE 1=1"
                params = []
                
                if action_type:
                    query += " AND action_type = ?"
                    params.append(action_type.value)
                
                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type.value)
                
                if entity_id:
                    query += " AND entity_id = ?"
                    params.append(entity_id)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to AuditEvent objects
                audit_events = []
                for row in rows:
                    audit_event = AuditEvent(
                        event_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        action_type=ActionType(row[2]),
                        entity_type=EntityType(row[3]),
                        entity_id=row[4],
                        user_id=row[5],
                        agent_id=row[6],
                        description=row[7],
                        details=json.loads(row[8]) if row[8] else {},
                        log_level=LogLevel(row[9]),
                        ip_address=row[10],
                        session_id=row[11],
                        correlation_id=row[12],
                        parent_event_id=row[13],
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    audit_events.append(audit_event)
                
                return audit_events
            else:
                # PostgreSQL query (simplified for now)
                return []
                
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return []
    
    def generate_trade_report(self, start_date: datetime, end_date: datetime,
                            symbols: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade report"""
        try:
            # Get trade history for period
            trades = self.get_trade_history(start_date=start_date, end_date=end_date)
            
            if not trades:
                return {
                    'period': f"{start_date.date()} to {end_date.date()}",
                    'total_trades': 0,
                    'summary': {},
                    'by_symbol': {},
                    'by_strategy': {},
                    'compliance_summary': {}
                }
            
            # Filter by symbols if specified
            if symbols:
                trades = [t for t in trades if t.symbol in symbols]
            
            # Calculate summary statistics
            total_trades = len(trades)
            total_volume = sum(t.quantity for t in trades)
            total_value = sum(t.quantity * t.price for t in trades)
            
            # Calculate P&L (simplified - would need more sophisticated logic)
            pnl = 0.0
            for trade in trades:
                if trade.side.lower() == 'buy':
                    pnl -= trade.quantity * trade.price
                else:
                    pnl += trade.quantity * trade.price
            
            # Group by symbol
            by_symbol = defaultdict(lambda: {
                'trades': 0,
                'volume': 0.0,
                'value': 0.0,
                'pnl': 0.0
            })
            
            for trade in trades:
                symbol_stats = by_symbol[trade.symbol]
                symbol_stats['trades'] += 1
                symbol_stats['volume'] += trade.quantity
                symbol_stats['value'] += trade.quantity * trade.price
                
                if trade.side.lower() == 'buy':
                    symbol_stats['pnl'] -= trade.quantity * trade.price
                else:
                    symbol_stats['pnl'] += trade.quantity * trade.price
            
            # Group by strategy
            by_strategy = defaultdict(lambda: {
                'trades': 0,
                'volume': 0.0,
                'value': 0.0
            })
            
            for trade in trades:
                strategy_stats = by_strategy[trade.strategy]
                strategy_stats['trades'] += 1
                strategy_stats['volume'] += trade.quantity
                strategy_stats['value'] += trade.quantity * trade.price
            
            # Compliance summary
            compliance_flags = []
            for trade in trades:
                compliance_flags.extend(trade.compliance_flags)
            
            compliance_summary = {
                'total_flags': len(compliance_flags),
                'unique_flags': list(set(compliance_flags)),
                'flag_counts': {flag: compliance_flags.count(flag) for flag in set(compliance_flags)}
            }
            
            return {
                'period': f"{start_date.date()} to {end_date.date()}",
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_value': total_value,
                'total_pnl': pnl,
                'summary': {
                    'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
                    'avg_trade_value': total_value / total_trades if total_trades > 0 else 0,
                    'trades_per_day': total_trades / max(1, (end_date - start_date).days)
                },
                'by_symbol': dict(by_symbol),
                'by_strategy': dict(by_strategy),
                'compliance_summary': compliance_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trade report: {e}")
            return {
                'error': str(e),
                'period': f"{start_date.date()} to {end_date.date()}"
            }
    
    def close(self):
        """Close database connections"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            if hasattr(self, 'engine'):
                self.engine.dispose()
            logger.info("Trade Logger connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


class TradeLoggingAuditAgent:
    """
    Main trade logging and audit trail agent
    
    This agent provides comprehensive trade logging, audit trails, reconciliation,
    and backup/recovery capabilities for regulatory compliance and operational integrity.
    """
    
    def __init__(self, db_connection_string: str = None, backup_directory: str = "backups"):
        # Initialize components
        self.trade_logger = TradeLogger(db_connection_string)
        self.reconciler = TradeReconciler(self.trade_logger)
        self.backup_manager = BackupRecoveryManager(backup_directory)
        
        # System state
        self.running = False
        self.start_time = datetime.now()
        
        logger.info("Trade Logging and Audit Agent initialized")
    
    async def start(self):
        """Start the trade logging and audit agent"""
        if self.running:
            logger.warning("Agent already running")
            return
        
        self.running = True
        
        # Log system startup
        self.trade_logger.log_audit_event(
            action_type=ActionType.SYSTEM_STARTUP,
            entity_type=EntityType.SYSTEM,
            entity_id="SYSTEM",
            description="Trade Logging and Audit Agent started",
            details={'start_time': self.start_time.isoformat()},
            log_level=LogLevel.INFO
        )
        
        logger.info("Trade Logging and Audit Agent started")
    
    def stop(self):
        """Stop the trade logging and audit agent"""
        try:
            logger.info("Stopping Trade Logging and Audit Agent")
            
            # Stop performance monitoring
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                try:
                    asyncio.create_task(self.performance_monitor.stop_monitoring())
                except Exception as e:
                    logger.warning(f"Error stopping performance monitor: {e}")
            
            # Close database connections safely
            if hasattr(self, 'conn') and self.conn:
                try:
                    # For SQLite, just close the connection
                    self.conn.close()
                    logger.info("Database connection closed")
                except Exception as e:
                    logger.warning(f"Error closing database connection: {e}")
            
            if hasattr(self, 'session_factory') and self.session_factory:
                try:
                    # Close any open sessions
                    pass
                except Exception as e:
                    logger.warning(f"Error closing session factory: {e}")
            
            # Log shutdown
            try:
                self.trade_logger.log_audit_event(
                    action_type=ActionType.SYSTEM_SHUTDOWN,
                    entity_type=EntityType.SYSTEM,
                    entity_id="TRADE_LOGGING",
                    description="Trade Logging and Audit Agent stopped",
                    details={'shutdown_time': datetime.now().isoformat()},
                    log_level=LogLevel.INFO
                )
            except Exception as e:
                logger.warning(f"Could not log shutdown event: {e}")
            
            logger.info("Trade Logging and Audit Agent stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Trade Logging and Audit Agent: {e}")
            # Don't raise the error to prevent system crashes
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """Log a trade with complete metadata"""
        return self.trade_logger.log_trade(trade_data)
    
    def log_audit_event(self, action_type: ActionType, entity_type: EntityType,
                       entity_id: str, description: str, details: Dict[str, Any] = None,
                       log_level: LogLevel = LogLevel.INFO, user_id: str = None,
                       agent_id: str = None, correlation_id: str = None,
                       parent_event_id: str = None) -> str:
        """Log an audit event"""
        return self.trade_logger.log_audit_event(
            action_type, entity_type, entity_id, description, details,
            log_level, user_id, agent_id, correlation_id, parent_event_id
        )
    
    def reconcile_positions(self, broker_positions: Dict[str, Any], 
                          system_positions: Dict[str, Any]) -> ReconciliationReport:
        """Reconcile broker positions with system positions"""
        return self.reconciler.reconcile_positions(broker_positions, system_positions)
    
    def generate_trade_report(self, start_date: datetime, end_date: datetime,
                            symbols: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade report"""
        return self.trade_logger.generate_trade_report(start_date, end_date, symbols)
    
    def get_trade_history(self, symbol: str = None, start_date: datetime = None,
                         end_date: datetime = None, limit: int = 1000) -> List[TradeLog]:
        """Get trade history with filters"""
        return self.trade_logger.get_trade_history(symbol, start_date, end_date, limit)
    
    def get_audit_trail(self, action_type: ActionType = None, entity_type: EntityType = None,
                        entity_id: str = None, start_date: datetime = None,
                        end_date: datetime = None, limit: int = 1000) -> List[AuditEvent]:
        """Get audit trail with filters"""
        return self.trade_logger.get_audit_trail(action_type, entity_type, entity_id, start_date, end_date, limit)
    
    def create_backup(self, source_path: str, backup_type: str = "manual") -> str:
        """Create a backup of the specified source"""
        return self.backup_manager.create_backup(source_path, backup_type)
    
    def restore_backup(self, backup_id: str, destination_path: str) -> bool:
        """Restore a backup to the specified destination"""
        return self.backup_manager.restore_backup(backup_id, destination_path)
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        return self.backup_manager.list_backups()
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get information about a specific backup"""
        return self.backup_manager.get_backup_info(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        return self.backup_manager.delete_backup(backup_id)
    
    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Clean up backups older than specified days"""
        return self.backup_manager.cleanup_old_backups(max_age_days)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        try:
            # Get recent audit events
            recent_events = self.get_audit_trail(limit=100)
            
            # Get recent trades
            recent_trades = self.get_trade_history(limit=100)
            
            # Get backup information
            backups = self.list_backups()
            
            return {
                'status': 'running' if self.running else 'stopped',
                'start_time': self.start_time.isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'recent_audit_events': len(recent_events),
                'recent_trades': len(recent_trades),
                'total_backups': len(backups),
                'last_backup': max([b.timestamp for b in backups]).isoformat() if backups else None,
                'database_status': 'connected' if hasattr(self.trade_logger, 'conn') or hasattr(self.trade_logger, 'engine') else 'disconnected'
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize agent
        agent = TradeLoggingAuditAgent()
        
        # Start agent
        await agent.start()
        
        # Log some sample trades
        sample_trade = {
            'order_id': 'ORD_001',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now(),
            'strategy': 'momentum',
            'agent_id': 'momentum_agent',
            'signal_strength': 0.8,
            'market_conditions': {'volatility': 'high', 'trend': 'up'},
            'execution_quality': {'slippage': 0.001, 'fill_quality': 'good'},
            'risk_metrics': {'var': 0.02, 'position_size': 0.1},
            'compliance_flags': [],
            'metadata': {'market_regime': 'trending'}
        }
        
        trade_id = agent.log_trade(sample_trade)
        print(f"Trade logged with ID: {trade_id}")
        
        # Test reconciliation
        broker_positions = {
            'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pl': 500},
            'TSLA': {'quantity': 50, 'market_value': 10000, 'unrealized_pl': -250}
        }
        
        system_positions = {
            'AAPL': {'quantity': 100, 'market_value': 15000, 'unrealized_pl': 500},
            'TSLA': {'quantity': 50, 'market_value': 10000, 'unrealized_pl': -250}
        }
        
        reconciliation = agent.reconcile_positions(broker_positions, system_positions)
        print(f"Reconciliation status: {reconciliation.reconciliation_status}")
        
        # Test backup
        backup_id = agent.create_backup("trade_logs.db", "test")
        print(f"Backup created with ID: {backup_id}")
        
        # Get system status
        status = agent.get_system_status()
        print(f"System status: {status}")
        
        # Stop agent
        agent.stop()
        
        print("Trade Logging and Audit Agent demo completed!")
    
    # Run the demo
    asyncio.run(main()) 