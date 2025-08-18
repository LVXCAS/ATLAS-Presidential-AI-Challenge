#!/usr/bin/env python3
"""
Database Schema Validation Script
Validates the comprehensive database schema implementation for the LangGraph trading system.
"""

import asyncio
import asyncpg
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSchemaValidator:
    """Validates the database schema implementation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.validation_results = {
            'tables': {},
            'indexes': {},
            'functions': {},
            'triggers': {},
            'hypertables': {},
            'retention_policies': {},
            'performance_tests': {},
            'data_integrity_tests': {}
        }
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Run comprehensive schema validation."""
        logger.info("Starting database schema validation...")
        
        try:
            # Connect to database
            await self.db_manager.connect()
            
            # Run validation tests
            await self.validate_tables()
            await self.validate_indexes()
            await self.validate_functions()
            await self.validate_triggers()
            await self.validate_hypertables()
            await self.validate_retention_policies()
            await self.test_data_insertion()
            await self.test_query_performance()
            await self.test_data_integrity()
            
            # Generate summary
            summary = self.generate_validation_summary()
            
            logger.info("Database schema validation completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        finally:
            await self.db_manager.disconnect()
    
    async def validate_tables(self):
        """Validate all required tables exist with correct structure."""
        logger.info("Validating database tables...")
        
        expected_tables = [
            'market_data_hf', 'market_data_daily', 'options_data', 'forex_crypto_data',
            'signals', 'fused_signals', 'signal_performance',
            'orders', 'trades', 'positions', 'portfolio_snapshots',
            'model_performance', 'risk_metrics', 'risk_alerts', 'factor_exposures',
            'news_articles', 'social_sentiment', 'alternative_data',
            'system_logs', 'agent_metrics', 'audit_trail',
            'backtest_runs', 'backtest_trades'
        ]
        
        # Check table existence
        query = """
        SELECT table_name, table_type
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = ANY($1)
        ORDER BY table_name
        """
        
        existing_tables = await self.db_manager.fetch_all(query, expected_tables)
        existing_table_names = {row['table_name'] for row in existing_tables}
        
        # Validate each table
        for table_name in expected_tables:
            if table_name in existing_table_names:
                # Check table structure
                columns = await self.get_table_columns(table_name)
                self.validation_results['tables'][table_name] = {
                    'exists': True,
                    'columns': len(columns),
                    'column_details': columns
                }
                logger.info(f"✓ Table {table_name} exists with {len(columns)} columns")
            else:
                self.validation_results['tables'][table_name] = {
                    'exists': False,
                    'error': 'Table not found'
                }
                logger.error(f"✗ Table {table_name} not found")
    
    async def get_table_columns(self, table_name: str) -> List[Dict]:
        """Get column information for a table."""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
        """
        return await self.db_manager.fetch_all(query, table_name)
    
    async def validate_indexes(self):
        """Validate all required indexes exist."""
        logger.info("Validating database indexes...")
        
        # Get all indexes
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
        """
        
        indexes = await self.db_manager.fetch_all(query)
        
        # Group by table
        table_indexes = {}
        for idx in indexes:
            table_name = idx['tablename']
            if table_name not in table_indexes:
                table_indexes[table_name] = []
            table_indexes[table_name].append({
                'name': idx['indexname'],
                'definition': idx['indexdef']
            })
        
        self.validation_results['indexes'] = table_indexes
        
        # Log summary
        total_indexes = sum(len(idxs) for idxs in table_indexes.values())
        logger.info(f"✓ Found {total_indexes} indexes across {len(table_indexes)} tables")
    
    async def validate_functions(self):
        """Validate all required functions exist."""
        logger.info("Validating database functions...")
        
        expected_functions = [
            'calculate_portfolio_metrics',
            'calculate_signal_performance',
            'detect_market_data_anomalies',
            'calculate_var',
            'refresh_materialized_views',
            'get_database_performance_metrics',
            'daily_maintenance',
            'weekly_maintenance'
        ]
        
        query = """
        SELECT 
            routine_name,
            routine_type,
            data_type as return_type
        FROM information_schema.routines
        WHERE routine_schema = 'public' AND routine_name = ANY($1)
        ORDER BY routine_name
        """
        
        functions = await self.db_manager.fetch_all(query, expected_functions)
        
        for func in functions:
            self.validation_results['functions'][func['routine_name']] = {
                'exists': True,
                'type': func['routine_type'],
                'return_type': func['return_type']
            }
            logger.info(f"✓ Function {func['routine_name']} exists")
        
        # Check for missing functions
        existing_names = {f['routine_name'] for f in functions}
        for expected in expected_functions:
            if expected not in existing_names:
                self.validation_results['functions'][expected] = {
                    'exists': False,
                    'error': 'Function not found'
                }
                logger.error(f"✗ Function {expected} not found")
    
    async def validate_triggers(self):
        """Validate all required triggers exist."""
        logger.info("Validating database triggers...")
        
        query = """
        SELECT 
            trigger_name,
            event_object_table,
            action_timing,
            event_manipulation
        FROM information_schema.triggers
        WHERE trigger_schema = 'public'
        ORDER BY event_object_table, trigger_name
        """
        
        triggers = await self.db_manager.fetch_all(query)
        
        # Group by table
        table_triggers = {}
        for trigger in triggers:
            table_name = trigger['event_object_table']
            if table_name not in table_triggers:
                table_triggers[table_name] = []
            table_triggers[table_name].append({
                'name': trigger['trigger_name'],
                'timing': trigger['action_timing'],
                'event': trigger['event_manipulation']
            })
        
        self.validation_results['triggers'] = table_triggers
        
        total_triggers = sum(len(trigs) for trigs in table_triggers.values())
        logger.info(f"✓ Found {total_triggers} triggers across {len(table_triggers)} tables")
    
    async def validate_hypertables(self):
        """Validate TimescaleDB hypertables are properly configured."""
        logger.info("Validating TimescaleDB hypertables...")
        
        try:
            # Check if TimescaleDB extension is available
            query = "SELECT * FROM pg_extension WHERE extname = 'timescaledb'"
            extension = await self.db_manager.fetch_one(query)
            
            if not extension:
                logger.warning("TimescaleDB extension not found - skipping hypertable validation")
                return
            
            # Get hypertable information
            query = """
            SELECT 
                hypertable_schema,
                hypertable_name,
                num_dimensions,
                num_chunks,
                compression_enabled,
                replication_factor
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public'
            ORDER BY hypertable_name
            """
            
            hypertables = await self.db_manager.fetch_all(query)
            
            for ht in hypertables:
                self.validation_results['hypertables'][ht['hypertable_name']] = {
                    'exists': True,
                    'dimensions': ht['num_dimensions'],
                    'chunks': ht['num_chunks'],
                    'compression': ht['compression_enabled'],
                    'replication_factor': ht['replication_factor']
                }
                logger.info(f"✓ Hypertable {ht['hypertable_name']} configured with {ht['num_chunks']} chunks")
            
        except Exception as e:
            logger.warning(f"Could not validate hypertables: {e}")
            self.validation_results['hypertables']['error'] = str(e)
    
    async def validate_retention_policies(self):
        """Validate data retention policies are configured."""
        logger.info("Validating data retention policies...")
        
        try:
            query = """
            SELECT 
                hypertable_schema,
                hypertable_name,
                job_id,
                config
            FROM timescaledb_information.jobs
            WHERE proc_name = 'policy_retention'
            ORDER BY hypertable_name
            """
            
            policies = await self.db_manager.fetch_all(query)
            
            for policy in policies:
                table_name = policy['hypertable_name']
                self.validation_results['retention_policies'][table_name] = {
                    'exists': True,
                    'job_id': policy['job_id'],
                    'config': policy['config']
                }
                logger.info(f"✓ Retention policy configured for {table_name}")
            
        except Exception as e:
            logger.warning(f"Could not validate retention policies: {e}")
            self.validation_results['retention_policies']['error'] = str(e)
    
    async def test_data_insertion(self):
        """Test data insertion into key tables."""
        logger.info("Testing data insertion...")
        
        test_data = {
            'market_data_hf': {
                'symbol': 'TEST',
                'exchange': 'NASDAQ',
                'timestamp': datetime.now(),
                'timeframe': '1m',
                'open': Decimal('100.00'),
                'high': Decimal('101.00'),
                'low': Decimal('99.50'),
                'close': Decimal('100.50'),
                'volume': 10000,
                'provider': 'test'
            },
            'signals': {
                'symbol': 'TEST',
                'agent_name': 'test_agent',
                'signal_type': 'momentum',
                'value': Decimal('0.75'),
                'confidence': Decimal('0.85'),
                'strength': Decimal('0.80'),
                'direction': 'BUY',
                'top_3_reasons': json.dumps(['reason1', 'reason2', 'reason3']),
                'timestamp': datetime.now(),
                'model_version': '1.0.0'
            }
        }
        
        for table_name, data in test_data.items():
            try:
                # Build insert query
                columns = list(data.keys())
                placeholders = [f'${i+1}' for i in range(len(columns))]
                values = list(data.values())
                
                query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING id
                """
                
                result = await self.db_manager.fetch_one(query, *values)
                
                if result:
                    # Clean up test data
                    await self.db_manager.execute(f"DELETE FROM {table_name} WHERE id = $1", result['id'])
                    
                    self.validation_results['performance_tests'][f'{table_name}_insert'] = {
                        'success': True,
                        'test_id': result['id']
                    }
                    logger.info(f"✓ Data insertion test passed for {table_name}")
                else:
                    raise Exception("No ID returned from insert")
                    
            except Exception as e:
                self.validation_results['performance_tests'][f'{table_name}_insert'] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"✗ Data insertion test failed for {table_name}: {e}")
    
    async def test_query_performance(self):
        """Test query performance on key tables."""
        logger.info("Testing query performance...")
        
        test_queries = {
            'market_data_recent': """
                SELECT COUNT(*) FROM market_data_hf 
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
            """,
            'signals_by_agent': """
                SELECT agent_name, COUNT(*) 
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY agent_name
            """,
            'portfolio_metrics': """
                SELECT * FROM calculate_portfolio_metrics()
            """,
            'system_health': """
                SELECT * FROM system_health
            """
        }
        
        for test_name, query in test_queries.items():
            try:
                start_time = datetime.now()
                result = await self.db_manager.fetch_all(query)
                end_time = datetime.now()
                
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                self.validation_results['performance_tests'][test_name] = {
                    'success': True,
                    'duration_ms': duration_ms,
                    'row_count': len(result)
                }
                logger.info(f"✓ Query {test_name} completed in {duration_ms:.2f}ms ({len(result)} rows)")
                
            except Exception as e:
                self.validation_results['performance_tests'][test_name] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"✗ Query {test_name} failed: {e}")
    
    async def test_data_integrity(self):
        """Test data integrity constraints and triggers."""
        logger.info("Testing data integrity...")
        
        # Test trade validation trigger
        try:
            invalid_trade = {
                'trade_id': 'TEST_INVALID',
                'order_id': 'TEST_ORDER',
                'symbol': 'TEST',
                'exchange': 'NASDAQ',
                'side': 'BUY',
                'quantity': Decimal('-100'),  # Invalid: negative quantity for BUY
                'price': Decimal('100.00'),
                'executed_at': datetime.now(),
                'strategy': 'test',
                'agent_name': 'test_agent',
                'broker': 'test'
            }
            
            columns = list(invalid_trade.keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            values = list(invalid_trade.values())
            
            query = f"""
            INSERT INTO trades ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            await self.db_manager.execute(query, *values)
            
            # If we get here, the validation failed
            self.validation_results['data_integrity_tests']['trade_validation'] = {
                'success': False,
                'error': 'Invalid trade was allowed'
            }
            logger.error("✗ Trade validation trigger failed - invalid trade was allowed")
            
        except Exception as e:
            # This is expected - the trigger should prevent invalid data
            self.validation_results['data_integrity_tests']['trade_validation'] = {
                'success': True,
                'error_caught': str(e)
            }
            logger.info("✓ Trade validation trigger working - invalid trade rejected")
        
        # Test audit trail trigger
        try:
            # Insert a test position
            test_position = {
                'symbol': 'TEST_AUDIT',
                'exchange': 'NASDAQ',
                'strategy': 'test',
                'agent_name': 'test_agent',
                'quantity': Decimal('100'),
                'avg_cost': Decimal('50.00'),
                'market_value': Decimal('5000.00'),
                'unrealized_pnl': Decimal('0.00'),
                'first_trade_at': datetime.now(),
                'last_trade_at': datetime.now()
            }
            
            columns = list(test_position.keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            values = list(test_position.values())
            
            query = f"""
            INSERT INTO positions ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
            """
            
            result = await self.db_manager.fetch_one(query, *values)
            position_id = result['id']
            
            # Check if audit trail was created
            audit_query = """
            SELECT * FROM audit_trail 
            WHERE entity_type = 'POSITION' AND action_type = 'CREATE'
            ORDER BY timestamp DESC LIMIT 1
            """
            
            audit_record = await self.db_manager.fetch_one(audit_query)
            
            if audit_record:
                self.validation_results['data_integrity_tests']['audit_trail'] = {
                    'success': True,
                    'audit_id': audit_record['id']
                }
                logger.info("✓ Audit trail trigger working - position creation logged")
            else:
                self.validation_results['data_integrity_tests']['audit_trail'] = {
                    'success': False,
                    'error': 'No audit record created'
                }
                logger.error("✗ Audit trail trigger failed - no record created")
            
            # Clean up
            await self.db_manager.execute("DELETE FROM positions WHERE id = $1", position_id)
            
        except Exception as e:
            self.validation_results['data_integrity_tests']['audit_trail'] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"✗ Audit trail test failed: {e}")
    
    def generate_validation_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive validation summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'details': self.validation_results,
            'statistics': {
                'tables_validated': len(self.validation_results['tables']),
                'tables_passed': sum(1 for t in self.validation_results['tables'].values() if t.get('exists', False)),
                'functions_validated': len(self.validation_results['functions']),
                'functions_passed': sum(1 for f in self.validation_results['functions'].values() if f.get('exists', False)),
                'performance_tests': len(self.validation_results['performance_tests']),
                'performance_passed': sum(1 for t in self.validation_results['performance_tests'].values() if t.get('success', False)),
                'integrity_tests': len(self.validation_results['data_integrity_tests']),
                'integrity_passed': sum(1 for t in self.validation_results['data_integrity_tests'].values() if t.get('success', False))
            }
        }
        
        # Determine overall status
        failed_tests = []
        
        # Check table validation
        for table, result in self.validation_results['tables'].items():
            if not result.get('exists', False):
                failed_tests.append(f"Table {table} missing")
        
        # Check function validation
        for func, result in self.validation_results['functions'].items():
            if not result.get('exists', False):
                failed_tests.append(f"Function {func} missing")
        
        # Check performance tests
        for test, result in self.validation_results['performance_tests'].items():
            if not result.get('success', False):
                failed_tests.append(f"Performance test {test} failed")
        
        # Check integrity tests
        for test, result in self.validation_results['data_integrity_tests'].items():
            if not result.get('success', False):
                failed_tests.append(f"Integrity test {test} failed")
        
        if failed_tests:
            summary['overall_status'] = 'FAIL'
            summary['failed_tests'] = failed_tests
        
        return summary

async def main():
    """Main validation function."""
    validator = DatabaseSchemaValidator()
    
    try:
        summary = await validator.validate_schema()
        
        # Print summary
        print("\n" + "="*80)
        print("DATABASE SCHEMA VALIDATION SUMMARY")
        print("="*80)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Timestamp: {summary['timestamp']}")
        print("\nStatistics:")
        for key, value in summary['statistics'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if summary['overall_status'] == 'FAIL':
            print("\nFailed Tests:")
            for test in summary.get('failed_tests', []):
                print(f"  ✗ {test}")
        
        print("\n" + "="*80)
        
        # Save detailed results
        with open('database_validation_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("Detailed results saved to: database_validation_results.json")
        
        return summary['overall_status'] == 'PASS'
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)