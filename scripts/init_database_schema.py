#!/usr/bin/env python3
"""
Database Schema Initialization Script
Initializes the LangGraph trading system database schema.
"""

import asyncio
import asyncpg
import os
from pathlib import Path

# Database connection settings
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/trading_system')

async def init_database_schema():
    """Initialize the database schema."""
    print("Initializing LangGraph Trading System Database Schema")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✓ Database connection successful")
        
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        database_init_dir = project_root / "database" / "init"
        
        # SQL files to execute in order
        sql_files = [
            "01_create_tables.sql",
            "02_functions_and_triggers.sql", 
            "03_optimization_settings.sql"
        ]
        
        for sql_file in sql_files:
            sql_path = database_init_dir / sql_file
            
            if not sql_path.exists():
                print(f"  ✗ SQL file not found: {sql_path}")
                continue
            
            print(f"\n  Executing {sql_file}...")
            
            try:
                # Read and execute SQL file
                with open(sql_path, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as e:
                            # Some statements might fail if already exist, that's OK
                            if "already exists" not in str(e).lower():
                                print(f"    Warning: Statement {i+1} failed: {e}")
                
                print(f"  ✓ {sql_file} executed successfully ({len(statements)} statements)")
                
            except Exception as e:
                print(f"  ✗ Failed to execute {sql_file}: {e}")
        
        # Verify schema creation
        print("\n  Verifying schema creation...")
        
        # Count tables
        table_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        print(f"  ✓ Created {table_count} tables")
        
        # Count functions
        function_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.routines 
            WHERE routine_schema = 'public'
        """)
        print(f"  ✓ Created {function_count} functions")
        
        # Count indexes
        index_count = await conn.fetchval("""
            SELECT COUNT(*) FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        print(f"  ✓ Created {index_count} indexes")
        
        # Count triggers
        trigger_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.triggers 
            WHERE trigger_schema = 'public'
        """)
        print(f"  ✓ Created {trigger_count} triggers")
        
        print("\n" + "=" * 60)
        print("Database schema initialization completed successfully!")
        print(f"Summary: {table_count} tables, {function_count} functions, {index_count} indexes, {trigger_count} triggers")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Database initialization failed: {e}")
        return False

async def main():
    """Main initialization function."""
    success = await init_database_schema()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)