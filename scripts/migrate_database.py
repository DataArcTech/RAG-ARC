"""
Database migration script for RAG-ARC ORM model updates.

This script handles the migration from the old User model to the new enhanced model
with department and role support.

⚠️ WARNING: This is a destructive operation. Make sure to backup your data first!

Usage:
    # For development (drops and recreates all tables)
    python scripts/migrate_database.py --mode=dev
    
    # For production (attempts to preserve data - NOT IMPLEMENTED YET)
    python scripts/migrate_database.py --mode=prod
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime

from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import Base
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def migrate_dev_mode(db: PostgreSQLDB):
    """
    Development mode migration: Drop and recreate all tables.
    
    ⚠️ WARNING: This will DELETE ALL DATA!
    """
    logger.warning("=" * 80)
    logger.warning("DEVELOPMENT MODE MIGRATION")
    logger.warning("=" * 80)
    logger.warning("This will DROP ALL TABLES and recreate them.")
    logger.warning("ALL DATA WILL BE LOST!")
    logger.warning("=" * 80)
    
    if not confirm_action("Are you sure you want to continue?"):
        logger.info("Migration cancelled by user")
        return False
    
    logger.info("Starting development mode migration...")
    
    try:
        # Drop all tables
        logger.info("Dropping all tables...")
        Base.metadata.drop_all(db.engine)
        logger.info("✓ All tables dropped")
        
        # Recreate all tables
        logger.info("Creating new table structure...")
        Base.metadata.create_all(db.engine)
        logger.info("✓ All tables created")
        
        logger.info("=" * 80)
        logger.info("✅ Development mode migration completed successfully!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Run: python scripts/initialize_default_data.py")
        logger.info("2. This will create default departments, roles, and admin user")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        return False


def migrate_prod_mode(db: PostgreSQLDB):
    """
    Production mode migration: Attempt to preserve existing data.
    
    NOT IMPLEMENTED YET - This requires careful data migration logic.
    """
    logger.error("=" * 80)
    logger.error("PRODUCTION MODE MIGRATION - NOT IMPLEMENTED")
    logger.error("=" * 80)
    logger.error("Production mode migration is not yet implemented.")
    logger.error("This requires:")
    logger.error("1. Backup of existing user data")
    logger.error("2. Creation of default department and roles")
    logger.error("3. Migration of existing users to new schema")
    logger.error("4. Careful handling of foreign key constraints")
    logger.error("")
    logger.error("For now, please use development mode or perform manual migration.")
    logger.error("=" * 80)
    return False


def check_database_connection(db: PostgreSQLDB) -> bool:
    """Check if database connection is working."""
    try:
        with db.engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {str(e)}")
        return False


def show_current_schema(db: PostgreSQLDB):
    """Show current database schema."""
    logger.info("Current database tables:")
    try:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        if not tables:
            logger.info("  (No tables found)")
        else:
            for table in tables:
                logger.info(f"  - {table}")
                
    except Exception as e:
        logger.error(f"Could not retrieve schema: {str(e)}")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate RAG-ARC database to new schema"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Migration mode: dev (destructive) or prod (preserves data)"
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip confirmation prompts (use with caution!)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RAG-ARC Database Migration")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info("=" * 80)
    
    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        config = PostgreSQLConfig()
        db = config.build()
        
        logger.info(f"Database: {config.database}")
        logger.info(f"Host: {config.host}:{config.port}")
        logger.info(f"User: {config.user}")
        
        # Check connection
        if not check_database_connection(db):
            logger.error("Cannot proceed without database connection")
            sys.exit(1)
        
        # Show current schema
        logger.info("")
        show_current_schema(db)
        logger.info("")
        
        # Perform migration based on mode
        if args.mode == "dev":
            success = migrate_dev_mode(db)
        else:
            success = migrate_prod_mode(db)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nMigration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

