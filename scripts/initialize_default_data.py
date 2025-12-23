"""
Initialize default departments and roles for the RAG-ARC system.

This script should be run once after database migration to set up:
1. Default department
2. System roles (Super Admin, Department Admin, User, Guest)

Usage:
    python scripts/initialize_default_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime

from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import (
    Base,
    Department,
    Role,
    RoleType,
    User,
    UserStatus
)
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from encapsulation.database.relational_db.schema_patches import apply_postgres_schema_patches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_departments(session):
    """Initialize default department structure."""
    logger.info("Initializing departments...")
    
    # Check if default department already exists
    existing_dept = session.query(Department).filter_by(name="Default").first()
    if existing_dept:
        logger.info("Default department already exists, skipping...")
        return existing_dept
    
    # Create default department
    default_dept = Department(
        name="Default",
        description="Default department for new users",
        path="Default",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    session.add(default_dept)
    session.flush()  # Get the ID without committing
    
    logger.info(f"Created default department: {default_dept.id}")
    return default_dept


def initialize_roles(session):
    """Initialize system roles."""
    logger.info("Initializing roles...")
    
    roles_config = [
        {
            "name": "Super Admin",
            "role_type": RoleType.SUPER_ADMIN,
            "description": "Full system access with all permissions",
            "permissions": {
                "files": ["view", "edit", "delete", "share", "manage"],
                "users": ["view", "edit", "delete", "manage"],
                "departments": ["view", "edit", "delete", "manage"],
                "roles": ["view", "edit", "delete", "manage"],
                "audit_logs": ["view"]
            }
        },
        {
            "name": "Department Admin",
            "role_type": RoleType.DEPT_ADMIN,
            "description": "Department administrator with user and file management within department",
            "permissions": {
                "files": ["view", "edit", "delete", "share", "manage"],
                "users": ["view", "edit", "manage"],
                "departments": ["view"],
                "audit_logs": ["view"]
            }
        },
        {
            "name": "User",
            "role_type": RoleType.USER,
            "description": "Regular user with file management permissions",
            "permissions": {
                "files": ["view", "edit", "delete", "share"],
                "users": ["view"],
                "departments": ["view"]
            }
        },
        {
            "name": "Guest",
            "role_type": RoleType.GUEST,
            "description": "Read-only guest access",
            "permissions": {
                "files": ["view"],
                "users": [],
                "departments": []
            }
        }
    ]
    
    created_roles = {}
    
    for role_config in roles_config:
        # Check if role already exists
        existing_role = session.query(Role).filter_by(
            role_type=role_config["role_type"]
        ).first()
        
        if existing_role:
            logger.info(f"Role {role_config['name']} already exists, skipping...")
            created_roles[role_config["role_type"]] = existing_role
            continue
        
        # Create new role
        role = Role(
            name=role_config["name"],
            role_type=role_config["role_type"],
            description=role_config["description"],
            permissions=role_config["permissions"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        session.add(role)
        session.flush()
        
        created_roles[role_config["role_type"]] = role
        logger.info(f"Created role: {role.name} ({role.id})")
    
    return created_roles


def create_admin_user(session, default_dept, admin_role):
    """Create default admin user."""
    logger.info("Creating admin user...")
    
    # Check if admin user already exists
    existing_admin = session.query(User).filter_by(user_name="admin").first()
    if existing_admin:
        logger.info("Admin user already exists, skipping...")
        return existing_admin
    
    # Import password hashing
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Create admin user
    admin_user = User(
        user_name="admin",
        email="admin@rag-arc.local",
        hashed_password=pwd_context.hash("admin123"),  # Default password
        department_id=default_dept.id,
        role_id=admin_role.id,
        full_name="System Administrator",
        status=UserStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    session.add(admin_user)
    session.flush()
    
    logger.info(f"Created admin user: {admin_user.id}")
    logger.warning("⚠️  Default admin password is 'admin123' - PLEASE CHANGE IT!")
    
    return admin_user


def main():
    """Main initialization function."""
    logger.info("=" * 80)
    logger.info("RAG-ARC System Initialization")
    logger.info("=" * 80)
    
    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        config = PostgreSQLConfig()
        db = config.build()
        
        # Create tables if they don't exist
        logger.info("Creating database schema...")
        Base.metadata.create_all(db.engine)
        apply_postgres_schema_patches(db.engine)
        logger.info("Database schema ready")
        
        # Start transaction
        session = db.SessionMaker()
        
        try:
            # Initialize departments
            default_dept = initialize_departments(session)
            
            # Initialize roles
            roles = initialize_roles(session)
            
            # Create admin user
            admin_user = create_admin_user(
                session, 
                default_dept, 
                roles[RoleType.SUPER_ADMIN]
            )
            
            # Commit all changes
            session.commit()
            logger.info("All changes committed successfully")
            
            # Print summary
            logger.info("=" * 80)
            logger.info("Initialization Summary")
            logger.info("=" * 80)
            logger.info(f"Default Department: {default_dept.name} ({default_dept.id})")
            logger.info(f"Roles created: {len(roles)}")
            for role_type, role in roles.items():
                logger.info(f"  - {role.name} ({role_type.value})")
            logger.info(f"Admin User: {admin_user.user_name} ({admin_user.email})")
            logger.info("=" * 80)
            logger.info("✅ Initialization completed successfully!")
            logger.info("=" * 80)
            
            # Important notes
            logger.warning("")
            logger.warning("IMPORTANT NOTES:")
            logger.warning("1. Default admin credentials:")
            logger.warning("   Username: admin")
            logger.warning("   Password: admin123")
            logger.warning("2. PLEASE CHANGE THE ADMIN PASSWORD IMMEDIATELY!")
            logger.warning("3. You can now create additional users through the API")
            logger.warning("")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
