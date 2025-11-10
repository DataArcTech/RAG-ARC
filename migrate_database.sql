-- Database Migration Script
-- This script updates the database schema to match the new ORM models
-- Run this script on your PostgreSQL database

-- 1. Add session_metadata and is_shared columns to chat_session table
DO $$
BEGIN
    -- Add is_shared column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chat_session' AND column_name = 'is_shared'
    ) THEN
        ALTER TABLE chat_session ADD COLUMN is_shared BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE 'Added is_shared column to chat_session table';
    END IF;
    
    -- Rename metadata to session_metadata if metadata exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chat_session' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE chat_session RENAME COLUMN metadata TO session_metadata;
        RAISE NOTICE 'Renamed chat_session.metadata to session_metadata';
    -- Add session_metadata column if it doesn't exist
    ELSIF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chat_session' AND column_name = 'session_metadata'
    ) THEN
        ALTER TABLE chat_session ADD COLUMN session_metadata JSON;
        RAISE NOTICE 'Added session_metadata column to chat_session table';
    END IF;
END $$;

-- 2. Add log_metadata column to audit_log table (rename if exists, add if not)
DO $$
BEGIN
    -- Rename metadata to log_metadata if metadata exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'audit_log' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE audit_log RENAME COLUMN metadata TO log_metadata;
        RAISE NOTICE 'Renamed audit_log.metadata to log_metadata';
    -- Add log_metadata column if it doesn't exist
    ELSIF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'audit_log' AND column_name = 'log_metadata'
    ) THEN
        ALTER TABLE audit_log ADD COLUMN log_metadata JSON NOT NULL DEFAULT '{}'::json;
        RAISE NOTICE 'Added log_metadata column to audit_log table';
    END IF;
END $$;

-- 3. Add department_id and role_id columns to user table if they don't exist
DO $$
BEGIN
    -- Add department_id column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user' AND column_name = 'department_id'
    ) THEN
        ALTER TABLE "user" ADD COLUMN department_id UUID;
        CREATE INDEX IF NOT EXISTS idx_user_department_id ON "user"(department_id);
        
        -- Add foreign key constraint if department table exists
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'department') THEN
            ALTER TABLE "user" 
            ADD CONSTRAINT fk_user_department 
            FOREIGN KEY (department_id) 
            REFERENCES department(id);
        END IF;
        
        RAISE NOTICE 'Added department_id column to user table';
    ELSE
        RAISE NOTICE 'Column user.department_id already exists, skipping';
    END IF;
    
    -- Add role_id column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user' AND column_name = 'role_id'
    ) THEN
        ALTER TABLE "user" ADD COLUMN role_id UUID;
        CREATE INDEX IF NOT EXISTS idx_user_role_id ON "user"(role_id);
        
        -- Add foreign key constraint if role table exists
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'role') THEN
            ALTER TABLE "user" 
            ADD CONSTRAINT fk_user_role 
            FOREIGN KEY (role_id) 
            REFERENCES role(id);
        END IF;
        
        RAISE NOTICE 'Added role_id column to user table';
    ELSE
        RAISE NOTICE 'Column user.role_id already exists, skipping';
    END IF;
    
    -- Add status column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user' AND column_name = 'status'
    ) THEN
        ALTER TABLE "user" ADD COLUMN status VARCHAR(255) NOT NULL DEFAULT 'ACTIVE';
        RAISE NOTICE 'Added status column to user table';
    ELSE
        RAISE NOTICE 'Column user.status already exists, skipping';
    END IF;
    
    -- Add last_login_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user' AND column_name = 'last_login_at'
    ) THEN
        ALTER TABLE "user" ADD COLUMN last_login_at TIMESTAMP;
        RAISE NOTICE 'Added last_login_at column to user table';
    ELSE
        RAISE NOTICE 'Column user.last_login_at already exists, skipping';
    END IF;
END $$;

-- 4. Create role table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'role') THEN
        CREATE TABLE role (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX idx_role_name ON role(name);
        RAISE NOTICE 'Created role table';
    ELSE
        RAISE NOTICE 'Table role already exists, skipping';
    END IF;
END $$;

-- 5. Create department table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'department') THEN
        CREATE TABLE department (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            path TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX idx_department_name ON department(name);
        RAISE NOTICE 'Created department table';
    ELSE
        RAISE NOTICE 'Table department already exists, skipping';
    END IF;
END $$;

-- 6. Create file_permission table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'file_permission') THEN
        CREATE TABLE file_permission (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            file_id VARCHAR(255) NOT NULL,
            permission_receiver_type VARCHAR(20) NOT NULL,
            user_id UUID,
            department_id UUID,
            permission_type VARCHAR(20) NOT NULL,
            granted_by UUID NOT NULL,
            granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES file_metadata(file_id),
            FOREIGN KEY (user_id) REFERENCES "user"(id),
            FOREIGN KEY (department_id) REFERENCES department(id),
            FOREIGN KEY (granted_by) REFERENCES "user"(id)
        );
        CREATE INDEX idx_file_permission_file_id ON file_permission(file_id);
        CREATE INDEX idx_file_permission_user_id ON file_permission(user_id);
        CREATE INDEX idx_file_permission_department_id ON file_permission(department_id);
        RAISE NOTICE 'Created file_permission table';
    ELSE
        RAISE NOTICE 'Table file_permission already exists, checking column constraints...';
        -- Ensure user_id and department_id are nullable
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'file_permission' 
            AND column_name = 'user_id' 
            AND is_nullable = 'NO'
        ) THEN
            ALTER TABLE file_permission ALTER COLUMN user_id DROP NOT NULL;
            RAISE NOTICE 'Made user_id nullable in file_permission table';
        END IF;
        
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'file_permission' 
            AND column_name = 'department_id' 
            AND is_nullable = 'NO'
        ) THEN
            ALTER TABLE file_permission ALTER COLUMN department_id DROP NOT NULL;
            RAISE NOTICE 'Made department_id nullable in file_permission table';
        END IF;
    END IF;
END $$;

-- 7. Add source_file_ids column to chat_message table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chat_message' AND column_name = 'source_file_ids'
    ) THEN
        ALTER TABLE chat_message ADD COLUMN source_file_ids JSON;
        RAISE NOTICE 'Added source_file_ids column to chat_message table';
    ELSE
        RAISE NOTICE 'Column chat_message.source_file_ids already exists, skipping';
    END IF;
END $$;

-- Migration completed
DO $$
BEGIN
    RAISE NOTICE 'Migration completed successfully!';
END $$;

