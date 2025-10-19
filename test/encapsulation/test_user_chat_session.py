"""
Test script for User and Chat management storage modules.

Tests:
1. User creation, retrieval, update, deletion
2. ChatSession creation, retrieval, listing, deletion
3. ChatMessage creation, retrieval, listing, deletion
4. User isolation (sessions belong to specific users)
5. Cascade deletion (deleting user deletes sessions and messages)
"""

import sys
import uuid
sys.path.insert(0, '.')

from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.core.file_management.storage.user_storage import UserStorageConfig
from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from core.user_management.user import UserStorage
from core.user_management.chat_session import ChatSessionStorage
from core.user_management.chat_message import ChatMessageStorage
import hashlib


def hash_password(password: str) -> str:
    """Simple password hashing for testing"""
    return hashlib.sha256(password.encode()).hexdigest()


def get_db_config():
    """Get PostgreSQL configuration"""
    return PostgreSQLConfig(
        type="postgresql",
        host="localhost",
        port=5555,
        database="rag_test",
        user="postgres",
        password="123"
    )


def test_user_management():
    """Test user management operations"""
    print("\n" + "=" * 80)
    print("测试 1: 用户管理")
    print("=" * 80)

    # Initialize database
    db_config = get_db_config()
    db = PostgreSQLDB(db_config)

    # Create UserStorage
    user_storage_config = UserStorageConfig(relational_db_config=db_config)
    user_storage = UserStorage(user_storage_config)

    # Clean up existing test users
    for username in ["alice_test", "bob_test"]:
        existing_user = user_storage.get_user_by_username(username)
        if existing_user:
            user_storage.delete_user(str(existing_user.id))
            print(f"清理已存在的测试用户: {username}")

    # Test 1: Create users
    print("\n 创建用户...")
    user1_id = user_storage.create_user(
        user_name="alice_test",
        hashed_password=hash_password("password123")
    )
    print(f" 创建用户 1: alice_test (ID: {user1_id})")

    user2_id = user_storage.create_user(
        user_name="bob_test",
        hashed_password=hash_password("password456")
    )
    print(f" 创建用户 2: bob_test (ID: {user2_id})")

    # Test 2: Get user by ID
    print("\n 通过 ID 获取用户...")
    user1 = user_storage.get_user(user1_id)
    assert user1 is not None, "User 1 should exist"
    assert user1.user_name == "alice_test", "Username should match"
    print(f" 获取用户成功: {user1.user_name}")

    # Test 3: Get user by username
    print("\n 通过用户名获取用户...")
    user2 = user_storage.get_user_by_username("bob_test")
    assert user2 is not None, "User 2 should exist"
    assert user2.id == user2_id, "User ID should match"
    print(f" 获取用户成功: {user2.user_name} (ID: {user2.id})")

    # Test 4: List users
    print("\n 列出所有用户...")
    users = user_storage.list_users(limit=10)
    print(f" 找到 {len(users)} 个用户")
    for u in users:
        print(f"   - {u.user_name} (ID: {u.id})")

    # Test 5: Update user
    print("\n 更新用户...")
    success = user_storage.update_user(
        user1_id,
        {"user_name": "alice_updated"}
    )
    assert success, "Update should succeed"
    user1_updated = user_storage.get_user(user1_id)
    assert user1_updated.user_name == "alice_updated", "Username should be updated"
    print(f" 用户更新成功: {user1_updated.user_name}")

    return user1_id, user2_id


def test_chat_session_management(user1_id, user2_id):
    """Test chat session management operations"""
    print("\n" + "=" * 80)
    print("测试 2: 聊天会话管理")
    print("=" * 80)

    # Initialize ChatSessionStorage
    db_config = get_db_config()
    session_storage_config = ChatSessionStorageConfig(relational_db_config=db_config)
    session_storage = ChatSessionStorage(session_storage_config)

    # Test 1: Create sessions
    print("\n 创建聊天会话...")
    session1_id = session_storage.create_session(
        user_id=user1_id,
        name="Alice's Research Session"
    )
    print(f" 创建会话 1: Alice's Research Session (ID: {session1_id})")

    session2_id = session_storage.create_session(
        user_id=user1_id,
        name="Alice's Coding Session"
    )
    print(f" 创建会话 2: Alice's Coding Session (ID: {session2_id})")

    session3_id = session_storage.create_session(
        user_id=user2_id,
        name="Bob's Project Session"
    )
    print(f" 创建会话 3: Bob's Project Session (ID: {session3_id})")

    # Test 2: Get session
    print("\n 获取会话...")
    session1 = session_storage.get_session(session1_id)
    assert session1 is not None, "Session 1 should exist"
    assert session1.name == "Alice's Research Session", "Session name should match"
    print(f" 获取会话成功: {session1.name}")

    # Test 3: List sessions by user
    print("\n 列出用户的所有会话...")
    alice_sessions = session_storage.list_sessions_by_user(user1_id)
    print(f" Alice 的会话: {len(alice_sessions)} 个")
    for s in alice_sessions:
        print(f"   - {s.name} (ID: {s.id})")

    bob_sessions = session_storage.list_sessions_by_user(user2_id)
    print(f" Bob 的会话: {len(bob_sessions)} 个")
    for s in bob_sessions:
        print(f"   - {s.name} (ID: {s.id})")

    # Test 4: Verify session ownership
    print("\n 验证会话所有权...")
    assert session_storage.verify_session_ownership(session1_id, user1_id), "Session 1 should belong to Alice"
    assert not session_storage.verify_session_ownership(session1_id, user2_id), "Session 1 should not belong to Bob"
    print(" 会话所有权验证成功")

    # Test 5: Update session
    print("\n 更新会话...")
    success = session_storage.update_session(
        session1_id,
        {"name": "Alice's Updated Research Session"}
    )
    assert success, "Update should succeed"
    session1_updated = session_storage.get_session(session1_id)
    assert session1_updated.name == "Alice's Updated Research Session", "Session name should be updated"
    print(f" 会话更新成功: {session1_updated.name}")

    return session1_id, session2_id, session3_id


def test_chat_message_management(session1_id: str, session2_id: str):
    """Test chat message management operations"""
    print("\n" + "=" * 80)
    print("测试 3: 聊天消息管理")
    print("=" * 80)

    # Initialize ChatMessageStorage
    db_config = get_db_config()
    message_storage_config = ChatMessageStorageConfig(relational_db_config=db_config)
    message_storage = ChatMessageStorage(message_storage_config)

    # Test 1: Create messages
    print("\n 创建聊天消息...")
    msg1_id = message_storage.create_message(
        session_id=uuid.UUID(session1_id),
        content={
            "role": "user",
            "content": "What is machine learning?",
            "metadata": {"timestamp": "2025-10-15T10:00:00"}
        }
    )
    print(f" 创建消息 1: user message (ID: {msg1_id})")

    msg2_id = message_storage.create_message(
        session_id=uuid.UUID(session1_id),
        content={
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence...",
            "metadata": {"model": "gpt-4", "tokens": 150}
        }
    )
    print(f" 创建消息 2: assistant message (ID: {msg2_id})")

    msg3_id = message_storage.create_message(
        session_id=uuid.UUID(session1_id),
        content={
            "role": "user",
            "content": "Can you give me an example?",
            "metadata": {"timestamp": "2025-10-15T10:01:00"}
        }
    )
    print(f" 创建消息 3: user message (ID: {msg3_id})")

    # Test 2: Get message
    print("\n 获取消息...")
    msg1 = message_storage.get_message(msg1_id)
    assert msg1 is not None, "Message 1 should exist"
    assert msg1.content["role"] == "user", "Message role should match"
    assert msg1.content["content"] == "What is machine learning?", "Message content should match"
    print(f" 获取消息成功: {msg1.content['content'][:50]}...")

    # Test 3: List messages by session
    print("\n 列出会话的所有消息...")
    messages = message_storage.list_messages_by_session(session1_id)
    print(f" 会话 1 的消息: {len(messages)} 条")
    for i, msg in enumerate(messages, 1):
        role = msg.content.get("role", "unknown")
        content = msg.content.get("content", "")[:50]
        print(f"   {i}. [{role}] {content}...")

    # Test 4: Get conversation history
    print("\n 获取对话历史...")
    history = message_storage.get_conversation_history(session1_id, limit=10)
    print(f" 对话历史: {len(history)} 条消息")
    for i, msg in enumerate(history, 1):
        print(f"   {i}. [{msg['role']}] {msg['content'][:50]}...")

    return msg1_id, msg2_id, msg3_id


def test_cascade_deletion(user1_id):
    """Test cascade deletion"""
    print("\n" + "=" * 80)
    print("测试 4: 级联删除")
    print("=" * 80)

    db_config = get_db_config()

    user_storage_config = UserStorageConfig(relational_db_config=db_config)
    user_storage = UserStorage(user_storage_config)

    session_storage_config = ChatSessionStorageConfig(relational_db_config=db_config)
    session_storage = ChatSessionStorage(session_storage_config)

    # Check sessions before deletion
    print("\n 删除前检查...")
    sessions_before = session_storage.list_sessions_by_user(user1_id)
    print(f"   用户 1 的会话数: {len(sessions_before)}")

    # Delete user (should cascade delete sessions and messages)
    print("\n 删除用户...")
    success = user_storage.delete_user(user1_id)
    assert success, "User deletion should succeed"
    print(f" 用户删除成功: {user1_id}")

    # Verify user is deleted
    user1 = user_storage.get_user(user1_id)
    assert user1 is None, "User should be deleted"
    print(" 验证：用户已删除")

    # Verify sessions are deleted (cascade)
    sessions_after = session_storage.list_sessions_by_user(user1_id)
    assert len(sessions_after) == 0, "Sessions should be cascade deleted"
    print(f" 验证：会话已级联删除 (删除前: {len(sessions_before)}, 删除后: {len(sessions_after)})")


if __name__ == "__main__":

    try:
        # Test user management
        user1_id, user2_id = test_user_management()

        # Test chat session management
        session1_id, session2_id, session3_id = test_chat_session_management(user1_id, user2_id)

        # Test chat message management
        msg1_id, msg2_id, msg3_id = test_chat_message_management(session1_id, session2_id)

        # Test cascade deletion
        test_cascade_deletion(user1_id)

        print("\n" + "=" * 80)
        print(" 所有测试通过！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

