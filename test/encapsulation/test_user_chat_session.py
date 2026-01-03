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
import os
import pytest

sys.path.insert(0, '.')

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGARC_CHAT_STORAGE_TESTS") != "1",
    reason="Requires PostgreSQL services; set RUN_RAGARC_CHAT_STORAGE_TESTS=1 to run.",
)

from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.core.file_management.storage.user_storage import UserStorageConfig
from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from core.user_management.user import UserStorage
from core.user_management.chat_session import ChatSessionStorage
from core.user_management.chat_message import ChatMessageStorage
from encapsulation.data_model.orm_models import ChatMessage
import hashlib


def hash_password(password: str) -> str:
    """Simple password hashing for testing"""
    return hashlib.sha256(password.encode()).hexdigest()


def get_db_config():
    """Get PostgreSQL configuration"""
    return PostgreSQLConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=str(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "rag_test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "123")
    )

@pytest.fixture(scope="module")
def db_config() -> PostgreSQLConfig:
    return get_db_config()


@pytest.fixture(scope="module")
def user_storage(db_config: PostgreSQLConfig) -> UserStorage:
    return UserStorageConfig(relational_db_config=db_config).build()


@pytest.fixture(scope="module")
def session_storage(db_config: PostgreSQLConfig) -> ChatSessionStorage:
    return ChatSessionStorageConfig(relational_db_config=db_config).build()


@pytest.fixture(scope="module")
def message_storage(db_config: PostgreSQLConfig) -> ChatMessageStorage:
    return ChatMessageStorageConfig(relational_db_config=db_config).build()


@pytest.fixture(scope="module")
def user_ids(user_storage: UserStorage) -> tuple[uuid.UUID, uuid.UUID]:
    """Create two temporary users for chat storage tests."""
    suffix = str(uuid.uuid4())[:8]
    user1_id = user_storage.create_user(
        user_name=f"alice_test_{suffix}",
        hashed_password=hash_password("password123"),
    )
    user2_id = user_storage.create_user(
        user_name=f"bob_test_{suffix}",
        hashed_password=hash_password("password456"),
    )
    try:
        yield user1_id, user2_id
    finally:
        user_storage.delete_user(user1_id)
        user_storage.delete_user(user2_id)


@pytest.fixture
def session_ids(session_storage: ChatSessionStorage, user_ids: tuple[uuid.UUID, uuid.UUID]) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    user1_id, user2_id = user_ids
    session1_id = uuid.UUID(session_storage.create_session(user_id=user1_id, name="Alice's Research Session"))
    session2_id = uuid.UUID(session_storage.create_session(user_id=user1_id, name="Alice's Coding Session"))
    session3_id = uuid.UUID(session_storage.create_session(user_id=user2_id, name="Bob's Project Session"))
    try:
        yield session1_id, session2_id, session3_id
    finally:
        session_storage.delete_session(session1_id)
        session_storage.delete_session(session2_id)
        session_storage.delete_session(session3_id)


@pytest.fixture
def message_ids(message_storage: ChatMessageStorage, session_ids: tuple[uuid.UUID, uuid.UUID, uuid.UUID]) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    session1_id, _, _ = session_ids
    msg1 = message_storage.create_message(
        ChatMessage(
            session_id=session1_id,
            content={
                "role": "user",
                "content": "What is machine learning?",
                "metadata": {"timestamp": "2025-10-15T10:00:00"},
            },
        )
    )
    msg2 = message_storage.create_message(
        ChatMessage(
            session_id=session1_id,
            content={
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"model": "gpt-4", "tokens": 150},
            },
        )
    )
    msg3 = message_storage.create_message(
        ChatMessage(
            session_id=session1_id,
            content={
                "role": "user",
                "content": "Can you give me an example?",
                "metadata": {"timestamp": "2025-10-15T10:01:00"},
            },
        )
    )
    try:
        yield msg1.id, msg2.id, msg3.id
    finally:
        message_storage.delete_messages_by_session(session1_id)


def test_user_management(user_storage: UserStorage, user_ids: tuple[uuid.UUID, uuid.UUID]):
    """Test user management operations"""
    print("\n" + "=" * 80)
    print("测试 1: 用户管理")
    print("=" * 80)

    # Note: user creation is handled by the `user_ids` fixture.
    print("\n 创建用户... (由 pytest fixture 提供)")
    user1_id, user2_id = user_ids

    # Test 2: Get user by ID
    print("\n 通过 ID 获取用户...")
    user1 = user_storage.get_user(user1_id)
    assert user1 is not None, "User 1 should exist"
    assert user1.user_name, "Username should not be empty"
    print(f" 获取用户成功: {user1.user_name}")

    # Test 3: Get user by username
    print("\n 通过用户名获取用户...")
    user2 = user_storage.get_user(user2_id)
    assert user2 is not None, "User 2 should exist"
    user2_by_username = user_storage.get_user_by_username(user2.user_name)
    assert user2_by_username is not None, "User 2 should be retrievable by username"
    assert user2_by_username.id == user2_id, "User ID should match"
    print(f" 获取用户成功: {user2_by_username.user_name} (ID: {user2_by_username.id})")

    # Test 4: List users
    print("\n 列出所有用户...")
    users = user_storage.list_users(limit=10)
    print(f" 找到 {len(users)} 个用户")
    for u in users:
        print(f"   - {u.user_name} (ID: {u.id})")

    # Test 5: Update user
    print("\n 更新用户...")
    new_username = f"{user1.user_name}_updated"
    success = user_storage.update_user(
        user1_id,
        {"user_name": new_username}
    )
    assert success, "Update should succeed"
    user1_updated = user_storage.get_user(user1_id)
    assert user1_updated.user_name == new_username, "Username should be updated"
    print(f" 用户更新成功: {user1_updated.user_name}")


def test_chat_session_management(user_ids, session_ids, session_storage):
    """Test chat session management operations"""
    print("\n" + "=" * 80)
    print("测试 2: 聊天会话管理")
    print("=" * 80)

    user1_id, user2_id = user_ids
    session1_id, session2_id, session3_id = session_ids
    print("\n 创建聊天会话... (由 pytest fixture 提供)")

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
    success = session_storage.update_session(session1_id, {"name": "Alice's Updated Research Session"})
    assert success, "Update should succeed"
    session1_updated = session_storage.get_session(session1_id)
    assert session1_updated.name == "Alice's Updated Research Session", "Session name should be updated"
    print(f" 会话更新成功: {session1_updated.name}")


def test_chat_message_management(message_storage, session_ids, message_ids):
    """Test chat message management operations"""
    print("\n" + "=" * 80)
    print("测试 3: 聊天消息管理")
    print("=" * 80)

    session1_id, _, _ = session_ids
    msg1_id, msg2_id, msg3_id = message_ids
    print("\n 创建聊天消息... (由 pytest fixture 提供)")

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

    # Conversation history (derived)
    print("\n 获取对话历史（由消息列表推导）...")
    history = [{"role": m.content.get("role"), "content": m.content.get("content")} for m in messages]
    print(f" 对话历史: {len(history)} 条消息")
    for i, msg in enumerate(history, 1):
        print(f"   {i}. [{msg['role']}] {str(msg['content'])[:50]}...")


def test_cascade_deletion(user_storage: UserStorage, session_storage: ChatSessionStorage, message_storage: ChatMessageStorage):
    """Test cascade deletion"""
    print("\n" + "=" * 80)
    print("测试 4: 级联删除")
    print("=" * 80)

    suffix = str(uuid.uuid4())[:8]
    cascade_user_id = user_storage.create_user(
        user_name=f"cascade_user_{suffix}",
        hashed_password=hash_password("password123"),
    )
    cascade_session_id = uuid.UUID(
        session_storage.create_session(user_id=cascade_user_id, name="Cascade Session")
    )
    message_storage.create_message(
        ChatMessage(
            session_id=cascade_session_id,
            content={"role": "user", "content": "hello", "metadata": {}},
        )
    )

    # Check sessions before deletion
    print("\n 删除前检查...")
    sessions_before = session_storage.list_sessions_by_user(cascade_user_id)
    print(f"   用户 1 的会话数: {len(sessions_before)}")

    # Delete user (should cascade delete sessions and messages)
    print("\n 删除用户...")
    success = user_storage.delete_user(cascade_user_id)
    assert success, "User deletion should succeed"
    print(f" 用户删除成功: {cascade_user_id}")

    # Verify user is deleted
    user1 = user_storage.get_user(cascade_user_id)
    assert user1 is None, "User should be deleted"
    print(" 验证：用户已删除")

    # Verify sessions are deleted (cascade)
    sessions_after = session_storage.list_sessions_by_user(cascade_user_id)
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
