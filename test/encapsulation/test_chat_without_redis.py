"""
Test script for Chat Message storage WITHOUT Redis (PostgreSQL only).

This test verifies that the system works correctly when Redis is disabled.
"""

import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
from config.core.file_management.storage.user_storage import UserStorageConfig
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


def test_without_redis():
    """Test chat message storage without Redis"""
    print("\n" + "=" * 80)
    print("测试：不使用 Redis 的聊天消息存储（仅 PostgreSQL）")
    print("=" * 80)

    db_config = get_db_config()

    # ==================== Setup ====================
    print("\n 创建测试数据...")
    
    # Create user
    user_storage_config = UserStorageConfig(relational_db_config=db_config)
    user_storage = user_storage_config.build()

    # Clean up existing test user
    existing_user = user_storage.get_user_by_username("test_no_redis_user")
    if existing_user:
        user_storage.delete_user(str(existing_user.id))
        print(" 清理已存在的测试用户")

    user_id = user_storage.create_user(
        user_name="test_no_redis_user",
        hashed_password=hash_password("password123")
    )
    print(f" 创建用户: {user_id}")

    # Create session
    session_storage_config = ChatSessionStorageConfig(relational_db_config=db_config)
    session_storage = session_storage_config.build()

    session_id = session_storage.create_session(
        user_id=user_id,
        name="Test Session Without Redis"
    )
    print(f" 创建会话: {session_id}")

    # ==================== Create ChatMessageStorage WITHOUT Redis ====================
    print("\n 初始化 ChatMessageStorage（不使用 Redis）...")
    
    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        # cache_db_config=None,  # 不传入 Redis 配置
    )
    message_storage = message_storage_config.build()
    
    # Verify Redis is not enabled
    assert message_storage.cache_store is None, "Redis should be disabled"
    print(" 确认 Redis 已禁用")

    # ==================== Test Message Creation ====================
    print("\n 测试消息创建...")
    
    message_ids = []
    for i in range(5):
        msg_id = message_storage.create_message(
            session_id=uuid.UUID(session_id),
            content={
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i+1} without Redis",
                "metadata": {"index": i}
            }
        )
        message_ids.append(msg_id)
        print(f" 创建消息 {i+1}: {msg_id}")

    # ==================== Test Message Retrieval ====================
    print("\n 测试消息读取...")
    
    messages = message_storage.list_messages_by_session(uuid.UUID(session_id), limit=10)
    print(f" 读取到 {len(messages)} 条消息")
    
    assert len(messages) == 5, "Should have 5 messages"
    
    for i, msg in enumerate(messages):
        print(f"   {i+1}. [{msg.content['role']}] {msg.content['content']}")

    # ==================== Test Message Deletion ====================
    print("\n  测试消息删除...")
    
    success = message_storage.delete_message(message_ids[0])
    assert success, "Message deletion should succeed"
    print(f" 删除消息: {message_ids[0]}")

    # Verify deletion
    messages_after = message_storage.list_messages_by_session(session_id, limit=10)
    assert len(messages_after) == 4, "Should have 4 messages after deletion"
    print(f" 验证删除成功，剩余 {len(messages_after)} 条消息")

    # ==================== Test Conversation History ====================
    print("\n测试对话历史...")
    
    history = message_storage.get_conversation_history(uuid.UUID(session_id), limit=10)
    print(f" 获取对话历史: {len(history)} 条消息")
    
    for i, msg in enumerate(history):
        print(f"   {i+1}. [{msg['role']}] {msg['content'][:50]}...")

    # ==================== Cleanup ====================
    print("\n 清理测试数据...")
    
    user_storage.delete_user(user_id)
    print(f" 删除用户: {user_id}")



if __name__ == "__main__":
    try:
        test_without_redis()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

