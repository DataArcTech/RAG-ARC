"""
Test script for Chat Message dual-layer storage (Redis + PostgreSQL).

Tests:
1. Redis connection and basic operations
2. Message creation with dual-layer write (Redis + PostgreSQL)
3. Message retrieval with cache hit (from Redis)
4. Message retrieval with cache miss (from PostgreSQL, then backfill Redis)
5. Cache invalidation on message deletion
6. Cache TTL and expiration
7. Performance comparison (Redis vs PostgreSQL)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import time
import uuid
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.encapsulation.database.cache_db.redis_config import RedisConfig
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


def get_redis_config():
    """Get Redis configuration"""
    return RedisConfig(
        type="redis",
        host="localhost",
        port=6379,
        db=0,
        password=None
    )


def test_redis_connection():
    """Test Redis connection"""
    print("\n" + "=" * 80)
    print("测试 1: Redis 连接")
    print("=" * 80)

    redis_config = get_redis_config()
    redis_db = redis_config.build()

    # Test ping
    assert redis_db.ping(), "Redis ping should succeed"
    print(" Redis 连接成功")

    # Test basic operations
    redis_db.set("test_key", "test_value", ttl=60)
    value = redis_db.get("test_key")
    assert value == "test_value", "Value should match"
    print(" Redis 基本操作成功")

    # Test list operations
    redis_db.delete("test_list")
    redis_db.lpush("test_list", "item1", "item2", "item3")
    items = redis_db.lrange("test_list", 0, -1)
    assert len(items) == 3, "List should have 3 items"
    print(f" Redis 列表操作成功: {items}")

    # Clean up
    redis_db.delete("test_key")
    redis_db.delete("test_list")

    return redis_db


def setup_test_data():
    """Setup test user and session"""
    print("\n" + "=" * 80)
    print("测试 2: 创建测试数据（用户和会话）")
    print("=" * 80)

    db_config = get_db_config()

    # Create user
    user_storage_config = UserStorageConfig(relational_db_config=db_config)
    user_storage = user_storage_config.build()

    # Try to delete existing test user first
    existing_user = user_storage.get_user_by_username("test_dual_storage_user")
    if existing_user:
        user_storage.delete_user(str(existing_user.id))
        print(" 清理已存在的测试用户")

    user_id = user_storage.create_user(
        user_name="test_dual_storage_user",
        hashed_password=hash_password("password123")
    )
    print(f" 创建用户: {user_id}")

    # Create session
    session_storage_config = ChatSessionStorageConfig(relational_db_config=db_config)
    session_storage = session_storage_config.build()

    session_id = session_storage.create_session(
        user_id=user_id,
        name="Test Dual Storage Session"
    )
    print(f" 创建会话: {session_id}")

    return user_id, session_id


def test_dual_layer_write(session_id: str, redis_db):
    """Test message creation with dual-layer write"""
    print("\n" + "=" * 80)
    print("测试 3: 双层写入（Redis + PostgreSQL）")
    print("=" * 80)

    db_config = get_db_config()
    redis_config = get_redis_config()

    # Create ChatMessageStorage with Redis cache
    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        cache_db_config=redis_config,
        cache_max_messages=100,
        cache_ttl=3600
    )
    message_storage = message_storage_config.build()

    # Create messages
    message_ids = []
    for i in range(5):
        msg_id = message_storage.create_message(
            session_id=uuid.UUID(session_id),
            content={
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i+1}",
                "metadata": {"index": i}
            }
        )
        message_ids.append(msg_id)
        print(f" 创建消息 {i+1}: {msg_id}")

    # Verify Redis cache
    cache_key = f"chat:session:{session_id}:messages"
    cached_messages = redis_db.lrange(cache_key, 0, -1)
    print(f"\n Redis 缓存状态:")
    print(f"   - 缓存 key: {cache_key}")
    print(f"   - 缓存消息数: {len(cached_messages)}")
    assert len(cached_messages) == 5, "Redis should have 5 messages"
    print(" Redis 缓存验证成功")

    # Verify PostgreSQL
    messages_from_db = message_storage.metadata_store.list_chat_messages_by_session(
        session_id=uuid.UUID(session_id),
        limit=100
    )
    print(f"\n PostgreSQL 状态:")
    print(f"   - 数据库消息数: {len(messages_from_db)}")
    assert len(messages_from_db) == 5, "PostgreSQL should have 5 messages"
    print(" PostgreSQL 验证成功")

    return message_ids


def test_cache_hit(session_id: str):
    """Test message retrieval with cache hit"""
    print("\n" + "=" * 80)
    print("测试 4: 缓存命中（从 Redis 读取）")
    print("=" * 80)

    db_config = get_db_config()
    redis_config = get_redis_config()

    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        cache_db_config=redis_config,
        cache_max_messages=100,
        cache_ttl=3600
    )
    message_storage = message_storage_config.build()

    # Measure read time from Redis
    start_time = time.time()
    messages = message_storage.list_messages_by_session(uuid.UUID(session_id), limit=5)
    redis_time = time.time() - start_time

    print(f"\n 缓存命中性能:")
    print(f"   - 读取消息数: {len(messages)}")
    print(f"   - 读取时间: {redis_time*1000:.2f} ms")
    assert len(messages) == 5, "Should retrieve 5 messages"
    print(" 缓存命中测试成功")

    # Verify message order (oldest first)
    for i, msg in enumerate(messages):
        print(f"   {i+1}. [{msg.content['role']}] {msg.content['content']}")

    return redis_time


def test_cache_miss(session_id: str, redis_db):
    """Test message retrieval with cache miss and backfill"""
    print("\n" + "=" * 80)
    print("测试 5: 缓存未命中（从 PostgreSQL 读取并回填 Redis）")
    print("=" * 80)

    db_config = get_db_config()
    redis_config = get_redis_config()

    # Clear Redis cache
    cache_key = f"chat:session:{session_id}:messages"
    redis_db.delete(cache_key)
    print(" 清空 Redis 缓存")

    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        cache_db_config=redis_config,
        cache_max_messages=100,
        cache_ttl=3600
    )
    message_storage = message_storage_config.build()

    # Measure read time from PostgreSQL
    start_time = time.time()
    messages = message_storage.list_messages_by_session(uuid.UUID(session_id), limit=5)
    pg_time = time.time() - start_time

    print(f"\n 缓存未命中性能:")
    print(f"   - 读取消息数: {len(messages)}")
    print(f"   - 读取时间: {pg_time*1000:.2f} ms")
    assert len(messages) == 5, "Should retrieve 5 messages"
    print(" PostgreSQL 读取成功")

    # Verify Redis backfill
    cached_messages = redis_db.lrange(cache_key, 0, -1)
    print(f"\n Redis 回填状态:")
    print(f"   - 回填消息数: {len(cached_messages)}")
    assert len(cached_messages) == 5, "Redis should be backfilled with 5 messages"
    print(" Redis 回填成功")

    return pg_time


def test_cache_invalidation(session_id: str, message_ids: list, redis_db):
    """Test cache invalidation on message deletion"""
    print("\n" + "=" * 80)
    print("测试 6: 缓存失效（删除消息时）")
    print("=" * 80)

    db_config = get_db_config()
    redis_config = get_redis_config()

    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        cache_db_config=redis_config,
        cache_max_messages=100,
        cache_ttl=3600
    )
    message_storage = message_storage_config.build()

    # Verify cache exists
    cache_key = f"chat:session:{session_id}:messages"
    cached_before = redis_db.lrange(cache_key, 0, -1)
    print(f"删除前缓存消息数: {len(cached_before)}")

    # Delete a message
    success = message_storage.delete_message(message_ids[0])
    assert success, "Message deletion should succeed"
    print(f" 删除消息: {message_ids[0]}")

    # Verify cache is invalidated
    cached_after = redis_db.lrange(cache_key, 0, -1)
    print(f"删除后缓存消息数: {len(cached_after)}")
    assert len(cached_after) == 0, "Cache should be invalidated"
    print(" 缓存失效成功")

    # Verify PostgreSQL
    messages_from_db = message_storage.metadata_store.list_chat_messages_by_session(
        session_id=uuid.UUID(session_id),
        limit=100
    )
    print(f"PostgreSQL 剩余消息数: {len(messages_from_db)}")
    assert len(messages_from_db) == 4, "PostgreSQL should have 4 messages"
    print(" PostgreSQL 删除成功")


def cleanup_test_data(user_id: str):
    """Clean up test data"""
    print("\n" + "=" * 80)
    print("清理测试数据")
    print("=" * 80)

    db_config = get_db_config()

    user_storage_config = UserStorageConfig(relational_db_config=db_config)
    user_storage = user_storage_config.build()

    # Delete user (cascades to sessions and messages)
    success = user_storage.delete_user(user_id)
    assert success, "User deletion should succeed"
    print(f" 删除用户: {user_id}")


if __name__ == "__main__":
    print("开始测试")
    try:
        # Test 1: Redis connection
        redis_db = test_redis_connection()

        # Test 2: Setup test data
        user_id, session_id = setup_test_data()

        # Test 3: Dual-layer write
        message_ids = test_dual_layer_write(session_id, redis_db)

        # Test 4: Cache hit
        redis_time = test_cache_hit(session_id)

        # Test 5: Cache miss and backfill
        pg_time = test_cache_miss(session_id, redis_db)

        # Test 6: Cache invalidation
        test_cache_invalidation(session_id, message_ids, redis_db)


        # Cleanup
        cleanup_test_data(user_id)


    except Exception as e:
        print(f"\n 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

