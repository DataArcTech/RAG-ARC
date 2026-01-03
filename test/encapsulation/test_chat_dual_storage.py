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
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGARC_CHAT_STORAGE_TESTS") != "1",
    reason="Requires Redis/PostgreSQL services; set RUN_RAGARC_CHAT_STORAGE_TESTS=1 to run.",
)

import time
import uuid
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.encapsulation.database.cache_db.redis_config import RedisConfig
from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
from config.core.file_management.storage.user_storage import UserStorageConfig
from encapsulation.data_model.orm_models import ChatMessage
from core.user_management.chat_message import ChatMessageStorage

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


def get_redis_config():
    """Get Redis configuration"""
    return RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=str(os.getenv("REDIS_PORT", "6379")),
        db=str(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD")
    )

@pytest.fixture(scope="module")
def db_config() -> PostgreSQLConfig:
    return get_db_config()


@pytest.fixture(scope="module")
def redis_db():
    redis_config = get_redis_config()
    return redis_config.build()


@pytest.fixture(scope="module")
def test_user_id(db_config: PostgreSQLConfig) -> uuid.UUID:
    user_storage = UserStorageConfig(relational_db_config=db_config).build()

    existing_user = user_storage.get_user_by_username("test_dual_storage_user")
    if existing_user:
        user_storage.delete_user(existing_user.id)

    user_id = user_storage.create_user(
        user_name="test_dual_storage_user",
        hashed_password=hash_password("password123"),
    )
    try:
        yield user_id
    finally:
        user_storage.delete_user(user_id)


@pytest.fixture(scope="module")
def session_id(db_config: PostgreSQLConfig, test_user_id: uuid.UUID) -> uuid.UUID:
    session_storage = ChatSessionStorageConfig(relational_db_config=db_config).build()
    session_id_str = session_storage.create_session(
        user_id=test_user_id,
        name="Test Dual Storage Session",
    )
    return uuid.UUID(session_id_str)


@pytest.fixture(scope="module")
def message_storage(db_config: PostgreSQLConfig) -> ChatMessageStorage:
    redis_config = get_redis_config()
    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=db_config,
        cache_db_config=redis_config,
        cache_max_messages=100,
        cache_ttl=3600,
    )
    return message_storage_config.build()


@pytest.fixture
def seeded_messages(message_storage, session_id: uuid.UUID):
    message_storage.delete_messages_by_session(session_id)

    message_ids: list[uuid.UUID] = []
    for i in range(5):
        message = ChatMessage(
            session_id=session_id,
            content={
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i+1}",
                "metadata": {"index": i},
            },
        )
        stored = message_storage.create_message(message)
        message_ids.append(stored.id)

    try:
        yield message_ids
    finally:
        message_storage.delete_messages_by_session(session_id)


def test_redis_connection(redis_db):
    """Test Redis connection"""
    print("\n" + "=" * 80)
    print("测试 1: Redis 连接")
    print("=" * 80)

    # Test ping
    assert redis_db.ping(), "Redis ping should succeed"
    print(" Redis 连接成功")

    # Test basic operations
    key = f"test_key:{uuid.uuid4()}"
    redis_db.set(key, "test_value", ttl=60)
    value = redis_db.get(key)
    assert value == "test_value", "Value should match"
    print(" Redis 基本操作成功")

    # Test list operations
    list_key = f"test_list:{uuid.uuid4()}"
    redis_db.delete(list_key)
    redis_db.lpush(list_key, "item1", "item2", "item3")
    items = redis_db.lrange(list_key, 0, -1)
    assert len(items) == 3, "List should have 3 items"
    print(f" Redis 列表操作成功: {items}")

    # Clean up
    redis_db.delete(key)
    redis_db.delete(list_key)


def test_dual_layer_write(session_id: uuid.UUID, redis_db, message_storage, seeded_messages):
    """Test message creation with dual-layer write"""
    print("\n" + "=" * 80)
    print("测试 3: 双层写入（Redis + PostgreSQL）")
    print("=" * 80)

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
        session_id=session_id,
        limit=100
    )
    print(f"\n PostgreSQL 状态:")
    print(f"   - 数据库消息数: {len(messages_from_db)}")
    assert len(messages_from_db) == 5, "PostgreSQL should have 5 messages"
    print(" PostgreSQL 验证成功")


def test_cache_hit(session_id: uuid.UUID, message_storage, seeded_messages):
    """Test message retrieval with cache hit"""
    print("\n" + "=" * 80)
    print("测试 4: 缓存命中（从 Redis 读取）")
    print("=" * 80)

    # Measure read time from Redis
    start_time = time.time()
    messages = message_storage.list_messages_by_session(session_id, limit=5)
    redis_time = time.time() - start_time

    print(f"\n 缓存命中性能:")
    print(f"   - 读取消息数: {len(messages)}")
    print(f"   - 读取时间: {redis_time*1000:.2f} ms")
    assert len(messages) == 5, "Should retrieve 5 messages"
    print(" 缓存命中测试成功")

    # Verify message order (oldest first)
    for i, msg in enumerate(messages):
        print(f"   {i+1}. [{msg.content['role']}] {msg.content['content']}")


def test_cache_miss(session_id: uuid.UUID, redis_db, message_storage, seeded_messages):
    """Test message retrieval with cache miss and backfill"""
    print("\n" + "=" * 80)
    print("测试 5: 缓存未命中（从 PostgreSQL 读取并回填 Redis）")
    print("=" * 80)

    # Clear Redis cache
    cache_key = f"chat:session:{session_id}:messages"
    redis_db.delete(cache_key)
    print(" 清空 Redis 缓存")

    # Measure read time from PostgreSQL
    start_time = time.time()
    messages = message_storage.list_messages_by_session(session_id, limit=5)
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


def test_cache_invalidation(session_id: uuid.UUID, seeded_messages: list[uuid.UUID], redis_db, message_storage):
    """Test cache invalidation on message deletion"""
    print("\n" + "=" * 80)
    print("测试 6: 缓存失效（删除消息时）")
    print("=" * 80)

    # Verify cache exists
    cache_key = f"chat:session:{session_id}:messages"
    cached_before = redis_db.lrange(cache_key, 0, -1)
    print(f"删除前缓存消息数: {len(cached_before)}")
    assert len(cached_before) == 5, "Cache should contain seeded messages"

    # Delete a message
    success = message_storage.delete_message(seeded_messages[0])
    assert success, "Message deletion should succeed"
    print(f" 删除消息: {seeded_messages[0]}")

    # Verify cache is invalidated
    cached_after = redis_db.lrange(cache_key, 0, -1)
    print(f"删除后缓存消息数: {len(cached_after)}")
    assert len(cached_after) == 0, "Cache should be invalidated"
    print(" 缓存失效成功")

    # Verify PostgreSQL
    messages_from_db = message_storage.metadata_store.list_chat_messages_by_session(
        session_id=session_id,
        limit=100
    )
    print(f"PostgreSQL 剩余消息数: {len(messages_from_db)}")
    assert len(messages_from_db) == 4, "PostgreSQL should have 4 messages"
    print(" PostgreSQL 删除成功")
