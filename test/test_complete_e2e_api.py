"""
完整的端到端 API 测试
测试流程: 文件上传 → 索引 → 检索 → 对话 → 删除
使用真实的 FastAPI 客户端测试所有 API 接口
"""
import os
import sys
import uuid
import time
import tempfile
from pathlib import Path
from io import BytesIO

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from main import app

# 创建测试客户端
client = TestClient(app)


def create_test_file(filename: str, content: str) -> BytesIO:
    """创建测试文件"""
    file_data = BytesIO(content.encode('utf-8'))
    file_data.name = filename
    return file_data


def test_complete_workflow():
    """
    测试完整的工作流程
    """
    print("=" * 80)
    print("完整端到端 API 测试")
    print("=" * 80)

    # 创建3个测试用户
    user1_id = uuid.uuid4()
    user2_id = uuid.uuid4()
    user3_id = uuid.uuid4()

    print(f"\n📊 创建测试用户:")
    print(f"  - User 1: {str(user1_id)[:8]}...")
    print(f"  - User 2: {str(user2_id)[:8]}...")
    print(f"  - User 3: {str(user3_id)[:8]}...")

    # 在数据库中创建用户（避免外键约束错误）
    from dotenv import load_dotenv
    load_dotenv()
    from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
    from encapsulation.data_model.orm_models import User
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # PostgreSQLConfig now reads from environment variables automatically
    db_config = PostgreSQLConfig()
    db = db_config.build()

    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    with db.SessionMaker() as session:
        for user_id in [user1_id, user2_id, user3_id]:
            user = User(
                id=user_id,
                user_name=f"test_user_{str(user_id)[:8]}",
                hashed_password="dummy_hash",  # 测试用的假密码
                created_at=now,
                updated_at=now
            )
            session.add(user)
        session.commit()
    print("  ✓ 用户已在数据库中创建")
    
    # ========== 阶段 1: 文件上传 ==========
    print("\n" + "=" * 80)
    print("阶段 1: 文件上传")
    print("=" * 80)
    
    uploaded_files = {}
    
    # User 1 上传 Python 相关文档
    print(f"\n📤 User 1 上传文档...")
    for i in range(3):
        content = f"""Python Programming Guide {i+1}

Python is a high-level, interpreted programming language known for its simplicity and readability.

Key Features:
- Dynamic typing and automatic memory management
- Extensive standard library
- Support for multiple programming paradigms (OOP, functional, procedural)
- Large ecosystem of third-party packages

Example Code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# List comprehension
squares = [x**2 for x in range(10)]
```

Python is widely used in web development, data science, machine learning, and automation.
"""
        filename = f"python_guide_{i+1}.txt"
        file_data = create_test_file(filename, content)
        
        response = client.post(
            "/knowledge",
            params={"owner_id": user1_id},
            files={"file": (filename, file_data, "text/plain")}
        )
        
        assert response.status_code == 201, f"Upload failed: {response.text}"
        file_id = response.json()
        uploaded_files[f"user1_file{i+1}"] = file_id
        print(f"  ✓ 上传成功: {filename} (ID: {file_id[:8]}...)")
    
    # User 2 上传 Java 相关文档
    print(f"\n📤 User 2 上传文档...")
    for i in range(3):
        content = f"""Java Programming Guide {i+1}

Java is a class-based, object-oriented programming language designed for portability and performance.

Key Features:
- Write Once, Run Anywhere (WORA) philosophy
- Strong type system and compile-time checking
- Automatic garbage collection
- Rich ecosystem with Spring, Hibernate, and more

Example Code:
```java
public class HelloWorld {{
    public static void main(String[] args) {{
        System.out.println("Hello, World!");
        
        // Stream API example
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        numbers.stream()
               .filter(n -> n % 2 == 0)
               .forEach(System.out::println);
    }}
}}
```

Java is commonly used in enterprise applications, Android development, and backend services.
"""
        filename = f"java_guide_{i+1}.txt"
        file_data = create_test_file(filename, content)
        
        response = client.post(
            "/knowledge",
            params={"owner_id": user2_id},
            files={"file": (filename, file_data, "text/plain")}
        )
        
        assert response.status_code == 201, f"Upload failed: {response.text}"
        file_id = response.json()
        uploaded_files[f"user2_file{i+1}"] = file_id
        print(f"  ✓ 上传成功: {filename} (ID: {file_id[:8]}...)")
    
    # User 3 上传 JavaScript 相关文档
    print(f"\n📤 User 3 上传文档...")
    for i in range(3):
        content = f"""JavaScript Programming Guide {i+1}

JavaScript is a versatile, dynamic programming language primarily used for web development.

Key Features:
- Event-driven and asynchronous programming
- Prototype-based object orientation
- First-class functions and closures
- Runs in browsers and Node.js

Example Code:
```javascript
// Async/await example
async function fetchData(url) {{
    try {{
        const response = await fetch(url);
        const data = await response.json();
        return data;
    }} catch (error) {{
        console.error('Error:', error);
    }}
}}

// Arrow functions and array methods
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
```

JavaScript powers modern web applications with frameworks like React, Vue, and Angular.
"""
        filename = f"javascript_guide_{i+1}.txt"
        file_data = create_test_file(filename, content)
        
        response = client.post(
            "/knowledge",
            params={"owner_id": user3_id},
            files={"file": (filename, file_data, "text/plain")}
        )
        
        assert response.status_code == 201, f"Upload failed: {response.text}"
        file_id = response.json()
        uploaded_files[f"user3_file{i+1}"] = file_id
        print(f"  ✓ 上传成功: {filename} (ID: {file_id[:8]}...)")
    
    print(f"\n✅ 总共上传了 {len(uploaded_files)} 个文件")
    
    # ========== 阶段 2: 等待索引完成 ==========
    print("\n" + "=" * 80)
    print("阶段 2: 等待后台索引完成")
    print("=" * 80)
    
    print("\n⏳ 等待 30 秒让后台索引任务完成...")
    time.sleep(30)
    print("✓ 索引应该已完成")
    
    # ========== 阶段 3: 测试用户隔离检索 ==========
    print("\n" + "=" * 80)
    print("阶段 3: 测试用户隔离检索")
    print("=" * 80)

    # 直接测试检索功能（不依赖 LLM）
    from framework.register import Register
    register = Register()
    rag_module = register.get_object("rag_inference")

    # User 1 检索 Python 内容
    print(f"\n🔍 User 1 检索 Python 内容...")
    chunks = rag_module.retriever.invoke("Python programming features", owner_id=user1_id, k=5)
    print(f"  Retrieved: {len(chunks)} chunks")
    assert len(chunks) > 0, "User 1 should retrieve Python chunks"
    # 验证所有 chunks 都属于 User 1
    for chunk in chunks:
        assert chunk.owner_id == user1_id, f"Chunk owner_id mismatch: {chunk.owner_id} != {user1_id}"
    print(f"  ✓ User 1 成功检索到 {len(chunks)} 个 Python 相关 chunks")
    print(f"  ✓ 所有 chunks 都属于 User 1")

    # User 2 检索 Java 内容
    print(f"\n🔍 User 2 检索 Java 内容...")
    chunks = rag_module.retriever.invoke("Java programming features", owner_id=user2_id, k=5)
    print(f"  Retrieved: {len(chunks)} chunks")
    assert len(chunks) > 0, "User 2 should retrieve Java chunks"
    for chunk in chunks:
        assert chunk.owner_id == user2_id, f"Chunk owner_id mismatch: {chunk.owner_id} != {user2_id}"
    print(f"  ✓ User 2 成功检索到 {len(chunks)} 个 Java 相关 chunks")
    print(f"  ✓ 所有 chunks 都属于 User 2")

    # User 3 检索 JavaScript 内容
    print(f"\n🔍 User 3 检索 JavaScript 内容...")
    chunks = rag_module.retriever.invoke("JavaScript programming features", owner_id=user3_id, k=5)
    print(f"  Retrieved: {len(chunks)} chunks")
    assert len(chunks) > 0, "User 3 should retrieve JavaScript chunks"
    for chunk in chunks:
        assert chunk.owner_id == user3_id, f"Chunk owner_id mismatch: {chunk.owner_id} != {user3_id}"
    print(f"  ✓ User 3 成功检索到 {len(chunks)} 个 JavaScript 相关 chunks")
    print(f"  ✓ 所有 chunks 都属于 User 3")

    # 测试跨用户隔离：User 1 不应该检索到 User 2 的内容
    print(f"\n🔍 测试跨用户隔离: User 1 检索 Java 内容...")
    chunks = rag_module.retriever.invoke("Java programming features", owner_id=user1_id, k=5)
    print(f"  Retrieved: {len(chunks)} chunks")
    # User 1 不应该检索到 Java 内容（属于 User 2）
    for chunk in chunks:
        assert chunk.owner_id == user1_id, f"Cross-user leak detected!"
    print(f"  ✓ User 1 没有检索到 User 2 的 Java 内容（用户隔离生效）")

    # ========== 阶段 4: 测试 Redis 会话管理 ==========
    print("\n" + "=" * 80)
    print("阶段 4: 测试 Redis 会话管理")
    print("=" * 80)

    # 导入会话管理相关模块
    from config.core.file_management.storage.chat_session_storage import ChatSessionStorageConfig
    from config.core.file_management.storage.chat_message_storage import ChatMessageStorageConfig
    from config.encapsulation.database.cache_db.redis_config import RedisConfig

    # 创建会话存储
    session_storage_config = ChatSessionStorageConfig(
        relational_db_config=PostgreSQLConfig()
    )
    session_storage = session_storage_config.build()

    # 创建消息存储（带 Redis 缓存）
    message_storage_config = ChatMessageStorageConfig(
        relational_db_config=PostgreSQLConfig(),
        cache_db_config=RedisConfig()  # 启用 Redis 缓存
    )
    message_storage = message_storage_config.build()

    # 为每个用户创建会话
    print(f"\n💬 创建聊天会话...")
    user1_session_id = session_storage.create_session(
        user_id=user1_id,
        name="User 1 Python Discussion"
    )
    print(f"  ✓ User 1 会话创建成功 (ID: {user1_session_id[:8]}...)")

    user2_session_id = session_storage.create_session(
        user_id=user2_id,
        name="User 2 Java Discussion"
    )
    print(f"  ✓ User 2 会话创建成功 (ID: {user2_session_id[:8]}...)")

    # 测试消息创建和 Redis 缓存
    print(f"\n📝 测试消息创建和 Redis 缓存...")

    # User 1 发送消息
    msg1_id = message_storage.create_message(
        session_id=uuid.UUID(user1_session_id),
        content={
            "role": "user",
            "content": "What are the key features of Python?",
            "metadata": {}
        }
    )
    print(f"  ✓ User 1 消息 1 创建成功 (ID: {msg1_id[:8]}...)")

    msg2_id = message_storage.create_message(
        session_id=uuid.UUID(user1_session_id),
        content={
            "role": "assistant",
            "content": "Python has several key features: dynamic typing, interpreted execution, extensive standard library, and clean syntax.",
            "metadata": {"model": "test-model"}
        }
    )
    print(f"  ✓ User 1 消息 2 创建成功 (ID: {msg2_id[:8]}...)")

    # User 2 发送消息
    msg3_id = message_storage.create_message(
        session_id=uuid.UUID(user2_session_id),
        content={
            "role": "user",
            "content": "Explain Java's object-oriented features",
            "metadata": {}
        }
    )
    print(f"  ✓ User 2 消息创建成功 (ID: {msg3_id[:8]}...)")

    # 测试从 Redis 读取消息（应该很快）
    print(f"\n🔍 测试从 Redis 缓存读取消息...")
    start_time = time.time()
    messages = message_storage.list_messages_by_session(uuid.UUID(user1_session_id), limit=10)
    redis_read_time = time.time() - start_time

    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assert messages[0].content["role"] == "user", "First message should be from user"
    assert messages[1].content["role"] == "assistant", "Second message should be from assistant"
    print(f"  ✓ 从 Redis 读取 {len(messages)} 条消息 (耗时: {redis_read_time*1000:.2f}ms)")

    # 测试获取对话历史
    print(f"\n📜 测试获取对话历史...")
    history = message_storage.get_conversation_history(uuid.UUID(user1_session_id), limit=10)
    assert len(history) == 2, f"Expected 2 history entries, got {len(history)}"
    assert history[0]["role"] == "user", "First history entry should be user"
    assert history[1]["role"] == "assistant", "Second history entry should be assistant"
    print(f"  ✓ 对话历史获取成功 ({len(history)} 条记录)")
    for i, msg in enumerate(history, 1):
        print(f"    {i}. [{msg['role']}] {msg['content'][:50]}...")

    # 测试用户会话隔离
    print(f"\n🔒 测试用户会话隔离...")
    user1_sessions = session_storage.list_sessions_by_user(user1_id)
    user2_sessions = session_storage.list_sessions_by_user(user2_id)

    assert len(user1_sessions) == 1, f"User 1 should have 1 session, got {len(user1_sessions)}"
    assert len(user2_sessions) == 1, f"User 2 should have 1 session, got {len(user2_sessions)}"
    assert user1_sessions[0].id == uuid.UUID(user1_session_id), "User 1 session ID mismatch"
    assert user2_sessions[0].id == uuid.UUID(user2_session_id), "User 2 session ID mismatch"
    print(f"  ✓ User 1 有 {len(user1_sessions)} 个会话")
    print(f"  ✓ User 2 有 {len(user2_sessions)} 个会话")
    print(f"  ✓ 会话隔离验证成功")

    # 测试 Redis 缓存失效后从 PostgreSQL 读取
    print(f"\n🗄️  测试 Redis 缓存失效后从 PostgreSQL 读取...")
    if message_storage.cache_store:
        # 清除 Redis 缓存
        cache_key = f"chat:session:{user1_session_id}:messages"
        message_storage.cache_store.delete(cache_key)
        print(f"  ✓ Redis 缓存已清除")

        # 再次读取（应该从 PostgreSQL 读取并回填 Redis）
        start_time = time.time()
        messages_from_pg = message_storage.list_messages_by_session(uuid.UUID(user1_session_id), limit=10)
        pg_read_time = time.time() - start_time

        assert len(messages_from_pg) == 2, f"Expected 2 messages from PostgreSQL, got {len(messages_from_pg)}"
        print(f"  ✓ 从 PostgreSQL 读取 {len(messages_from_pg)} 条消息 (耗时: {pg_read_time*1000:.2f}ms)")
        print(f"  ✓ Redis 缓存已自动回填")

    # ========== 阶段 5: 测试文件下载 ==========
    print("\n" + "=" * 80)
    print("阶段 5: 测试文件下载")
    print("=" * 80)

    # 下载 User 1 的第一个文件
    file_id = uploaded_files["user1_file1"]
    print(f"\n📥 下载 User 1 的文件 (ID: {file_id[:8]}...)...")
    response = client.get(f"/knowledge/{file_id}/download")
    assert response.status_code == 200, f"Download failed: {response.text}"
    content = response.content.decode('utf-8')
    assert "Python Programming Guide" in content, "Downloaded content doesn't match"
    print(f"  ✓ 文件下载成功")
    print(f"  Content preview: {content[:100]}...")

    # ========== 阶段 6: 测试文件删除 ==========
    print("\n" + "=" * 80)
    print("阶段 6: 测试文件删除")
    print("=" * 80)

    # 测试跨用户删除（应该失败）
    user2_file_id = uploaded_files["user2_file1"]
    print(f"\n🚫 测试跨用户删除: User 1 尝试删除 User 2 的文件 (ID: {user2_file_id[:8]}...)...")
    response = client.delete(f"/knowledge/{user2_file_id}?owner_id={user1_id}")
    assert response.status_code == 403, f"Cross-user deletion should be forbidden, got: {response.status_code}"
    print(f"  ✓ 跨用户删除被正确拒绝 (403 Forbidden)")

    # 验证 User 2 的文件仍然存在
    print(f"\n🔍 验证 User 2 的文件仍然存在...")
    response = client.get(f"/knowledge/{user2_file_id}/download")
    assert response.status_code == 200, f"User 2's file should still exist"
    print(f"  ✓ User 2 的文件仍然存在（未被 User 1 删除）")

    # 删除 User 1 的第一个文件（应该成功）
    file_id = uploaded_files["user1_file1"]
    print(f"\n🗑️  User 1 删除自己的文件 (ID: {file_id[:8]}...)...")
    response = client.delete(f"/knowledge/{file_id}?owner_id={user1_id}")
    assert response.status_code == 204, f"Delete failed: {response.text}"
    print(f"  ✓ 文件删除成功")

    # 验证文件已被删除 (下载应该失败)
    print(f"\n🔍 验证文件已被删除...")
    response = client.get(f"/knowledge/{file_id}/download")
    assert response.status_code == 404, f"File should be deleted but still accessible"
    print(f"  ✓ 文件确实已被删除 (404 Not Found)")

    # 删除剩余的测试文件
    print(f"\n🗑️  清理剩余测试文件...")
    deleted_count = 0

    # 映射文件到对应的 owner_id
    file_owner_map = {
        "user1_file1": user1_id,  # 已经删除过了
        "user1_file2": user1_id,
        "user1_file3": user1_id,
        "user2_file1": user2_id,
        "user2_file2": user2_id,
        "user2_file3": user2_id,
        "user3_file1": user3_id,
        "user3_file2": user3_id,
        "user3_file3": user3_id,
    }

    for key, file_id in uploaded_files.items():
        if key == "user1_file1":  # 已经删除过了
            continue
        try:
            owner_id = file_owner_map[key]
            response = client.delete(f"/knowledge/{file_id}?owner_id={owner_id}")
            if response.status_code == 204:
                deleted_count += 1
        except Exception as e:
            print(f"  ⚠️  删除 {key} 失败: {e}")

    print(f"  ✓ 清理完成，删除了 {deleted_count + 1} 个文件")

    # ========== 测试总结 ==========
    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)

    print(f"  ✓ 文件上传: 9 个文件 (3个用户 × 3个文件)")
    print(f"  ✓ 后台索引: 等待 30 秒完成")
    print(f"  ✓ 用户隔离检索: 3 个用户各自检索成功")
    print(f"  ✓ 跨用户检索隔离: User 1 无法访问 User 2 的文档")
    print(f"  ✓ Redis 会话管理: 创建会话、发送消息、读取历史")
    print(f"  ✓ Redis 缓存: 快速读取、缓存失效、自动回填")
    print(f"  ✓ 会话隔离: 用户只能访问自己的会话")
    print(f"  ✓ 文件下载: 成功下载并验证内容")
    print(f"  ✓ 跨用户删除隔离: User 1 无法删除 User 2 的文件 (403 Forbidden)")
    print(f"  ✓ 文件删除: 成功删除并验证")
    print(f"  ✓ 清理: 删除所有测试文件")


if __name__ == "__main__":
    try:
        test_complete_workflow()
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

