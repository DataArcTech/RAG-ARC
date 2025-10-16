"""清理数据库中的测试数据"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

try:
    # PostgreSQLConfig now reads from environment variables automatically
    db_config = PostgreSQLConfig()
    db = db_config.build()

    # 清空所有表（包括 user 表）
    print("清理数据库...")
    from sqlalchemy import text
    with db.SessionMaker() as session:
        session.execute(text("TRUNCATE TABLE chunk_metadata CASCADE"))
        session.execute(text("TRUNCATE TABLE parsed_content_metadata CASCADE"))
        session.execute(text("TRUNCATE TABLE file_metadata CASCADE"))
        session.execute(text("TRUNCATE TABLE chat_message CASCADE"))
        session.execute(text("TRUNCATE TABLE chat_session CASCADE"))
        session.execute(text("TRUNCATE TABLE \"user\" CASCADE"))  # user 是保留字，需要引号
        session.commit()
    print("✓ 数据库清理完成")

except Exception as e:
    print(f"❌ 数据库清理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

