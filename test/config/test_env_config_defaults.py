from config.encapsulation.database.cache_db.redis_config import RedisConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig


def test_postgres_config_reads_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "db-host")
    monkeypatch.setenv("POSTGRES_PORT", "9999")
    monkeypatch.setenv("POSTGRES_DB", "db-name")
    monkeypatch.setenv("POSTGRES_USER", "db-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "db-pass")

    cfg = PostgreSQLConfig()
    assert cfg.host == "db-host"
    assert cfg.port == "9999"
    assert cfg.database == "db-name"
    assert cfg.user == "db-user"
    assert cfg.password == "db-pass"


def test_redis_config_reads_env(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis-host")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-pass")

    cfg = RedisConfig()
    assert cfg.host == "redis-host"
    assert cfg.port == "6380"
    assert cfg.db == "2"
    assert cfg.password == "redis-pass"


def test_faiss_config_reads_env_index_path(monkeypatch):
    monkeypatch.setenv("FAISS_INDEX_PATH", "./data/custom_faiss_index")
    cfg = FaissVectorDBConfig(embedding_config=OpenAIEmbeddingConfig(model_name="text-embedding-3-small"))
    assert cfg.index_path == "./data/custom_faiss_index"


def test_bm25_config_reads_env_index_path(monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", "./data/custom_bm25_index")
    cfg = BM25BuilderConfig()
    assert cfg.index_path == "./data/custom_bm25_index"


def test_graph_config_reads_env_storage_path(monkeypatch):
    monkeypatch.setenv("GRAPH_STORAGE_PATH", "./data/custom_graph")
    monkeypatch.setenv("GRAPH_INDEX_NAME", "custom_index")
    cfg = PrunedHippoRAGNeo4jConfig(
        embedding=OpenAIEmbeddingConfig(model_name="text-embedding-3-small"),
    )
    assert cfg.storage_path == "./data/custom_graph"
    assert cfg.index_name == "custom_index"

