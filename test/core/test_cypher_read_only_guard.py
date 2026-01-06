import pytest

from core.graph_adapter.cypher import is_read_only_cypher


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n",
        "// comment\nMATCH (n) RETURN n",
        "/* block */\nMATCH (n) RETURN n",
        (
            "MATCH (a)\n"
            "CALL { WITH a RETURN a AS x }\n"
            "RETURN x"
        ),
        (
            "// comment\n"
            "MATCH (a)\n"
            "CALL { WITH a RETURN a AS x }\n"
            "UNION ALL\n"
            "MATCH (b) RETURN b AS x"
        ),
    ],
)
def test_is_read_only_cypher_accepts_read_only_queries(cypher: str) -> None:
    assert is_read_only_cypher(cypher) is True


@pytest.mark.parametrize(
    "cypher",
    [
        "CALL dbms.components()",
        "MATCH (n) CALL dbms.components() YIELD name RETURN name",
        "CALL apoc.help('x')",
        "MATCH (n) CALL gds.version() RETURN n",
        "CREATE (n:Entity {name:'x'}) RETURN n",
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) RETURN n; MATCH (m) RETURN m",
        "MATCH (n) CALL { WITH n RETURN n } IN TRANSACTIONS RETURN n",
        "CREATE CONSTRAINT foo IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    ],
)
def test_is_read_only_cypher_rejects_mutations_or_procedures(cypher: str) -> None:
    assert is_read_only_cypher(cypher) is False
