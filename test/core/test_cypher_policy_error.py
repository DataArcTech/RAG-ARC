import pytest

from core.graph_adapter.cypher import assert_read_only_cypher, CypherPolicyError


def test_assert_read_only_cypher_raises_policy_error_on_mutation() -> None:
    with pytest.raises(CypherPolicyError):
        assert_read_only_cypher("CREATE (n:Entity {name:'x'}) RETURN n")

