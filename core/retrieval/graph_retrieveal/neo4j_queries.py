from typing import Dict


_NEO4J_QUERIES: Dict[str, str] = {
    "semantic_search_entities": """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Chunk)
        WITH e, count(d) as mention_count
        RETURN e.id_ as entity_id,
               e.entity_name as entity_name,
               e.entity_type as entity_type,
               e.embedding as embedding,
               mention_count,
               e.attributes as attributes
        ORDER BY mention_count DESC
        LIMIT 2000
    """,
    "semantic_search_chunks": """
        MATCH (d:Chunk)
        WHERE d.embedding IS NOT NULL
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
        WITH d, count(e) as entity_count, collect(e.entity_name) as entity_names
        RETURN d.id_ as chunk_id,
               d.content as content,
               d.embedding as embedding,
               d.metadata as metadata,
               entity_count,
               entity_names
        ORDER BY entity_count DESC
        LIMIT 2000
    """,
    "get_entity_data": """
        MATCH (e:Entity {id_: $entity_id})
        RETURN e.entity_name as name, e.entity_type as type, e.attributes as attributes
    """,
    "get_entity_neighbors": """
        MATCH (e1:Entity {id_: $entity_id})-[r]->(e2:Entity)
        OPTIONAL MATCH (e2)<-[:MENTIONS]-(d:Chunk)
        WITH e2, type(r) as relation_type, count(d) as mention_count
        RETURN e2.id_ as neighbor_id, relation_type, mention_count
        UNION
        MATCH (e1:Entity)-[r]->(e2:Entity {id_: $entity_id})
        OPTIONAL MATCH (e1)<-[:MENTIONS]-(d:Chunk)
        WITH e1, type(r) as relation_type, count(d) as mention_count
        RETURN e1.id_ as neighbor_id, relation_type, mention_count
    """,
    "get_entity_degree": """
        MATCH (e:Entity {id_: $entity_id})-[r]-()
        RETURN count(r) as degree
    """,
    "get_entity_mentions": """
        MATCH (e:Entity {id_: $entity_id})<-[:MENTIONS]-(d:Chunk)
        RETURN count(d) as mention_count
    """,
    "get_chunk_entities": """
        MATCH (d:Chunk {id_: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN e.id_ as entity_id
    """,
    "backtrack_chunks": """
        MATCH (e:Entity {id_: $entity_id})<-[:MENTIONS]-(d:Chunk)
        WHERE d.embedding IS NOT NULL
        RETURN d.id_ as chunk_id, d.content as content, d.embedding as embedding
        LIMIT $chunks_per_entity
    """,
}


def get_neo4j_query(query_type: str) -> str:
    try:
        return _NEO4J_QUERIES[query_type]
    except KeyError as exc:
        raise ValueError(f"Unknown Neo4j query type: {query_type}") from exc


__all__ = ["get_neo4j_query"]

