from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
import json
import pickle
import os
import networkx as nx
from pathlib import Path

import logging
from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Chunk, GraphData

logger = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """NetworkX Graph Store Implementation with BaseIndex interface"""

    def __init__(self, config):
        """Initialize NetworkX graph store"""
        self.config = config

        # Initialize NetworkX graph based on configuration
        if getattr(config, 'allow_parallel_edges', True):
            if getattr(config, 'allow_self_loops', True):
                self.graph = nx.MultiDiGraph()
            else:
                self.graph = nx.MultiDiGraph()
                # Note: NetworkX doesn't have built-in self-loop prevention,
                # we'll handle this in add operations
        else:
            if getattr(config, 'allow_self_loops', True):
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.DiGraph()

        # Store chunks separately for efficient retrieval
        self.chunks = {}  # chunk_id -> chunk data

        # Auto-save configuration
        self.auto_save = getattr(config, 'auto_save', False)
        self.storage_path = getattr(config, 'storage_path', None)
        self.index_name = getattr(config, 'index_name', 'networkx_index')

        # Load existing data if storage path is provided
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                self.load_index(self.storage_path, self.index_name)
                logger.info(f"Loaded existing graph from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Could not load existing graph: {e}")

        logger.info("Successfully initialized NetworkX graph store")

    def close(self):
        """Close graph store and auto-save if configured"""
        if self.auto_save and self.storage_path:
            try:
                self.save_index(self.storage_path, self.index_name)
                logger.info(f"Auto-saved graph to {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to auto-save graph: {e}")
        logger.info("NetworkX graph store closed")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method"""
        # Suppress unused parameter warnings
        _ = exc_type, exc_val, exc_tb
        self.close()

    def _auto_save_if_enabled(self):
        """Auto-save graph if enabled"""
        if self.auto_save and self.storage_path:
            try:
                self.save_index(self.storage_path, self.index_name)
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    # =============================================================================
    # BaseIndex Interface Implementation
    # =============================================================================

    def build_index(self, chunks: List[Chunk]) -> List[str]:
        """Build graph from a list of Chunks."""
        return self.add_chunks(chunks)

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update existing chunks' graphs in the database."""
        try:
            self.update_chunks(chunks)
            return True
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Add multiple chunks to the graph"""
        added_ids = []

        for chunk in chunks:
            try:
                # Check if chunk already exists
                if chunk.id in self.chunks:
                    logger.warning(f"Chunk with ID {chunk.id} already exists, skipping...")
                    continue

                # Add chunk
                self.add_chunk(chunk)

                # Add graph data if available
                if hasattr(chunk, 'graph') and chunk.graph:
                    self.add_graph_data(chunk.graph, chunk.id)

                added_ids.append(chunk.id)
                logger.info(f"Successfully added chunk: {chunk.id}")

            except Exception as e:
                logger.error(f"Failed to add chunk {chunk.id}: {e}")
                continue

        # Auto-save if enabled and chunks were added
        if added_ids:
            self._auto_save_if_enabled()

        return added_ids

    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk node to the graph"""
        chunk_node_id = f"chunk_{chunk.id}"

        # Add chunk node to NetworkX graph
        self.graph.add_node(
            chunk_node_id,
            node_type='Chunk',
            id_=chunk.id,
            content=chunk.content,
            metadata=json.dumps(chunk.metadata, ensure_ascii=False) if chunk.metadata else "{}",
            create_time=datetime.now().isoformat(),
            update_time=datetime.now().isoformat()
        )

        # Store chunk data separately for efficient retrieval
        self.chunks[chunk.id] = {
            'content': chunk.content,
            'metadata': chunk.metadata or {},
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        }

    def add_graph_data(self, graph_data: GraphData, chunk_id: str) -> None:
        """Add graph data (entities and relations) to the graph"""
        chunk_node_id = f"chunk_{chunk_id}"
        
        # Add entities
        entity_node_mapping = {}  # entity_name -> node_id for this chunk
        
        for entity in graph_data.entities:
            entity_id = chunk_id + '_' + entity['id']  # Prefix with chunk_id to avoid conflicts
            entity_node_id = f"entity_{entity_id}"
            entity_type = entity.get('entity_type', 'Entity')
            
            # Add entity node
            self.graph.add_node(
                entity_node_id,
                node_type='Entity',
                entity_subtype=entity_type,
                id_=entity_id,
                entity_name=entity['entity_name'],
                entity_type=entity_type,
                chunk_id=chunk_id,
                create_time=datetime.now().isoformat(),
                update_time=datetime.now().isoformat(),
                attributes=json.dumps(entity['attributes'], ensure_ascii=False) if entity.get('attributes') else "{}"
            )
            
            # Store mapping for relation creation
            entity_node_mapping[entity['entity_name']] = entity_node_id
            
            # Create Chunk-Entity relationship
            self.graph.add_edge(
                chunk_node_id,
                entity_node_id,
                relation_type='MENTIONS',
                create_time=datetime.now().isoformat()
            )

        # Add relations between entities
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head_name, relation_type, tail_name = relation[0], relation[1], relation[2]
                
                # Find entity nodes by name within this chunk
                head_node_id = entity_node_mapping.get(head_name)
                tail_node_id = entity_node_mapping.get(tail_name)
                
                if head_node_id and tail_node_id:
                    # Add relationship edge
                    self.graph.add_edge(
                        head_node_id,
                        tail_node_id,
                        relation_type=relation_type,
                        chunk_id=chunk_id,
                        create_time=datetime.now().isoformat()
                    )
                else:
                    logger.warning(f"Could not find entities for relation: {head_name} -> {tail_name}")

    def delete_index(self, ids: Optional[List[str]] = None) -> bool:
        """Delete chunks and their graph data"""
        try:
            if not ids:
                raise ValueError("Must provide chunk IDs to delete")

            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")

            # Delete specific chunks and their related data
            for chunk_id in unique_ids:
                chunk_node_id = f"chunk_{chunk_id}"

                # Find all entity nodes related to this chunk
                entity_nodes_to_remove = []
                for node_id, node_data in self.graph.nodes(data=True):
                    if (node_data.get('node_type') == 'Entity' and 
                        node_data.get('chunk_id') == chunk_id):
                        entity_nodes_to_remove.append(node_id)
                
                # Remove entity nodes and their edges
                for entity_node_id in entity_nodes_to_remove:
                    if self.graph.has_node(entity_node_id):
                        self.graph.remove_node(entity_node_id)
                
                # Remove chunk node
                if self.graph.has_node(chunk_node_id):
                    self.graph.remove_node(chunk_node_id)
                
                # Remove from chunks storage
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]

                logger.info(f"Deleted chunk: {chunk_id}")

            # Auto-save if enabled
            self._auto_save_if_enabled()
            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graph data"""
        if not confirm:
            raise ValueError("Dangerous operation: delete_all_chunks requires confirm=True")
        try:
            self.graph.clear()
            self.chunks.clear()
            logger.info("Deleted all data from NetworkX graph")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all data: {e}")
            return False

    def update_chunks(self, chunks: List[Chunk]) -> None:
        """Update chunks and their graph data"""
        for chunk in chunks:
            try:
                # Delete existing graph data for this chunk
                self.delete_index([chunk.id])

                # Add updated chunk and graph data
                self.add_chunks([chunk])

                logger.info(f"Successfully updated chunk: {chunk.id}")

            except Exception as e:
                logger.error(f"Failed to update chunk {chunk.id}: {e}")

    def get_by_ids(self, ids: Sequence[str]) -> List[Chunk]:
        """Retrieve chunks (including their graphs) by IDs."""
        ids_list = list(ids)
        return self._get_chunks_impl(ids_list)

    def get_chunks(self, ids: List[str]) -> List[Chunk]:
        """Deprecated: Use get_by_ids instead"""
        return self._get_chunks_impl(ids)

    def _get_chunks_impl(self, ids: List[str]) -> List[Chunk]:
        """Retrieve chunks by their IDs"""
        chunks = []

        for chunk_id in ids:
            try:
                # Check if chunk exists
                if chunk_id not in self.chunks:
                    logger.warning(f"⚠️ Chunk not found: {chunk_id}")
                    continue

                chunk_data = self.chunks[chunk_id]

                # Create chunk
                chunk = Chunk(
                    content=chunk_data.get('content', ''),
                    id=chunk_id,
                    metadata=chunk_data.get('metadata', {})
                )

                # Get graph data
                chunk.graph = self.get_graph_data(chunk_id)
                chunks.append(chunk)

            except Exception as e:
                logger.error(f"Failed to get chunk {chunk_id}: {e}")
                continue

        return chunks

    def get_graph_data(self, chunk_id: str) -> GraphData:
        """Get graph data for a specific chunk"""
        # Get entities for this chunk
        entities = []
        entity_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('node_type') == 'Entity' and 
                node_data.get('chunk_id') == chunk_id):
                entity_nodes.append((node_id, node_data))
                
                entity = {
                    'id': node_data.get('id_', '').replace(f"{chunk_id}_", ""),  # Remove chunk_id prefix
                    'entity_name': node_data.get('entity_name', ''),
                    'entity_type': node_data.get('entity_type', ''),
                    'attributes': json.loads(node_data.get('attributes') or '{}')
                }
                entities.append(entity)

        # Get relations between entities in this chunk
        relations = []
        for edge in self.graph.edges(data=True):
            source, target, edge_data = edge
            
            # Check if this is an entity-entity relationship for this chunk
            if (edge_data.get('chunk_id') == chunk_id and 
                edge_data.get('relation_type') != 'MENTIONS'):
                
                # Get entity names from nodes
                source_data = self.graph.nodes[source]
                target_data = self.graph.nodes[target]
                
                if (source_data.get('node_type') == 'Entity' and 
                    target_data.get('node_type') == 'Entity'):
                    relations.append([
                        source_data.get('entity_name', ''),
                        edge_data.get('relation_type', ''),
                        target_data.get('entity_name', '')
                    ])

        return GraphData(entities=entities, relations=relations, metadata={})

    def save_index(self, path: str, name: str = "index") -> None:
        """Persist the graph database to filesystem using pickle"""
        try:
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)

            # Save graph and chunks
            graph_file = os.path.join(path, f"{name}_graph.pkl")
            docs_file = os.path.join(path, f"{name}_docs.pkl")

            with open(graph_file, 'wb') as f:
                pickle.dump(self.graph, f)

            with open(docs_file, 'wb') as f:
                pickle.dump(self.chunks, f)

            logger.info(f"Successfully saved NetworkX graph to {path}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, path: str, name: str = "index") -> None:
        """Load persisted graph database from filesystem"""
        try:
            graph_file = os.path.join(path, f"{name}_graph.pkl")
            docs_file = os.path.join(path, f"{name}_docs.pkl")

            if not os.path.exists(graph_file) or not os.path.exists(docs_file):
                logger.warning(f"Index files not found in {path}, starting with empty graph")
                self.graph = nx.MultiDiGraph()
                self.chunks = {}
                return

            with open(graph_file, 'rb') as f:
                self.graph = pickle.load(f)

            with open(docs_file, 'rb') as f:
                self.chunks = pickle.load(f)

            logger.info(f"Successfully loaded NetworkX graph from {path}")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Run a query on the graph database.

        For NetworkX, this provides basic graph querying capabilities.
        The query parameter can be:
        - 'nodes': return all nodes
        - 'edges': return all edges
        - 'chunks': return all chunk IDs
        - 'entities': return all entity nodes
        - 'stats': return graph statistics
        """
        try:
            params = params or {}

            if query == 'nodes':
                return list(self.graph.nodes(data=True))
            elif query == 'edges':
                return list(self.graph.edges(data=True))
            elif query == 'chunks':
                return list(self.chunks.keys())
            elif query == 'entities':
                return [(node_id, data) for node_id, data in self.graph.nodes(data=True)
                       if data.get('node_type') == 'Entity']
            elif query == 'stats':
                return self.get_graph_db_info()
            else:
                logger.warning(f"Unsupported query type: {query}")
                return None

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Health check for NetworkX graph store"""
        try:
            # Basic health check - verify graph is accessible
            _ = self.graph.number_of_nodes()
            _ = self.graph.number_of_edges()

            # Get basic statistics
            stats = self.get_graph_db_info()

            return {
                "status": "healthy",
                "graph_type": "NetworkX",
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_graph_db_info(self) -> Dict[str, Any]:
        """Return statistics or metadata about the graph database."""
        try:
            # Count chunks
            total_chunks = len(self.chunks)

            # Count entities
            total_entities = sum(1 for _, data in self.graph.nodes(data=True)
                               if data.get('node_type') == 'Entity')

            # Count relationships (excluding MENTIONS relationships)
            total_relationships = sum(1 for _, _, data in self.graph.edges(data=True)
                                    if data.get('relation_type') != 'MENTIONS')

            # Count MENTIONS relationships
            mentions_relationships = sum(1 for _, _, data in self.graph.edges(data=True)
                                       if data.get('relation_type') == 'MENTIONS')

            return {
                'total_chunks': total_chunks,
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'mentions_relationships': mentions_relationships,
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'graph_type': 'NetworkX MultiDiGraph'
            }

        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                'total_chunks': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'error': str(e)
            }
