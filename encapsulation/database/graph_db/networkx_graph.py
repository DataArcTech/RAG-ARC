from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
import json
import pickle
import os
import networkx as nx
from pathlib import Path

import logging
from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Document, GraphData

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

        # Store documents separately for efficient retrieval
        self.documents = {}  # doc_id -> document data

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

    def build_index(self, documents: List[Document]) -> List[str]:
        """Build graph from a list of Documents."""
        return self.add_documents(documents)

    def update_index(self, documents: List[Document]) -> Optional[bool]:
        """Update existing documents' graphs in the database."""
        try:
            self.update_documents(documents)
            return True
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add multiple documents to the graph"""
        added_ids = []

        for doc in documents:
            try:
                # Check if document already exists
                if doc.id in self.documents:
                    logger.warning(f"Document with ID {doc.id} already exists, skipping...")
                    continue

                # Add document
                self.add_document(doc)

                # Add graph data if available
                if hasattr(doc, 'graph') and doc.graph:
                    self.add_graph_data(doc.graph, doc.id)

                added_ids.append(doc.id)
                logger.info(f"Successfully added document: {doc.id}")

            except Exception as e:
                logger.error(f"Failed to add document {doc.id}: {e}")
                continue

        # Auto-save if enabled and documents were added
        if added_ids:
            self._auto_save_if_enabled()

        return added_ids

    def add_document(self, document: Document) -> None:
        """Add document node to the graph"""
        doc_node_id = f"doc_{document.id}"
        
        # Add document node to NetworkX graph
        self.graph.add_node(
            doc_node_id,
            node_type='Document',
            id_=document.id,
            content=document.content,
            metadata=json.dumps(document.metadata, ensure_ascii=False) if document.metadata else "{}",
            create_time=datetime.now().isoformat(),
            update_time=datetime.now().isoformat()
        )
        
        # Store document data separately for efficient retrieval
        self.documents[document.id] = {
            'content': document.content,
            'metadata': document.metadata or {},
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        }

    def add_graph_data(self, graph_data: GraphData, doc_id: str) -> None:
        """Add graph data (entities and relations) to the graph"""
        doc_node_id = f"doc_{doc_id}"
        
        # Add entities
        entity_node_mapping = {}  # entity_name -> node_id for this document
        
        for entity in graph_data.entities:
            entity_id = doc_id + '_' + entity['id']  # Prefix with doc_id to avoid conflicts
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
                document_id=doc_id,
                create_time=datetime.now().isoformat(),
                update_time=datetime.now().isoformat(),
                attributes=json.dumps(entity['attributes'], ensure_ascii=False) if entity.get('attributes') else "{}"
            )
            
            # Store mapping for relation creation
            entity_node_mapping[entity['entity_name']] = entity_node_id
            
            # Create Document-Entity relationship
            self.graph.add_edge(
                doc_node_id,
                entity_node_id,
                relation_type='MENTIONS',
                create_time=datetime.now().isoformat()
            )

        # Add relations between entities
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head_name, relation_type, tail_name = relation[0], relation[1], relation[2]
                
                # Find entity nodes by name within this document
                head_node_id = entity_node_mapping.get(head_name)
                tail_node_id = entity_node_mapping.get(tail_name)
                
                if head_node_id and tail_node_id:
                    # Add relationship edge
                    self.graph.add_edge(
                        head_node_id,
                        tail_node_id,
                        relation_type=relation_type,
                        document_id=doc_id,
                        create_time=datetime.now().isoformat()
                    )
                else:
                    logger.warning(f"Could not find entities for relation: {head_name} -> {tail_name}")

    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete documents and their graphs by IDs. Delete all if ids is None."""
        if ids is None:
            raise ValueError("Dangerous operation: delete_index requires specific IDs. Use delete_all_documents() if you want to clear all data.")
        else:
            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))  # Preserves order, removes duplicates
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")
            return self.delete_documents(unique_ids)

    def delete_documents(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents and their graph data"""
        try:
            if not ids:
                raise ValueError("Must provide document IDs to delete")

            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")

            # Delete specific documents and their related data
            for doc_id in unique_ids:
                doc_node_id = f"doc_{doc_id}"
                
                # Find all entity nodes related to this document
                entity_nodes_to_remove = []
                for node_id, node_data in self.graph.nodes(data=True):
                    if (node_data.get('node_type') == 'Entity' and 
                        node_data.get('document_id') == doc_id):
                        entity_nodes_to_remove.append(node_id)
                
                # Remove entity nodes and their edges
                for entity_node_id in entity_nodes_to_remove:
                    if self.graph.has_node(entity_node_id):
                        self.graph.remove_node(entity_node_id)
                
                # Remove document node
                if self.graph.has_node(doc_node_id):
                    self.graph.remove_node(doc_node_id)
                
                # Remove from documents storage
                if doc_id in self.documents:
                    del self.documents[doc_id]
                
                logger.info(f"Deleted document: {doc_id}")

            # Auto-save if enabled
            self._auto_save_if_enabled()
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all documents and their graph data"""
        if not confirm:
            raise ValueError("Dangerous operation: delete_all_documents requires confirm=True")
        try:
            self.graph.clear()
            self.documents.clear()
            logger.info("Deleted all data from NetworkX graph")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all data: {e}")
            return False

    def update_documents(self, documents: List[Document]) -> None:
        """Update documents and their graph data"""
        for doc in documents:
            try:
                # Delete existing graph data for this document
                self.delete_documents([doc.id])

                # Add updated document and graph data
                self.add_documents([doc])

                logger.info(f"Successfully updated document: {doc.id}")

            except Exception as e:
                logger.error(f"Failed to update document {doc.id}: {e}")

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by IDs"""
        return self.get_documents(list(ids))

    def get_documents(self, ids: List[str]) -> List[Document]:
        """Retrieve documents by their IDs"""
        documents = []

        for doc_id in ids:
            try:
                # Check if document exists
                if doc_id not in self.documents:
                    logger.warning(f"⚠️ Document not found: {doc_id}")
                    continue

                doc_data = self.documents[doc_id]

                # Create document
                document = Document(
                    content=doc_data.get('content', ''),
                    id=doc_id,
                    metadata=doc_data.get('metadata', {})
                )

                # Get graph data
                document.graph = self.get_graph_data(doc_id)
                documents.append(document)

            except Exception as e:
                logger.error(f"Failed to get document {doc_id}: {e}")
                continue

        return documents

    def get_graph_data(self, doc_id: str) -> GraphData:
        """Get graph data for a specific document"""
        # Get entities for this document
        entities = []
        entity_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('node_type') == 'Entity' and 
                node_data.get('document_id') == doc_id):
                entity_nodes.append((node_id, node_data))
                
                entity = {
                    'id': node_data.get('id_', '').replace(f"{doc_id}_", ""),  # Remove doc_id prefix
                    'entity_name': node_data.get('entity_name', ''),
                    'entity_type': node_data.get('entity_type', ''),
                    'attributes': json.loads(node_data.get('attributes') or '{}')
                }
                entities.append(entity)

        # Get relations between entities in this document
        relations = []
        for edge in self.graph.edges(data=True):
            source, target, edge_data = edge
            
            # Check if this is an entity-entity relationship for this document
            if (edge_data.get('document_id') == doc_id and 
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

            # Save graph and documents
            graph_file = os.path.join(path, f"{name}_graph.pkl")
            docs_file = os.path.join(path, f"{name}_docs.pkl")

            with open(graph_file, 'wb') as f:
                pickle.dump(self.graph, f)

            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)

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
                self.documents = {}
                return

            with open(graph_file, 'rb') as f:
                self.graph = pickle.load(f)

            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)

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
        - 'documents': return all document IDs
        - 'entities': return all entity nodes
        - 'stats': return graph statistics
        """
        try:
            params = params or {}

            if query == 'nodes':
                return list(self.graph.nodes(data=True))
            elif query == 'edges':
                return list(self.graph.edges(data=True))
            elif query == 'documents':
                return list(self.documents.keys())
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
            # Count documents
            total_documents = len(self.documents)

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
                'total_documents': total_documents,
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
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'error': str(e)
            }
