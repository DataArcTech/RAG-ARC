from encapsulation.data_model.schema import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Iterator
import logging
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from framework.module import AbstractModule
from core.utils.owner_guard import normalize_owner_id, is_admin_owner, get_admin_owner_id
from framework.register import Register
from framework.thread_pool import get_thread_pool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.application.rag_inference_config import RAGInferenceConfig
    from application.knowledge.module import Knowledge
 
class RAGInference(AbstractModule):
    def __init__(self, config: 'RAGInferenceConfig'):
        super().__init__(config=config)
        logger.info("Building query_rewriter...")
        self.query_rewriter = self.config.query_rewrite_config.build()
        logger.info("Query rewriter built successfully")
        
        logger.info("Building retriever...")
        self.retriever = self.config.retrieval_config.build()
        logger.info("Retriever built successfully")
        self.graph_retriever = None
        graph_cls_name = "PrunedHippoRAGNeo4jRetriever"
        if hasattr(self.retriever, "retrievers"):
            for candidate in self.retriever.retrievers:
                if candidate.__class__.__name__ == graph_cls_name:
                    self.graph_retriever = candidate
                    logger.info("Detected graph retriever for admin/global access")
                    break
        
        logger.info("Building reranker...")
        self.reranker = self.config.reranker_config.build()
        logger.info("Reranker built successfully")
        
        logger.info("Building llm...")
        self.llm = self.config.llm_config.build()
        logger.info("LLM built successfully")
        self._knowledge_module: Optional["Knowledge"] = None

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a separate thread to avoid blocking the event loop."""
        return await get_thread_pool().run_blocking(func, *args, **kwargs)

    def chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system (synchronous version, kept for backward compatibility).
        For async non-blocking version, use chat_async().
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.chat_async(query, owner_id, return_subgraph))

        def _run_in_thread():
            return asyncio.run(self.chat_async(query, owner_id, return_subgraph))

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run_in_thread).result()

    async def chat_async(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval
        return_subgraph: If True, include serialized subgraph data in the response

        Returns:
            Tuple of (LLM response, chunks, subgraph_data, subgraph_info)
            - subgraph_data is None if return_subgraph=False or retriever doesn't support it
            - subgraph_info mirrors the retriever diagnostics used for graph export when available
        """
        messages, chunks, subgraph_data, subgraph_info = await self._run_blocking(
            self._build_messages_and_context,
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
        )
        response = await self._run_blocking(self.llm.chat, messages)

        if return_subgraph and subgraph_data is None:
            subgraph_data = await self._run_blocking(
                self._generate_mindmap,
                query,
                response,
                chunks,
            )
        return (response, chunks, subgraph_data, subgraph_info)

    def stream_chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
    ) -> tuple[Iterator[str], list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Stream chat completion from the configured LLM."""

        messages, chunks, subgraph_data, subgraph_info = self._build_messages_and_context(
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
        )
        return (self.llm.stream_chat(messages), chunks, subgraph_data, subgraph_info)

    def _build_messages_and_context(
        self,
        *,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool,
    ) -> tuple[List[Dict[str, str]], list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        rewritten_query = self.query_rewriter.rewrite_query(query)
        chunks: list[Chunk] = self.retriever.invoke(
            rewritten_query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph,
        )

        if owner_id is not None and is_admin_owner(owner_id) and not chunks and self.graph_retriever is not None:
            logger.info("Admin/global mode: multipath returned 0 results, falling back to graph retriever")
            graph_chunks = self.graph_retriever.invoke(
                rewritten_query,
                owner_id=owner_id,
                return_subgraph_info=return_subgraph,
            )
            if graph_chunks:
                chunks = graph_chunks

        chunks = self._filter_chunks_by_file_status(chunks)
        subgraph_info = self._consume_subgraph_info(chunks)
        chunks = self.reranker.rerank(rewritten_query, chunks)

        subgraph_data = None
        if subgraph_info and return_subgraph:
            try:
                graph_store = self._locate_graph_store()
                if graph_store:
                    graph_store_class_name = graph_store.__class__.__name__
                    if graph_store_class_name == "PrunedHippoRAGNeo4jStore":
                        from encapsulation.database.utils.graph_export_utils_neo4j import (
                            GraphExporterNeo4j as GraphExporter,
                        )

                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_ids=set(subgraph_info["subgraph_nodes"]),
                            seed_entity_ids=set(subgraph_info["seed_entity_ids"]),
                            retrieved_chunk_ids=subgraph_info["retrieved_chunk_ids"],
                            node_ppr_scores=subgraph_info.get("node_ppr_scores", {}),
                        )
                    else:
                        from encapsulation.database.utils.graph_export_utils import GraphExporter

                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_indices=set(subgraph_info["subgraph_nodes"]),
                            seed_entity_ids=set(subgraph_info["seed_entity_ids"]),
                            retrieved_chunk_ids=subgraph_info["retrieved_chunk_ids"],
                            node_ppr_scores=subgraph_info.get("node_ppr_scores", {}),
                        )
                    if subgraph_data is not None:
                        logger.info(
                            "Exported subgraph: %d nodes, %d edges",
                            len(subgraph_data.get("nodes", [])),
                            len(subgraph_data.get("edges", [])),
                        )
                else:
                    logger.warning("Graph store not found in retriever")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to export subgraph: %s", exc)
                import traceback
                logger.debug("Traceback: %s", traceback.format_exc())

        messages: List[Dict[str, str]] = []
        for i, chunk in enumerate(chunks):
            messages.append({"role": "user", "content": f"Chunk {i+1}:\n{chunk.content}"})
        messages.append(
            {
                "role": "user",
                "content": f"Based on the above chunks, please answer question: {rewritten_query}",
            }
        )
        logger.info("Invoked chat with query: %s (owner_id=%s)", query, owner_id)
        logger.info("Query rewritten to: %s", rewritten_query)
        logger.info("Prepared %d messages for LLM", len(messages))
        return (messages, chunks, subgraph_data, subgraph_info)

    def _locate_graph_store(self):
        """Locate the configured graph store if one exists."""
        candidate_retrievers: List[Any] = []
        if self.graph_retriever is not None:
            candidate_retrievers.append(self.graph_retriever)

        if hasattr(self.retriever, 'graph_store'):
            candidate_retrievers.append(self.retriever)

        if hasattr(self.retriever, 'retrievers'):
            candidate_retrievers.extend(self.retriever.retrievers or [])

        nested = getattr(getattr(self.retriever, 'config', None), 'built_retrievers', None)
        if nested:
            candidate_retrievers.extend(nested)

        for retriever in candidate_retrievers:
            graph_store = getattr(retriever, 'graph_store', None)
            if graph_store is not None:
                return graph_store
        return None

    def get_graph_store(self):
        """Expose the underlying graph store for CLI and admin APIs."""
        return self._locate_graph_store()

    def _get_knowledge_module(self) -> Optional["Knowledge"]:
        if self._knowledge_module is None:
            try:
                registrator = Register()
                knowledge_module = registrator.get_object("knowledge")
                self._knowledge_module = knowledge_module
            except Exception as e:
                logger.warning(f"Failed to locate knowledge module for file filtering: {e}")
                self._knowledge_module = None
        return self._knowledge_module

    def _filter_chunks_by_file_status(self, chunks: List[Chunk]) -> List[Chunk]:
        knowledge_module = self._get_knowledge_module()
        if knowledge_module is None:
            return chunks

        filtered_chunks: List[Chunk] = []
        for chunk in chunks:
            file_id = None
            if hasattr(chunk, "metadata") and chunk.metadata:
                file_id = chunk.metadata.get("source_file_id")

            if not file_id or knowledge_module.is_file_active(file_id):
                filtered_chunks.append(chunk)
        if len(filtered_chunks) != len(chunks):
            logger.info(f"Filtered out {len(chunks) - len(filtered_chunks)} chunks from deleting files")
        return filtered_chunks

    @staticmethod
    def _consume_subgraph_info(chunks: List[Chunk]) -> Optional[Dict[str, Any]]:
        """Pop and return embedded subgraph diagnostics when available."""

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None) or {}
            if "_subgraph_info" in metadata:
                info = metadata.pop("_subgraph_info")
                logger.info("Extracted subgraph info before reranking")
                return info
        return None

    def export_graph_overview(
        self,
        owner_id: Optional[uuid.UUID],
        max_nodes: int = 1000,
        max_edges: int = 5000,
        include_node_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export a graph overview for visual inspection.

        Args:
            owner_id: Optional owner scope. None returns all owners (admin use).
            max_nodes: Maximum number of nodes to sample.
            max_edges: Maximum number of edges to sample.
            include_node_types: Optional white-list of node types.

        Returns:
            Graph overview payload compatible with frontend graph viewers.
        """
        graph_store = self._locate_graph_store()
        if graph_store is None:
            raise RuntimeError("Graph store is not configured for the current retriever profile")

        normalized_owner = normalize_owner_id(owner_id) if owner_id is not None else None
        owner_scope_label = normalized_owner
        if owner_scope_label is None:
            owner_scope_label = get_admin_owner_id() or "GLOBAL_ADMIN"
        graph_store_class_name = graph_store.__class__.__name__

        if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
            from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
        else:
            from encapsulation.database.utils.graph_export_utils import GraphExporter

        overview = GraphExporter.export_full_graph(
            graph_store=graph_store,
            max_nodes=max_nodes,
            max_edges=max_edges,
            include_node_types=include_node_types,
            owner_id=normalized_owner,
            owner_scope_label=owner_scope_label,
        )
        logger.info(
            "Exported graph overview (owner_scope=%s) with %d nodes and %d edges",
            normalized_owner,
            len(overview.get('nodes', [])) + len(overview.get('chunks', [])),
            len(overview.get('edges', [])),
        )
        return overview

    def _generate_mindmap(self, query: str, response: str, chunks: list[Chunk]) -> Dict[str, Any]:
        """
        Generate mind map data based on query, response and chunks

        Args:
            query: User query
            response: LLM response
            chunks: Retrieved chunks

        Returns:
            Dictionary containing chunks, nodes and edges for mind map visualization
        """
        try:
            # Prepare prompt for LLM to generate mind map structure
            mindmap_prompt = self._build_mindmap_prompt(query, response, chunks)

            # Call LLM to generate nodes and edges
            mindmap_messages = [
                {"role": "system", "content": "你是一个专业的思维导图生成助手。请根据用户的问题、回答和检索到的文档片段,生成结构化的思维导图数据。"},
                {"role": "user", "content": mindmap_prompt}
            ]

            logger.info("Generating mind map structure with LLM...")
            # Note: This is called from _run_blocking, so it's already in a thread
            mindmap_response = self.llm.chat(mindmap_messages)

            # Parse LLM response to extract JSON
            mindmap_json = self._extract_json_from_response(mindmap_response)

            # Build final subgraph data with chunks
            subgraph_data = self._build_subgraph_data(chunks, mindmap_json)

            logger.info(f"Generated mind map: {len(subgraph_data.get('nodes', []))} nodes, {len(subgraph_data.get('edges', []))} edges")
            return subgraph_data

        except Exception as e:
            logger.error(f"Failed to generate mind map: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"chunks": [], "nodes": [], "edges": []}

    def _build_mindmap_prompt(self, query: str, response: str, chunks: list[Chunk]) -> str:
        """Build prompt for LLM to generate mind map structure"""
        chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk.content}" for i, chunk in enumerate(chunks)])

        prompt = f"""请基于以下信息生成思维导图的节点(nodes)和边(edges)数据:

用户问题: {query}

回答内容: {response}

检索到的文档片段:
{chunks_text}

请生成一个JSON格式的思维导图结构,包含以下字段:
1. nodes: 节点数组,每个节点包含:
   - id: 节点唯一标识
   - name: 节点名称
   - category: 节点分类
   - weight: 节点深度(1为根节点,2为二级节点,3为三级节点,以此类推)

2. edges: 边数组,每个边包含:
   - id: 边的唯一标识
   - weight: 边的权重(0-1之间的浮点数)
   - source: 源节点id
   - target: 目标节点id
   - relation: 关系类型(如"包含"、"说明"、"步骤"、"依据"等)

要求:
- 根节点(weight=1)应该是对整个回答的总结
- 二级节点(weight=2)是主要的知识点或步骤
- 三级节点(weight=3)是详细的说明或子步骤
- 边的source应该是父节点,target应该是子节点
- 边的weight建议: 一级到二级为0.85,二级到三级为0.8
- 所有节点的id和name要清晰明确,便于理解
- 对于根节点和二级节点(weight=1和2)的category都是其自己的name，三级节点及更高节点的都和父节点的name相同。

请直接返回JSON格式的数据,不要包含任何其他说明文字。格式如下:
{{
  "nodes": [
    {{"id": "根节点标题", "name": "根节点标题", "category": "根节点标题", "weight": 1}},
    {{"id": "二级节点1", "name": "二级节点1", "category": "二级节点1", "weight": 2}},
    {{"id": "三级节点1", "name": "三级节点1", "category": "二级节点1", "weight": 3}}
  ],
  "edges": [
    {{"id": "edge-001", "weight": 0.85, "source": "根节点标题", "target": "二级节点1", "relation": "包含"}},
    {{"id": "edge-002", "weight": 0.8, "source": "二级节点1", "target": "三级节点1", "relation": "说明"}}
  ]
}}
"""
        return prompt

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON block in markdown code fence
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to parse the entire response as JSON
                json_str = response.strip()

            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            logger.debug(f"Response: {response}")
            return {"nodes": [], "edges": []}

    def _build_subgraph_data(self, chunks: list[Chunk], mindmap_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build final subgraph data combining chunks and mind map structure"""
        # Build chunks data
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk-{800 + i + 1}"
            # Try to infer chunk type from content or use default
            chunk_type = "检索片段"
            chunks_data.append({
                "id": chunk_id,
                "type": chunk_type,
                "content": chunk.content
            })

        # Combine with mind map nodes and edges
        return {
            "chunks": chunks_data,
            "nodes": mindmap_json.get("nodes", []),
            "edges": mindmap_json.get("edges", [])
        }
