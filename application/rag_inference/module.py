from chunk import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any
import logging
import uuid
import json
from framework.module import AbstractModule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.application.rag_inference_config import RAGInferenceConfig
 
class RAGInference(AbstractModule):
    def __init__(self, config: 'RAGInferenceConfig'):
        super().__init__(config=config)
        logger.info("Building query_rewriter...")
        self.query_rewriter = self.config.query_rewrite_config.build()
        logger.info("Query rewriter built successfully")
        
        logger.info("Building retriever...")
        self.retriever = self.config.retrieval_config.build()
        logger.info("Retriever built successfully")
        
        logger.info("Building reranker...")
        self.reranker = self.config.reranker_config.build()
        logger.info("Reranker built successfully")
        
        logger.info("Building llm...")
        self.llm = self.config.llm_config.build()
        logger.info("LLM built successfully")

    def chat(self, query: str, owner_id: uuid.UUID, return_subgraph: bool = False) -> tuple[str, list[Chunk], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval
            return_subgraph: If True, return subgraph visualization data

        Returns:
            Tuple of (LLM response, chunks, subgraph_data)
            - subgraph_data is None if return_subgraph=False or retriever doesn't support it
        """
        query = self.query_rewriter.rewrite_query(query)

        # Pass owner_id and return_subgraph_info to retriever
        # All retrievers support invoke() method which will handle these parameters
        chunks = self.retriever.invoke(
            query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph
        )

        # Extract subgraph info BEFORE reranking (to avoid losing it after reordering)
        subgraph_info = None
        if return_subgraph and chunks:
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata and '_subgraph_info' in chunk.metadata:
                    subgraph_info = chunk.metadata.pop('_subgraph_info')
                    logger.info("Extracted subgraph info before reranking")
                    break

        chunks = self.reranker.rerank(query, chunks)

        # Export subgraph data if subgraph_info is available
        # subgraph_data = None
        # if subgraph_info:
        #     # Import GraphExporter here to avoid circular dependency
        #     try:
        #         # Find graph_store from retriever or its children
        #         graph_store = None
        #         if hasattr(self.retriever, 'graph_store'):
        #             # Direct graph retriever
        #             graph_store = self.retriever.graph_store
        #         elif hasattr(self.retriever, 'config') and hasattr(self.retriever.config, 'built_retrievers'):
        #             # Multipath retriever: find graph retriever
        #             for child_retriever in self.retriever.config.built_retrievers:
        #                 if hasattr(child_retriever, 'graph_store'):
        #                     graph_store = child_retriever.graph_store
        #                     break

        #         if graph_store:
        #             # Import appropriate GraphExporter based on graph_store type
        #             # Check by class name to avoid import issues
        #             graph_store_class_name = graph_store.__class__.__name__

        #             if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
        #                 from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
        #                 # Neo4j version uses node IDs (strings)
        #                 subgraph_data = GraphExporter.export_subgraph(
        #                     graph_store=graph_store,
        #                     subgraph_node_ids=set(subgraph_info['subgraph_nodes']),
        #                     seed_entity_ids=set(subgraph_info['seed_entity_ids']),
        #                     retrieved_chunk_ids=subgraph_info['retrieved_chunk_ids'],
        #                     node_ppr_scores=subgraph_info.get('node_ppr_scores', {})
        #                 )
        #             else:
        #                 from encapsulation.database.utils.graph_export_utils import GraphExporter
        #                 # igraph version uses node indices (integers)
        #                 subgraph_data = GraphExporter.export_subgraph(
        #                     graph_store=graph_store,
        #                     subgraph_node_indices=set(subgraph_info['subgraph_nodes']),
        #                     seed_entity_ids=set(subgraph_info['seed_entity_ids']),
        #                     retrieved_chunk_ids=subgraph_info['retrieved_chunk_ids'],
        #                     node_ppr_scores=subgraph_info.get('node_ppr_scores', {})
        #                 )
        #             logger.info(f"Exported subgraph: {len(subgraph_data.get('nodes', []))} nodes, {len(subgraph_data.get('edges', []))} edges")
        #         else:
        #             logger.warning("Graph store not found in retriever")
        #     except Exception as e:
        #         logger.warning(f"Failed to export subgraph: {e}")
        #         import traceback
        #         logger.debug(f"Traceback: {traceback.format_exc()}")

        # Format chunks and query as messages
        messages = []
        prompt = """Please write your response in the following strict format.

<Write your detailed answer here.>
Do not include any concluding or summarizing phrases such as "in summary", "in conclusion", "to sum up", etc., in this section.
All summary or conclusion content must appear after the divider below.

---summary---
<Write a concise summary of the above content here.>

IMPORTANT:
- The line '---summary---' must appear exactly as shown (with three dashes before and after the word 'summary').
- Do not change, remove, or translate this divider.
- Do not include any extra text outside this format.
"""

        for i, chunk in enumerate(chunks):
            chunk_content = f"Chunk {i+1}:\n{chunk.content}"
            messages.append({"role": "user", "content": chunk_content})
        messages.append({"role": "user", "content": f"{prompt}\nBased on the above chunks, please answer question: {query}"})
        logger.info(f"Invoked chat with query: {query} (owner_id={owner_id})")
        logger.info(f"Query rewritten to: {self.query_rewriter.rewrite_query(query)}")
        logger.info(f"Retrieved chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Reranked chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Prepared messages for LLM: {messages}")
        response = self.llm.chat(messages)

        # Generate mind map data if return_subgraph is True
        subgraph_data = None
        if return_subgraph:
            subgraph_data = self._generate_mindmap(query, response, chunks)

        return (response, chunks, subgraph_data)

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
