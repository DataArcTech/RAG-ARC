import json
from core.file_management.extractor.Base_GraphExtractor import GraphExtractorBase
from core.prompts.prompt import HYPERRAG_EXTRACTION_PROMPT
from core.utils.data_model import Document
from encapsulation.llm.base import LLMBase
from typing import Any, Dict, List
import copy
import asyncio
import logging
import re

logger = logging.getLogger(__name__)

class HyperRAGGraphExtractor(GraphExtractorBase):
    """
    HyperRAGGraphExtractor用于提取完整的图结构，包括事件、实体和关系。
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        clean_prompt: str = None,
        max_rounds: int = 3,
        max_concurrent: int = 100,
        response_format: Any = None,
        clean_response_format: Any = None,
        enable_cleaning: bool = False
    ) -> None:
        """
        初始化HyperRAGGraphExtractor
        
        Args:
            llm: LLM实例
            extract_prompt: 提取提示模板。如果为None，则使用默认的HYPERRAG_EXTRACTION_PROMPT。
                            注意：模板中应包含 {text} 和 {history} 占位符。
            max_rounds: 最大提取轮次。
            max_concurrent: 最大并发数。
        """
        self.extract_prompt = extract_prompt if extract_prompt is not None else HYPERRAG_EXTRACTION_PROMPT
        self.response_format = response_format 
        self.clean_response_format = clean_response_format 
        self.clean_prompt = clean_prompt 

        super().__init__(   
            llm=llm,
            max_concurrent=max_concurrent,
            max_rounds=max_rounds,
            enable_cleaning=enable_cleaning  # This extractor does not have a cleaning step.
        )
    
    async def _aextract(
        self, 
        document: Document, 
        semaphore: asyncio.Semaphore, 
        history: Dict[str, List]
    ) -> Document:
        """
        从单个document中异步提取一轮图结构。
        多轮调用的逻辑由基类 `_aprocess_document` 处理。
        """
        async with semaphore:
            content = document.content
            if not content:
                return document

            history_str = json.dumps(history, indent=2, ensure_ascii=False)
            
            try:
                prompt = self.extract_prompt.format(text=content, history=history_str)
                messages = [{"role": "user", "content": prompt}]
                
                llm_response = await self.llm.aparse_chat(messages, response_format=self.response_format)
                
                if llm_response is None:
                    logger.warning("Warning: LLM returned an empty response.")
                    result = {}
                else:
                    result = llm_response.dict()
            
            except Exception as e:
                logger.error(f"An error occurred during graph extraction: {e}")
                result = {}

            document.metadata = {k: v for k, v in document.metadata.items() if k not in ['events', 'entities', 'event_relations', 'entity_relations']}

            events = result.get("events", [])
            document.metadata["events"] = events
            document.metadata["entities"] = result.get("entities", [])
            document.metadata["entity_relations"] = result.get("entity_relations", [])

            processed_event_relations = self._process_event_relations(
                result.get("event_relations", []), 
                events
            )
            document.metadata["event_relations"] = processed_event_relations
            
            return document


    async def _aprocess_document(self, document: Document, semaphore: asyncio.Semaphore) -> Document:
        current_doc = copy.deepcopy(document)
        history = self._init_extraction_history(current_doc)
        
        # 多轮提取
        for _ in range(self.max_rounds):
            logger.info(f"开始第{_}轮提取")
            extracted_doc = await self._aextract(current_doc, semaphore, history)
            history, added = self._merge_extraction_history(extracted_doc, history)
            current_doc.metadata['events'] = history['events']
            current_doc.metadata['entities'] = history['entities']
            current_doc.metadata['entity_relations'] = history['entity_relations']
            current_doc.metadata['event_relations'] = history['event_relations']

            if not added:
                logger.info("没有新增事件，结束提取")
                break

        if self.enable_cleaning:
            logger.info("开始清洗")
            cleaned_doc = await self._aclean(current_doc, semaphore)
            return cleaned_doc
        else:
            return current_doc
    
    def _init_extraction_history(self, document: Document) -> Dict[str, List]:
        """初始化抽取历史，从文档metadata中获取已抽取的实体和关系"""
        metadata = getattr(document, 'metadata', {})
        events = metadata.get('events', [])
        entities = metadata.get('entities', [])
        event_relations = metadata.get('event_relations', [])
        entity_relations = metadata.get('entity_relations', [])
        return {'events': list(events), 'entities': list(entities), 'event_relations': list(event_relations), 'entity_relations': list(entity_relations)}


    def _merge_extraction_history(self, document: Document, history: Dict[str, List]):
        """
        合并当前抽取结果与历史记录，自动去重。
        返回 (updated_history, added_flag)
        """
        # 确保 history 有默认结构
        history.setdefault('events', [])
        history.setdefault('entities', [])
        history.setdefault('entity_relations', [])
        history.setdefault('event_relations', [])

        new_events = document.metadata.get('events', []) or []
        new_entities = document.metadata.get('entities', []) or []
        new_e_rel = document.metadata.get('entity_relations', []) or []
        new_ev_rel = document.metadata.get('event_relations', []) or []

        # 已有 id/name 集合用于去重
        existing_event_ids = {e['id'] for e in history['events'] if 'id' in e}
        existing_entity_keys = {m.get('entity_name') for m in history['entities'] if 'entity_name' in m}

        added_events = []
        for e in new_events:
            if e.get('id') not in existing_event_ids:
                history['events'].append(e)
                existing_event_ids.add(e.get('id'))
                added_events.append(e)

        added_entities = []
        for m in new_entities:
            key = m.get('entity_name')
            if key not in existing_entity_keys:
                history['entities'].append(m)
                existing_entity_keys.add(key)
                added_entities.append(m)

        history['entity_relations'].extend([r for r in new_e_rel if r not in history['entity_relations']])
        history['event_relations'].extend([r for r in new_ev_rel if r not in history['event_relations']])

        added = bool(added_events or added_entities)
        return history, added


    def _process_event_relations(self, event_relations: List[Dict], events: List[Dict]) -> List[Dict]:
        """
        处理事件关系，将事件ID引用转换为具体的事件内容。
        """
        processed_relations = []
        if not event_relations or not events:
            return processed_relations

        event_map = {event['id']: event for event in events}

        for relation in event_relations:
            head_event_id = relation.get('head_event')
            tail_event_id = relation.get('tail_event')
            
            head_event = event_map.get(head_event_id)
            tail_event = event_map.get(tail_event_id)
            
            if head_event and tail_event:
                processed_relation = {
                    "head_event_content": head_event.get('content', ''),
                    "tail_event_content": tail_event.get('content', ''),
                    "relation_type": relation.get('relation_type', ''),
                    "description": relation.get('description', '')
                }
                processed_relations.append(processed_relation)
        
        return processed_relations


    async def _aclean(self, document: Document, semaphore: asyncio.Semaphore) -> Document:
        """
        异步清洗单个文档的图结构
        
        Args:
            document: 需要清洗的文档
            semaphore: 并发控制信号量
            
        Returns:
            清洗后的文档
        """
        async with semaphore:
            events = document.metadata.get("events", [])
            entities = document.metadata.get("entities", [])
            entity_relations = document.metadata.get("entity_relations", [])
            
            logger.debug(f"开始清洗文档，原始数据: {len(events)} events, {len(entities)} entities, {len(entity_relations)} entity_relations")

            # 使用线程池执行同步的清洗操作，避免阻塞事件循环
            cleaned_entities = await asyncio.to_thread(
                self._review_and_clean_entities, entities, document.content
            )
            
            # 更新文档的entities
            document.metadata["entities"] = cleaned_entities
            
            # 清洗实体关系
            cleaned_entity_relations = self._clean_entity_relations(entity_relations, cleaned_entities)
            document.metadata["entity_relations"] = cleaned_entity_relations
            
            logger.debug(f"清洗完成，结果: {len(events)} events, {len(cleaned_entities)} entities, {len(cleaned_entity_relations)} entity_relations")

            return document


    def _pre_filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        预处理过滤entities，去除明显无用的实体
        
        Args:
            entities: 原始entities列表
            
        Returns:
            预处理后的entities列表
        """
        if not entities:
            return []
        
        # 定义需要过滤的模式
        filter_patterns = [
            # 纯数字
            r'^\d+$',
            r'^\d+\.\d+$',
            r'^\d+年$',
            r'^\d+月$',
            r'^\d+日$',
            r'^\d+时$',
            r'^\d+分$',
            r'^\d+秒$',
            # 时间词汇
            r'^(昨天|今天|明天|上午|下午|晚上|现在|刚才|马上|立刻|立即)$',
            # 通用词汇
            r'^(这个|那个|什么|怎么|为什么|哪里|何时|如何|哪个|哪些)$',
            # 代词
            r'^(我|你|他|她|它|我们|你们|他们|她们|它们)$',
            # 量词
            r'^(一些|许多|几个|大量|少量|很多|很少|不少|不多)$',
            # 程度词
            r'^(很|非常|特别|极其|十分|相当|比较|稍微|略微)$',
            # 连接词
            r'^(和|或|但是|因为|所以|如果|虽然|尽管|然而|而且)$',
            # 标点符号
            r'^[！？。，；：""''（）【】《》\s]+$',
            # 单个字符（除了有意义的）
            r'^.$',
        ]
        
        filtered_entities = []
        
        for entity in entities:
            entity_name = entity.get('entity_name', '')
            
            # 跳过空字符串
            if not entity_name.strip():
                continue
                
            # 检查是否匹配过滤模式
            should_filter = False
            for pattern in filter_patterns:
                if re.match(pattern, entity_name.strip()):
                    should_filter = True
                    break
                    
            # 检查长度（太短的实体通常无意义）
            if len(entity_name.strip()) <= 1:
                should_filter = True
                
            # 检查是否全是数字和标点
            if re.match(r'^[\d\s\.,;:!?()\[\]{}""''\-_]+$', entity_name):
                should_filter = True
                
            if not should_filter:
                filtered_entities.append(entity)
        
        logger.info(f"预处理过滤entities: {len(entities)} -> {len(filtered_entities)}")
        return filtered_entities


    def _clean_entity_relations(self, entity_relations: List[Dict], cleaned_entities: List[Dict]) -> List[Dict]:
        """
        清洗实体关系，删除无效关系（包括头尾实体不存在、自环关系）
        
        Args:
            entity_relations: 原始实体关系列表
            cleaned_entities: 清洗后的entities列表
            
        Returns:
            清洗后的实体关系列表
        """
        if not entity_relations:
            return []
        
        # 创建entity_id到entity_name的映射
        id_to_entity_name = {}
        valid_entity_ids = set()
        
        for entity in cleaned_entities:
            entity_id = entity.get('id', '').strip()
            entity_name = entity.get('entity_name', '').strip()
            
            if entity_id and entity_name:
                id_to_entity_name[entity_id] = entity_name
                valid_entity_ids.add(entity_id)
        
        # 过滤有效的关系
        cleaned_relations = []
        removed_relations = []
        
        for relation in entity_relations:
            head_id = relation.get('head_id', '').strip()
            tail_id = relation.get('tail_id', '').strip()
            
            # 检查头尾实体ID是否存在于有效entity集合
            head_valid = head_id in valid_entity_ids
            tail_valid = tail_id in valid_entity_ids
            
            # 1. 过滤掉不存在的实体关系
            if not (head_valid and tail_valid):
                removed_relations.append({
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'relation_type': relation.get('relation_type', ''),
                    'reason': f"head_valid: {head_valid}, tail_valid: {tail_valid}"
                })
                continue
            
            # 2. 过滤掉自环关系（基于entity_id比较）
            if head_id == tail_id:
                removed_relations.append({
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'relation_type': relation.get('relation_type', ''),
                    'reason': "self_loop"
                })
                continue
            
            # 3. 检查是否为同一实体的不同ID（基于entity_name比较）
            head_name = id_to_entity_name.get(head_id, '')
            tail_name = id_to_entity_name.get(tail_id, '')
            
            if head_name.lower() == tail_name.lower():
                removed_relations.append({
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'relation_type': relation.get('relation_type', ''),
                    'reason': "same_entity_different_ids"
                })
                continue
            
            # 保留有效关系，并将ID转换为entity_name
            cleaned_relation = relation.copy()
            cleaned_relation['head_id'] = head_name
            cleaned_relation['tail_id'] = tail_name
            cleaned_relations.append(cleaned_relation)
        
        logger.info(f"清洗实体关系: {len(entity_relations)} -> {len(cleaned_relations)}")
        
        # 输出被删除的关系详情
        if removed_relations:
            logger.debug(f"删除了 {len(removed_relations)} 个无效关系:")
            for i, removed in enumerate(removed_relations[:5]):  # 只显示前5个
                logger.debug(f"  {i+1}. {removed['head_id']} -> {removed['tail_id']} ({removed['relation_type']}) - {removed['reason']}")
            if len(removed_relations) > 5:
                logger.debug(f"  ... 还有 {len(removed_relations) - 5} 个关系被删除")
        
        return cleaned_relations

    def _review_and_clean_entities(self, entities: List[Dict], document_content: str) -> List[Dict]:
        """
        使用LLM清洗和规范化entities
        
        Args:
            entities: 原始entities列表
            document_content: 文档内容，用于上下文
            
        Returns:
            清洗后的entities列表
        """
        if not entities:
            return []
        
        # 第一步：预处理过滤，去除明显无用的实体
        pre_filtered_entities = self._pre_filter_entities(entities)
        
        if not pre_filtered_entities:
            logger.info("预处理后没有剩余entities")
            return []
        
        # 使用配置的清洗提示词
        review_prompt = self.clean_prompt.format(
            document_content=document_content,
            entities_json=json.dumps(pre_filtered_entities, ensure_ascii=False, indent=2)
        )
        
        try:
            # 调用LLM进行清洗
            messages = [{"role": "user", "content": review_prompt}]
            logger.debug(f"开始调用LLM清洗entities，原始数量: {len(pre_filtered_entities)}")
            
            response = self.llm.parse_chat(messages, response_format=self.clean_response_format)
            
            # 解析LLM响应
            if response:
                logger.debug(f"LLM返回响应，entities数量: {len(response.entities)}")
                response_dicts = []
                for entity in response.entities:
                    response_dicts.append({
                        'id': entity.id,
                        'entity_name': entity.entity_name,
                        'entity_type': entity.entity_type,
                        'entity_description': entity.entity_description,
                        'event_indices': entity.event_indices
                    })
                
                logger.info(f"成功清洗entities: {len(entities)} -> {len(response_dicts)}")
                return response_dicts
            else:
                logger.warning("LLM响应格式异常，返回预处理后的entities")
                return pre_filtered_entities
                
        except Exception as e:
            logger.error(f"清洗entities时出错: {e}")
            return entities