import json
import os
import sys
import tempfile
import shutil
from typing import List

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig, BM25IndexBuilder


def load_test_data() -> List[Document]:
    """加载测试数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, "../../tcl_gb_chunk.json")
    
    print("Loading documents...")
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            docs_list = []
            for i, doc in enumerate(docs):  # 限制测试数据量
                docs_list.append(Document(
                    id=f"doc_{i}",
                    content=doc["content"], 
                    metadata=doc["metadata"]
                ))
        
        print(f"Loaded {len(docs_list)} documents")
        return docs_list
        
    except FileNotFoundError:
        # 如果测试数据文件不存在，创建模拟数据
        print("Test data file not found, creating mock data...")
        return create_mock_data()


def create_mock_data() -> List[Document]:
    """创建模拟测试数据"""
    mock_docs = []
    contents = [
        "这是一个关于冷凝器设计的技术文档，介绍了冷凝器的基本原理和设计要点。",
        "蒸发器是制冷系统的重要组成部分，其性能直接影响整个系统的效率。",
        "制冷剂在蒸发器中发生相变，从液态变为气态，吸收大量热量。",
        "冷凝器片距的选择需要考虑传热效率、压降和制造成本等因素。",
        "换热器的设计需要综合考虑传热、流动和经济性等多个方面。",
        "制冷系统的能效比是评价系统性能的重要指标之一。",
        "压缩机是制冷系统的心脏，其性能决定了整个系统的运行效果。",
        "制冷剂的选择需要考虑环保性、安全性和热物性等因素。",
        "热泵系统可以实现供暖和制冷的双重功能，具有很好的节能效果。",
        "变频技术在制冷系统中的应用可以显著提高系统的能效。"
    ]
    
    for i, content in enumerate(contents):
        metadata = {
            "source": f"document_{i}.pdf",
            "category": "technical" if i % 2 == 0 else "general",
            "author": f"author_{i % 3}",
            "region": "北京" if i % 3 == 0 else ("上海" if i % 3 == 1 else "深圳")
        }
        mock_docs.append(Document(
            id=f"doc_{i}",
            content=content,
            metadata=metadata
        ))
    
    print(f"Created {len(mock_docs)} mock documents")
    return mock_docs


def test_bm25_index_builder():
    """合并所有测试步骤的完整测试函数"""
    print("开始运行 BM25IndexBuilder 完整测试（合并版本）")
    print("=" * 60)
    
    passed_steps = []
    failed_steps = []
    
    # Step 1: 测试配置验证功能
    print("\n=== 步骤1: 测试配置验证功能 ===")
    try:
        # 测试必需的 index_path
        try:
            config = BM25IndexBuilderConfig()
            print("❌ 应该要求 index_path 参数")
            failed_steps.append("config_validation_index_path")
        except Exception as e:
            print(f"✅ 正确验证 index_path 必需: {type(e).__name__}")
            passed_steps.append("config_validation_index_path")
        
        # 测试有效配置
        try:
            config = BM25IndexBuilderConfig(
                index_path="./test_index",
                bm25_k1=1.5,
                bm25_b=0.8,
                batch_size=100
            )
            print(f"✅ 有效配置创建成功: {config}")
            passed_steps.append("config_validation_valid_config")
        except Exception as e:
            print(f"❌ 有效配置创建失败: {e}")
            failed_steps.append("config_validation_valid_config")
        
        # 测试无效参数
        try:
            config = BM25IndexBuilderConfig(
                index_path="./test_index",
                bm25_k1=-1.0  # 无效值
            )
            print("❌ 应该验证 bm25_k1 > 0")
            failed_steps.append("config_validation_invalid_k1")
        except ValueError as e:
            print(f"✅ 正确验证 bm25_k1: {e}")
            passed_steps.append("config_validation_invalid_k1")
        
        try:
            config = BM25IndexBuilderConfig(
                index_path="./test_index",
                bm25_b=1.5  # 无效值
            )
            print("❌ 应该验证 bm25_b 在 [0,1] 范围")
            failed_steps.append("config_validation_invalid_b")
        except ValueError as e:
            print(f"✅ 正确验证 bm25_b: {e}")
            passed_steps.append("config_validation_invalid_b")
            
        print("✅ 步骤1: 配置验证功能测试完成")
        passed_steps.append("config_validation_overall")
    except Exception as e:
        print(f"❌ 步骤1失败: {e}")
        failed_steps.append("config_validation_overall")
    
    # 加载测试数据
    docs_list = load_test_data()
    
    # Step 2: 测试 load_local 方法
    print("\n=== 步骤2: 测试 load_local 方法 ===")
    try:
        test_docs = docs_list[:5]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "load_local_index")
            
            # 首先创建一个索引
            config = BM25IndexBuilderConfig(index_path=index_path)
            builder1 = BM25IndexBuilder(config=config)
            builder1.from_documents(test_docs)
            builder1.close()
            print("✅ 初始索引创建完成")
            
            # 测试加载现有索引
            config2 = BM25IndexBuilderConfig(index_path=index_path)
            builder2 = config2.build()  # 通过build()获取实例，但不加载索引
            
            # 此时应该无法使用索引功能
            try:
                builder2.as_retriever()
                print("❌ 应该要求先加载索引")
                failed_steps.append("load_local_before_load")
            except RuntimeError as e:
                print(f"✅ 正确要求先加载索引: {e}")
                passed_steps.append("load_local_before_load")
            
            # 加载索引
            result_builder = builder2.load_local()
            if result_builder is builder2:
                print("✅ load_local 返回正确的实例")
                passed_steps.append("load_local_return_self")
            else:
                print("❌ load_local 应该返回自身")
                failed_steps.append("load_local_return_self")
            
            # 现在可以使用索引功能
            retriever = builder2.as_retriever()
            results = retriever.invoke("测试", k=3)
            print(f"✅ 加载后检索功能正常，结果数量: {len(results)}")
            passed_steps.append("load_local_retrieval_works")
            
            # 测试加载后不能使用from_documents
            try:
                builder2.from_documents(test_docs)
                print("❌ 加载后应该不能使用from_documents")
                failed_steps.append("load_local_from_documents_after_load")
            except RuntimeError as e:
                print(f"✅ 正确拒绝加载后使用from_documents: {e}")
                passed_steps.append("load_local_from_documents_after_load")
            
            builder2.close()
            
            # 测试加载不存在的索引
            try:
                config3 = BM25IndexBuilderConfig(index_path=index_path + "_nonexistent")
                builder3 = config3.build()
                builder3.load_local()
                print("❌ 应该拒绝不存在的索引路径")
                failed_steps.append("load_local_nonexistent_index")
            except FileNotFoundError as e:
                print(f"✅ 正确拒绝不存在的索引: {e}")
                passed_steps.append("load_local_nonexistent_index")
        
        print("✅ 步骤2: load_local 方法测试完成")
        passed_steps.append("load_local_overall")
    except Exception as e:
        print(f"❌ 步骤2失败: {e}")
        failed_steps.append("load_local_overall")
    
    # Step 3: 测试 from_documents 实例方法
    print("\n=== 步骤3: 测试 from_documents 实例方法 ===")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "from_documents_index")
            
            # 测试正常创建
            config = BM25IndexBuilderConfig(
                index_path=index_path,
                batch_size=500,
                max_workers=2
            )
            builder = config.build()  # 获取实例但不加载索引
            
            # 此时应该无法使用索引功能
            try:
                builder.as_retriever()
                print("❌ 应该要求先构建索引")
                failed_steps.append("from_documents_before_build")
            except RuntimeError as e:
                print(f"✅ 正确要求先构建索引: {e}")
                passed_steps.append("from_documents_before_build")
            
            # 构建索引
            result_builder = builder.from_documents(docs_list)
            
            # 验证返回的是同一个实例
            if result_builder is builder:
                print("✅ from_documents 返回正确的实例")
                passed_steps.append("from_documents_return_self")
            else:
                print("❌ from_documents 应该返回自身")
                failed_steps.append("from_documents_return_self")
            
            # 测试索引统计
            stats = builder.get_index_stats()
            print(f"✅ 索引统计: {stats}")
            passed_steps.append("from_documents_stats")
            
            # 测试检索功能
            retriever = builder.as_retriever()
            results = retriever.invoke("冷凝器", k=3)
            print(f"✅ 检索结果数量: {len(results)}")
            passed_steps.append("from_documents_retrieval")
            
            builder.close()
            
            # 测试空文档列表
            try:
                config = BM25IndexBuilderConfig(index_path=index_path + "_empty")
                builder = config.build()
                builder.from_documents([])
                print("❌ 应该拒绝空文档列表")
                failed_steps.append("from_documents_empty_list")
            except ValueError as e:
                print(f"✅ 正确拒绝空文档列表: {e}")
                passed_steps.append("from_documents_empty_list")
            
            # 测试重复调用from_documents（应该报错）
            try:
                config = BM25IndexBuilderConfig(index_path=index_path + "_repeat")
                builder = config.build()
                builder.from_documents(docs_list[:5])
                print("✅ 第一次from_documents成功")
                passed_steps.append("from_documents_first_call")
                
                # 再次调用应该报错
                try:
                    builder.from_documents(docs_list[5:10])
                    print("❌ 重复调用from_documents应该报错")
                    failed_steps.append("from_documents_repeat_call")
                except RuntimeError as e:
                    print(f"✅ 正确拒绝重复调用from_documents: {e}")
                    passed_steps.append("from_documents_repeat_call")
                
                # 正确的做法是使用add_documents
                added_ids = builder.add_documents(docs_list[5:10])
                print(f"✅ 使用add_documents添加文档成功: {len(added_ids)} 个文档")
                passed_steps.append("from_documents_add_documents_after")
                
                builder.close()
                
            except Exception as e:
                print(f"重复调用from_documents测试失败: {e}")
                failed_steps.append("from_documents_repeat_test")
        
        print("✅ 步骤3: from_documents 方法测试完成")
        passed_steps.append("from_documents_overall")
    except Exception as e:
        print(f"❌ 步骤3失败: {e}")
        failed_steps.append("from_documents_overall")
    
    # Step 4: 测试添加和更新文档功能
    print("\n=== 步骤4: 测试添加和更新文档功能 ===")
    try:
        initial_docs = docs_list[:5]
        additional_docs = docs_list[5:8]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "add_update_index")
            
            # 创建初始索引
            config = BM25IndexBuilderConfig(
                index_path=index_path,
                batch_size=5
            )
            
            builder = BM25IndexBuilder(config=config)
            builder.from_documents(initial_docs)
            initial_stats = builder.get_index_stats()
            print(f"✅ 初始索引创建: {initial_stats}")
            passed_steps.append("add_update_initial_index")
            
            # 测试添加文档
            added_ids = builder.add_documents(additional_docs)
            print(f"✅ 添加文档ID: {added_ids}")
            passed_steps.append("add_update_add_documents")
            
            updated_stats = builder.get_index_stats()
            print(f"✅ 添加后统计: {updated_stats}")
            passed_steps.append("add_update_stats_after_add")
            
            # 验证覆盖前的文档存在
            print("--- 覆盖前验证 ---")
            pre_overwrite_doc = builder.get_document_by_id("doc_0")
            if pre_overwrite_doc:
                print(f"覆盖前文档存在: {pre_overwrite_doc.content}")
                passed_steps.append("add_update_pre_overwrite_exists")
            else:
                print("警告: 覆盖前文档不存在")
                failed_steps.append("add_update_pre_overwrite_exists")
            
            # 测试覆盖模式
            overwrite_doc = Document(
                id="doc_0",  # 与初始文档ID重复
                content="这是覆盖后的新内容",
                metadata={"source": "overwrite_test.pdf", "category": "overwrite"}
            )
            
            print("--- 执行覆盖操作 ---")
            overwrite_ids = builder.add_documents([overwrite_doc], overwrite=True)
            print(f"✅ 覆盖模式添加: {overwrite_ids}")
            passed_steps.append("add_update_overwrite")
            
            # 获取覆盖后的索引统计
            post_overwrite_stats = builder.get_index_stats()
            print(f"覆盖后索引统计: {post_overwrite_stats}")
            
            # 验证覆盖效果
            print("--- 验证覆盖效果 ---")
            retrieved_doc = builder.get_document_by_id("doc_0")
            if retrieved_doc:
                print(f"检索到的文档内容: {retrieved_doc.content}")
                print(f"检索到的文档元数据: {retrieved_doc.metadata}")
                if "覆盖后的新内容" in retrieved_doc.content:
                    print("✅ 文档覆盖成功")
                    passed_steps.append("add_update_overwrite_success")
                else:
                    print("❌ 文档覆盖失败 - 内容不匹配")
                    failed_steps.append("add_update_overwrite_success")
            else:
                print("❌ 文档覆盖失败 - 无法检索到文档")
                failed_steps.append("add_update_overwrite_success")
            
            # 测试更新文档
            update_doc = Document(
                id="doc_1",
                content="这是更新后的内容",
                metadata={"source": "updated.pdf", "category": "updated"}
            )
            
            update_ids = builder.update_documents([update_doc])
            print(f"✅ 更新文档ID: {update_ids}")
            passed_steps.append("add_update_update_documents")
            
            builder.close()
        
        print("✅ 步骤4: 添加和更新文档功能测试完成")
        passed_steps.append("add_update_overall")
    except Exception as e:
        print(f"❌ 步骤4失败: {e}")
        failed_steps.append("add_update_overall")
    
    # Step 5: 测试删除文档功能
    print("\n=== 步骤5: 测试删除文档功能 ===")
    try:
        test_docs = docs_list[:10]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "delete_index")
            
            config = BM25IndexBuilderConfig(index_path=index_path)
            builder = BM25IndexBuilder(config=config).from_documents(test_docs)
            
            initial_stats = builder.get_index_stats()
            print(f"✅ 删除前统计: {initial_stats}")
            passed_steps.append("delete_initial_stats")
            
            # 测试删除存在的文档
            deleted_count = builder.delete_documents(["doc_0", "doc_1", "doc_2"])
            print(f"✅ 删除文档数量: {deleted_count}")
            passed_steps.append("delete_existing_docs")
            
            # 测试删除不存在的文档
            deleted_count2 = builder.delete_documents(["non_existent_1", "non_existent_2"])
            print(f"✅ 删除不存在文档数量: {deleted_count2}")
            passed_steps.append("delete_nonexistent_docs")
            
            # 测试混合删除（存在和不存在）
            deleted_count3 = builder.delete_documents(["doc_3", "non_existent_3", "doc_4"])
            print(f"✅ 混合删除文档数量: {deleted_count3}")
            passed_steps.append("delete_mixed_docs")
            
            final_stats = builder.get_index_stats()
            print(f"✅ 删除后统计: {final_stats}")
            passed_steps.append("delete_final_stats")
            
            # 验证文档确实被删除
            deleted_doc = builder.get_document_by_id("doc_0")
            if deleted_doc is None:
                print("✅ 文档删除验证成功")
                passed_steps.append("delete_verification")
            else:
                print("❌ 文档删除验证失败")
                failed_steps.append("delete_verification")
            
            builder.close()
        
        print("✅ 步骤5: 删除文档功能测试完成")
        passed_steps.append("delete_overall")
    except Exception as e:
        print(f"❌ 步骤5失败: {e}")
        failed_steps.append("delete_overall")
    
    # Step 6: 测试文档检索功能
    print("\n=== 步骤6: 测试文档检索功能 ===")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "retrieval_index")
            
            config = BM25IndexBuilderConfig(index_path=index_path)
            builder = BM25IndexBuilder(config=config).from_documents(docs_list)
            
            # 测试单个文档检索
            doc = builder.get_document_by_id("doc_0")
            if doc:
                print(f"✅ 检索单个文档成功: {doc.id}")
                passed_steps.append("retrieval_single_doc")
            else:
                print("❌ 检索单个文档失败")
                failed_steps.append("retrieval_single_doc")
            
            # 测试不存在的文档
            non_doc = builder.get_document_by_id("non_existent")
            if non_doc is None:
                print("✅ 正确处理不存在的文档")
                passed_steps.append("retrieval_nonexistent_doc")
            else:
                print("❌ 不存在文档处理错误")
                failed_steps.append("retrieval_nonexistent_doc")
            
            # 测试创建检索器
            retriever = builder.as_retriever()
            print(f"✅ 创建检索器成功")
            passed_steps.append("retrieval_create_retriever")
            
            # 测试基本检索
            results = retriever.invoke("冷凝器", k=3)
            print(f"✅ 基本检索结果数量: {len(results)}")
            passed_steps.append("retrieval_basic_search")
            
            # 测试带过滤的检索
            filtered_results = retriever.invoke(
                "冷凝器", 
                k=3, 
                filters={"category": "technical"}
            )
            print(f"✅ 过滤检索结果数量: {len(filtered_results)}")
            passed_steps.append("retrieval_filtered_search")
            
            # 测试多字段过滤
            multi_filtered_results = retriever.invoke(
                "制冷",
                k=5,
                filters={"category": "technical", "region": "北京"}
            )
            print(f"✅ 多字段过滤检索结果数量: {len(multi_filtered_results)}")
            passed_steps.append("retrieval_multi_filter_search")
            
            # 显示检索结果详情
            if results:
                print("检索结果示例:")
                for i, doc in enumerate(results[:2]):
                    print(f"  {i+1}. ID: {doc.id}")
                    print(f"     Score: {doc.metadata.get('score', 'N/A')}")
                    print(f"     Content: {doc.content[:100]}...")
                    print(f"     Metadata: {doc.metadata}")
            
            builder.close()
        
        print("✅ 步骤6: 文档检索功能测试完成")
        passed_steps.append("retrieval_overall")
    except Exception as e:
        print(f"❌ 步骤6失败: {e}")
        failed_steps.append("retrieval_overall")
    
    # Step 7: 测试索引持久化功能
    print("\n=== 步骤7: 测试索引持久化功能 ===")
    try:
        test_docs = docs_list[:5]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "persistence_index")
            
            # 创建并构建索引
            config1 = BM25IndexBuilderConfig(index_path=index_path)
            builder1 = BM25IndexBuilder(config=config1)
            builder1.from_documents(test_docs)
            
            stats1 = builder1.get_index_stats()
            print(f"✅ 初始索引统计: {stats1}")
            passed_steps.append("persistence_initial_stats")
            
            builder1.close()
            
            # 重新加载索引
            config2 = BM25IndexBuilderConfig(index_path=index_path)
            builder2 = BM25IndexBuilder(config=config2)
            builder2.load_local()  # 需要调用load_local()加载现有索引
            
            stats2 = builder2.get_index_stats()
            print(f"✅ 重新加载索引统计: {stats2}")
            passed_steps.append("persistence_reload_stats")
            
            # 验证数据一致性
            doc = builder2.get_document_by_id("doc_0")
            if doc:
                print("✅ 索引持久化验证成功")
                passed_steps.append("persistence_data_consistency")
            else:
                print("❌ 索引持久化验证失败")
                failed_steps.append("persistence_data_consistency")
            
            # 在已有索引基础上添加文档
            new_docs = [Document(
                id="new_doc_1",
                content="这是新添加的文档内容",
                metadata={"source": "new.pdf", "category": "new"}
            )]
            
            builder2.add_documents(new_docs)
            final_stats = builder2.get_index_stats()
            print(f"✅ 添加新文档后统计: {final_stats}")
            passed_steps.append("persistence_add_after_reload")
            
            builder2.close()
        
        print("✅ 步骤7: 索引持久化功能测试完成")
        passed_steps.append("persistence_overall")
    except Exception as e:
        print(f"❌ 步骤7失败: {e}")
        failed_steps.append("persistence_overall")
    
    # Step 8: 测试动态字段功能
    print("\n=== 步骤8: 测试动态字段功能 ===")
    try:
        # 创建包含不同元数据字段的文档
        docs_with_various_fields = [
            Document(
                id="field_test_1",
                content="测试文档1",
                metadata={"department": "研发部", "priority": "高", "year": "2024"}
            ),
            Document(
                id="field_test_2", 
                content="测试文档2",
                metadata={"department": "销售部", "status": "完成", "region": "华东"}
            ),
            Document(
                id="field_test_3",
                content="测试文档3", 
                metadata={"author": "张三", "priority": "中", "type": "报告"}
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "dynamic_fields_index")
            
            config = BM25IndexBuilderConfig(index_path=index_path)
            builder = BM25IndexBuilder(config=config)
            builder.from_documents(docs_with_various_fields)
            
            retriever = builder.as_retriever()
            
            # 测试不同字段的过滤
            dept_results = retriever.invoke("测试", filters={"department": "研发部"})
            print(f"✅ 按department过滤结果: {len(dept_results)}")
            passed_steps.append("dynamic_fields_dept_filter")
            
            priority_results = retriever.invoke("测试", filters={"priority": "高"})
            print(f"✅ 按priority过滤结果: {len(priority_results)}")
            passed_steps.append("dynamic_fields_priority_filter")
            
            author_results = retriever.invoke("测试", filters={"author": "张三"})
            print(f"✅ 按author过滤结果: {len(author_results)}")
            passed_steps.append("dynamic_fields_author_filter")
            
            # 测试多个值的过滤
            multi_value_results = retriever.invoke(
                "测试", 
                filters={"department": ["研发部", "销售部"]}
            )
            print(f"✅ 多值过滤结果: {len(multi_value_results)}")
            passed_steps.append("dynamic_fields_multi_value_filter")
            
            # 显示过滤结果
            if dept_results:
                print(f"部门过滤结果示例: {dept_results[0].metadata}")
            
            builder.close()
        
        print("✅ 步骤8: 动态字段功能测试完成")
        passed_steps.append("dynamic_fields_overall")
    except Exception as e:
        print(f"❌ 步骤8失败: {e}")
        failed_steps.append("dynamic_fields_overall")
    
    # Step 9: 测试进度回调功能
    print("\n=== 步骤9: 测试进度回调功能 ===")
    try:
        # 进度回调记录
        progress_records = []
        
        def progress_callback(processed, total, stats):
            progress_records.append({
                'processed': processed,
                'total': total,
                'stats': stats
            })
            print(f"进度: {processed}/{total} ({processed/total*100:.1f}%) - {stats['throughput_docs_sec']:.1f} docs/sec")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "progress_index")
            
            config = BM25IndexBuilderConfig(
                index_path=index_path,
                batch_size=500,
                progress_interval=1000,  # 每3个文档报告一次进度
                progress_callback=progress_callback
            )
            
            builder = BM25IndexBuilder(config=config).from_documents(docs_list)
            
            print(f"✅ 进度回调被调用 {len(progress_records)} 次")
            passed_steps.append("progress_callback_called")
            
            # 验证最后一次回调包含所有文档
            if progress_records:
                last_record = progress_records[-1]
                if last_record['processed'] == last_record['total']:
                    print("✅ 最终进度回调正确")
                    passed_steps.append("progress_callback_final_correct")
                else:
                    print(f"❌ 最终进度回调不正确: {last_record['processed']}/{last_record['total']}")
                    failed_steps.append("progress_callback_final_correct")
            
            builder.close()
        
        print("✅ 步骤9: 进度回调功能测试完成")
        passed_steps.append("progress_callback_overall")
    except Exception as e:
        print(f"❌ 步骤9失败: {e}")
        failed_steps.append("progress_callback_overall")
    
    # Step 10: 测试错误处理功能
    print("\n=== 步骤10: 测试错误处理功能 ===")
    try:
        # 测试无效路径
        try:
            config = BM25IndexBuilderConfig(index_path="/invalid/path/that/does/not/exist")
            builder = BM25IndexBuilder.from_documents([
                Document(id="test", content="test content", metadata={})
            ], config)
            print("❌ 应该处理无效路径错误")
            failed_steps.append("error_handling_invalid_path")
        except Exception as e:
            print(f"✅ 正确处理无效路径: {type(e).__name__}")
            passed_steps.append("error_handling_invalid_path")
        
        # 测试无效文档
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "error_index")
            
            try:
                config = BM25IndexBuilderConfig(index_path=index_path)
                
                # 测试包含None内容的文档
                invalid_docs = [
                    Document(id="valid", content="valid content", metadata={}),
                    Document(id="invalid", content=None, metadata={}),  # None content
                    Document(id="empty", content="", metadata={})  # Empty content
                ]
                
                builder = BM25IndexBuilder(config=config)
                builder.from_documents(invalid_docs)
                print("✅ 正确处理无效文档内容")
                passed_steps.append("error_handling_invalid_docs")
                
                builder.close()
                
            except Exception as e:
                print(f"处理无效文档时出错: {e}")
                failed_steps.append("error_handling_invalid_docs")
        
        print("✅ 步骤10: 错误处理功能测试完成")
        passed_steps.append("error_handling_overall")
    except Exception as e:
        print(f"❌ 步骤10失败: {e}")
        failed_steps.append("error_handling_overall")
    
    
    if failed_steps:
        print("\n失败的测试步骤:")
        for step in failed_steps:
            print(f"  - {step}")
    
    print("=" * 60)
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    success = test_bm25_index_builder()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 存在失败的测试，请检查以上输出")
        sys.exit(1)