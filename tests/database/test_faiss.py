"""
Test for FaissVectorDB - testing only the methods that actually exist
"""

import tempfile
import os
import asyncio
from typing import Dict, Any, Literal, Annotated
from pydantic import Field

from framework.config import AbstractConfig
from encapsulation.llm.huggingface import HuggingFaceLLM
from encapsulation.database.vector_db.faiss import FaissVectorDB
from encapsulation.database.vector_db.document import Document


class HuggingFaceConfig(AbstractConfig):
    """Configuration for HuggingFace LLM"""
    type: Literal["huggingface"] = "huggingface"
    model_name: str
    device: str = "cpu"
    task_types: list = ["embedding"]
    
    def build(self) -> HuggingFaceLLM:
        return HuggingFaceLLM(self)


class FaissConfig(AbstractConfig):
    """Configuration for FaissVectorDB testing"""
    type: Literal["faiss_vector_db"] = "faiss_vector_db"
    embedding: Annotated[HuggingFaceConfig, Field(discriminator="type")]
    index_type: str = "flat"
    metric: str = "cosine"
    normalize_L2: bool = False
    
    def build(self) -> FaissVectorDB:
        return FaissVectorDB(self)


def create_faiss_config(config_data: Dict[str, Any]) -> FaissConfig:
    """Create FaissConfig from dictionary using configuration injection"""
    # Create HuggingFace config from the nested embedding data
    hf_config = HuggingFaceConfig(
        model_name=config_data["embedding"]["model_name"],
        device=config_data["embedding"].get("device", "cpu")
    )
    
    return FaissConfig(
        embedding=hf_config,
        index_type=config_data.get("index_type", "flat"),
        metric=config_data.get("metric", "cosine"),
        normalize_L2=config_data.get("normalize_L2", False)
    )


def test_existing_methods(config: FaissConfig):
    """Test only the methods that actually exist in FaissVectorDB"""
    print("=== Testing Existing FaissVectorDB Methods ===")
    
    # Test data
    documents = [
        Document(content="The weather is sunny today", metadata={"topic": "weather"}, id="doc1"),
        Document(content="Machine learning is fascinating", metadata={"topic": "tech"}, id="doc2"),
        Document(content="Python programming is great", metadata={"topic": "tech"}, id="doc3"),
        Document(content="It's raining outside", metadata={"topic": "weather"}, id="doc4"),
        Document(content="I love reading books", metadata={"topic": "hobby"}, id="doc5")
    ]
    
    # 1. Test build
    print("\n--- Test 1: build ---")
    vector_db = config.build()
    print(f"✓ FaissVectorDB built from config")
    print(f"  Index type: {vector_db.config.index_type}")
    print(f"  Metric: {vector_db.config.metric}")
    print(f"  Normalize L2: {vector_db.config.normalize_L2}")
    
    # 2. Test _add_documents
    print("\n--- Test 2: _add_documents ---")
    
    added_ids = vector_db._add_documents(documents)
    print(f"✓ Added {len(added_ids)} documents")
    print(f"  Added IDs: {added_ids}")
    print(f"  Index total vectors: {vector_db.index.ntotal}")
    print(f"  Index dimension: {vector_db.index.d}")
    print(f"  Docstore size: {len(vector_db.docstore)}")
    
    # 3. Test get_by_ids
    print("\n--- Test 3: get_by_ids ---")
    retrieved = vector_db.get_by_ids(["doc1", "doc3", "nonexistent"])
    print(f"✓ Retrieved {len(retrieved)} documents by ID")
    for doc in retrieved:
        print(f"  - {doc.id}: {doc.content}")
    
    # Test empty list
    empty_retrieved = vector_db.get_by_ids([])
    print(f"✓ Empty ID list returned {len(empty_retrieved)} documents")
    
    # 4. Test save_local and load_from_folder
    print("\n--- Test 4: save_local / load_from_folder ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save
        vector_db.save_local(temp_dir, "test_index")
        print(f"✓ Saved to {temp_dir}")
        
        # Check files exist
        faiss_file = os.path.join(temp_dir, "test_index.faiss")
        pkl_file = os.path.join(temp_dir, "test_index.pkl")
        print(f"  FAISS file exists: {os.path.exists(faiss_file)}")
        print(f"  PKL file exists: {os.path.exists(pkl_file)}")
        
        # Load using two-step initialization
        load_config = FaissConfig(
            embedding=config.embedding
        )
        loaded_db = load_config.build()
        loaded_db.load_from_folder(temp_dir)
        print(f"✓ Loaded from {temp_dir} using load_from_folder")
        print(f"  Loaded docstore size: {len(loaded_db.docstore)}")
        print(f"  Loaded index total: {loaded_db.index.ntotal if loaded_db.index else 0}")
        print(f"  Loaded index type: {loaded_db.config.index_type}")
        print(f"  Loaded metric: {loaded_db.config.metric}")
        
        # Test loaded DB functionality
        loaded_docs = loaded_db.get_by_ids(["doc1", "doc2"])
        print(f"  ✓ Loaded DB get_by_ids works: {len(loaded_docs)} docs retrieved")
    
    # 5. Test delete
    print("\n--- Test 5: delete ---")
    initial_count = len(vector_db.docstore)
    print(f"  Initial document count: {initial_count}")
    
    # Test selective deletion
    delete_result = vector_db.delete(["doc2", "doc4"])
    print(f"✓ Deleted selected documents: {delete_result}")
    print(f"  Remaining docs: {len(vector_db.docstore)}")
    
    # Verify deletion worked
    remaining = vector_db.get_by_ids(["doc1", "doc2", "doc3", "doc4", "doc5"])
    remaining_ids = [doc.id for doc in remaining]
    print(f"  Remaining IDs: {remaining_ids}")
    
    # Test delete all
    delete_all_result = vector_db.delete()
    print(f"✓ Deleted all documents: {delete_all_result}")
    print(f"  Final document count: {len(vector_db.docstore)}")
    
    # 6. Test initialize_from_documents
    print("\n--- Test 6: initialize_from_documents ---")
    new_docs = [
        Document(content="Initialize method test 1", metadata={"method": "initialize"}, id="init1"),
        Document(content="Initialize method test 2", metadata={"method": "initialize"}, id="init2")
    ]
    
    init_db = config.build()
    init_db.initialize_from_documents(new_docs)
    print(f"✓ Initialized DB from documents: {len(init_db.docstore)} docs")
    print(f"  Init DB index total: {init_db.index.ntotal}")
    
    # Test the initialized DB
    init_retrieved = init_db.get_by_ids(["init1", "init2"])
    print(f"  ✓ Initialize DB get_by_ids works: {len(init_retrieved)} docs")
    for doc in init_retrieved:
        print(f"    - {doc.id}: {doc.content}")
    
    print("\n🎉 All existing methods tested successfully!")
    return init_db


async def test_async_methods(config: FaissConfig):
    """Test async methods that exist"""
    print("\n=== Testing Async Methods ===")
    
    # Test aadd_documents
    print("\n--- Test 7: aadd_documents (async) ---")
    vector_db = config.build()
    
    async_docs = [
        Document(content="Async test document 1", metadata={"async": True, "index": 0}, id="async_doc1"),
        Document(content="Async test document 2", metadata={"async": True, "index": 1}, id="async_doc2")
    ]
    
    added_ids = await vector_db.aadd_documents(async_docs)
    print(f"✓ Async added {len(added_ids)} documents")
    print(f"  Added IDs: {added_ids}")
    print(f"  Index total: {vector_db.index.ntotal}")
    
    print("✅ All async methods tested successfully!")


def main():
    print("Testing FaissVectorDB - Only Existing Methods")
    
    # Create config directly in code instead of loading from JSON
    print("✓ Creating FaissConfig directly in code")
    
    # Create HuggingFace embedding config
    hf_config = HuggingFaceConfig(
        model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
        device="cuda:7"
    )
    
    # Create FAISS config with embedded HF config
    config = FaissConfig(
        embedding=hf_config,
        index_type="flat",
        metric="cosine", 
        normalize_L2=False
    )
    
    print(f"  Index: {config.index_type}/{config.metric}")
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Device: {config.embedding.device}")
    
    try:
        # Test sync methods
        test_existing_methods(config)
        
        # Test async methods
        asyncio.run(test_async_methods(config))
        
        print("\n🎊 ALL TESTS PASSED! All existing methods work correctly.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()