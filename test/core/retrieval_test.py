import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.bm25 import jieba_preprocessing_func
from core.retrieval.dense import VectorStoreRetriever
from core.retrieval.multipath import MultiPathRetriever
from encapsulation.database.bm25_indexer import BM25IndexBuilder
from encapsulation.database.vector_db.VectorStoreBase import VectorStore
from core.file_management.embeddings.base import Embeddings

# Mock embedding class for testing
class MockEmbeddings(Embeddings):
    """Mock embeddings for testing"""
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embedding documents"""
        # Return random embeddings for testing
        import random
        return [[random.random() for _ in range(128)] for _ in texts]
    
    def embed_query(self, text: str) -> list[float]:
        """Mock embedding query"""
        # Return random embedding for testing
        import random
        return [random.random() for _ in range(128)]

# Mock vector store class for testing
class MockVectorStore(VectorStore):
    """Mock vector store for testing"""
    
    def __init__(self, embedding=None, **kwargs):
        super().__init__()
        self.embedding = embedding or MockEmbeddings()
        self.docstore = {}
    
    def add_documents(self, documents, **kwargs):
        """Mock add documents"""
        ids = []
        for doc in documents:
            if not doc.id:
                import uuid
                doc.id = str(uuid.uuid4())
            self.docstore[doc.id] = doc
            ids.append(doc.id)
        return ids
    
    def delete(self, ids=None, **kwargs):
        """Mock delete documents"""
        if ids:
            for id in ids:
                self.docstore.pop(id, None)
        else:
            self.docstore.clear()
        return True
    
    def similarity_search(self, query, k=4, **kwargs):
        """Mock similarity search - return first k documents"""
        docs = list(self.docstore.values())
        return docs[:k]
    
    def similarity_search_with_score(self, query, k=4, **kwargs):
        """Mock similarity search with scores"""
        docs = list(self.docstore.values())
        # Return documents with mock scores
        import random
        return [(doc, random.random()) for doc in docs[:k]]
    
    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5, **kwargs):
        """Mock max marginal relevance search - return first k documents"""
        docs = list(self.docstore.values())
        return docs[:k]
    
    def _select_relevance_score_fn(self):
        """Select relevance score function - return a simple cosine similarity function"""
        import random
        def cosine_similarity_fn(distance):
            # Return a random score between 0 and 1 for testing
            return random.random()
        return cosine_similarity_fn
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, *, ids=None, **kwargs):
        """Create MockVectorStore from texts"""
        instance = cls(embedding=embedding, **kwargs)
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc_id = ids[i] if ids and i < len(ids) else None
            doc = Document(content=text, metadata=metadata, id=doc_id)
            docs.append(doc)
        instance.add_documents(docs)
        return instance

# Load test data
def load_test_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, "..", "tcl_gb_chunk.json")
    
    print("Loading documents...")
    with open(json_file_path, "r") as f:
        docs = json.load(f)
        docs_list = []
        for doc in docs:
            source = doc["metadata"]["file_name"]
            metadata = {
                "source": source
            }
            docs_list.append(Document(content=doc["content"], metadata=metadata))

    print(f"Loaded {len(docs_list)} documents")

    return docs_list



# Test core/retrieval/bm25.py
def test_bm25_retriever():
    docs_list = load_test_data()
    
    # Create BM25Retriever instance
    docs_content_list = [doc.content for doc in docs_list]
    docs_metadata_list = [doc.metadata for doc in docs_list]
    docs_ids_list = [doc.id for doc in docs_list]
    retriever = BM25Retriever.from_texts(
        texts=docs_content_list, 
        metadatas=docs_metadata_list,
        ids=docs_ids_list,
        preprocess_func=jieba_preprocessing_func, 
        k=3
    )
    
    # Test retrieval functionality
    query = "冷凝管"
    results = retriever._get_relevant_documents(query)
    print("Results for query '冷凝管':")
    for doc in results:
        print(doc.content)
    print('\n')
    
    # Test score retrieval
    scores = retriever.get_scores("冷凝管")
    print("Scores for query '冷凝管':", scores)

    # Test top-k with scores
    top_k_with_scores = retriever.get_top_k_with_scores("冷凝管", k=2)
    print("Top-2 docs with scores for '冷凝管':")
    for doc, score in top_k_with_scores:
        print(doc.content, score)
    
    # Test document addition
    new_doc = Document(content="A new cat arrived.", metadata={}, id="new1")
    retriever.add_documents([new_doc])
    print("Document count after adding:", retriever.get_document_count())
    
    # Test document deletion
    retriever.delete_documents(ids=["new1"])
    print("Document count after deleting:", retriever.get_document_count())
    
    # Test information retrieval
    info = retriever.get_bm25_info()
    print("BM25 info:", info)
    print("\n" + "="*50 + "\n")


# Test core/retrieval/dense.py
def test_dense_retriever():
    docs_list = load_test_data()
    
    # Create mock embeddings
    embeddings = MockEmbeddings()
    
    # Create MockVectorStore
    vectorstore = MockVectorStore(embedding=embeddings)
    
    # Add documents to vector store
    vectorstore.add_documents(docs_list)
    
    # Create VectorStoreRetriever instance
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Test retrieval functionality
    query = "冷凝管"
    results = retriever._get_relevant_documents(query)
    print("Results for query '冷凝管' (dense):")
    for doc in results:
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
    print('\n')
    
    # Test similarity search with score threshold
    retriever_threshold = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5}
    )
    
    results_threshold = retriever_threshold._get_relevant_documents(query)
    print("Results for query '冷凝管' with score threshold (dense):")
    for doc in results_threshold:
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
    print('\n')
    
    # Test MMR search
    retriever_mmr = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    results_mmr = retriever_mmr._get_relevant_documents(query)
    print("Results for query '冷凝管' with MMR (dense):")
    for doc in results_mmr:
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
    print('\n')
    
    # Test document addition
    new_doc = Document(content="A new cat arrived.", metadata={}, id="new1")
    retriever.add_documents([new_doc])
    print("Document count after adding (dense):", len(vectorstore.docstore))
    
    # Test document deletion
    retriever.delete_documents(ids=["new1"])
    print("Document count after deleting (dense):", len(vectorstore.docstore))
    
    # Test information retrieval
    info = retriever.get_vectorstore_info()
    print("Dense retriever info:", info)
    print("\n" + "="*50 + "\n")


# Test core/retrieval/mutipath.py
def test_multipath_retriever():
    docs_list = load_test_data()
    
    # Create BM25Retriever instance
    docs_content_list = [doc.content for doc in docs_list]
    docs_metadata_list = [doc.metadata for doc in docs_list]
    docs_ids_list = [doc.id for doc in docs_list]
    bm25_retriever = BM25Retriever.from_texts(
        texts=docs_content_list, 
        metadatas=docs_metadata_list,
        ids=docs_ids_list,
        preprocess_func=jieba_preprocessing_func, 
        k=3
    )
    
    # Create MockVectorStore and VectorStoreRetriever
    embeddings = MockEmbeddings()
    vectorstore = MockVectorStore(embedding=embeddings)
    vectorstore.add_documents(docs_list)
    vector_retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create MultiPathRetriever
    multi_retriever = MultiPathRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        top_k_per_retriever=10
    )
    
    # Test retrieval functionality
    query = "冷凝管"
    results = multi_retriever.invoke(query, k=5)
    print("Results for query '冷凝管' (multipath):")
    for doc in results:
        print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
    print('\n')
    
    # Test information retrieval
    info = multi_retriever.get_multipath_info()
    print("Multipath retriever info:", info)
    print("\n" + "="*50 + "\n")


# Test encapsulation/database/bm25_indexer.py
def test_tantivy_bm25_retriever():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("script_dir: ", script_dir)
    
    docs_list = load_test_data()

    index_path = os.path.join(script_dir, "test_bm25_index")
    with BM25IndexBuilder(index_path=index_path, max_workers=2) as builder:
        builder.build_index(docs_list)
        builder.load_local(index_path)
        
        retriever = builder.as_retriever(top_k=3)
        
        print(f"Created retriever: {retriever}\n")
        
        # --- Normal search ---
        print("--- Normal Search for '冷凝器片距的选择依据是什么？' ---")
        results = retriever.invoke("冷凝器片距的选择依据是什么？")
        for doc in results:
            print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Content: {doc.content[:100]}...")  # Truncate display

        print("\n" + "="*50 + "\n")

        # --- Search with filters ---
        print("--- Filtered Search with source filter ---")
        filtered_results = retriever.invoke(
            "冷凝器片距的选择依据是什么？", 
            filters={"source": ["Q$TD-03.085-2024 蒸发器设计规范（AC版）"]}  # 👈 Use list wrapper
        )
        for doc in filtered_results:
            print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.content[:100]}...")

        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    print('step 1: test bm25 retriever')
    test_bm25_retriever()
    print('step 2: test dense retriever')
    test_dense_retriever()
    print('step 3: test multipath retriever')
    test_multipath_retriever()
    print('step 4: test tantivy bm25 retriever')
    test_tantivy_bm25_retriever()
    print("All tests passed.")