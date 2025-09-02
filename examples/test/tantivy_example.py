import os
import jieba
import json
import tantivy

# ========================
# 1. 定义中文分词函数
# ========================
def chinese_preprocess(text: str) -> str:
    """用 jieba 对中文进行分词，返回空格分隔的字符串"""
    return " ".join(jieba.cut(text))


# ========================
# 2. 定义 Schema
# ========================
schema_builder = tantivy.SchemaBuilder()
schema_builder.add_text_field("title", stored=True, tokenizer="default")
schema_builder.add_text_field("body", stored=True, tokenizer="default")  # 仍然用 default，但写入时我们手动切词
schema = schema_builder.build()

# 索引路径
index_path = "index_cn"

# ========================
# 3. 创建或加载索引
# ========================
if os.path.exists(index_path):
    print("加载已有索引...")
    index = tantivy.Index.open(index_path)
else:
    print("创建新索引...")
    os.makedirs(index_path, exist_ok=True)
    index = tantivy.Index(schema, path=index_path)

    # 构造写入器
    writer = index.writer(heap_size=15_000_000)

    # 示例文档，可以替换成你的 JSON 数据
    docs = [
        {"title": "制冷系统", "body": "蒸发器是制冷系统中的重要部件，它的作用是吸收热量。"},
        {"title": "空调原理", "body": "空调通过压缩机、冷凝器、蒸发器等部件实现制冷和制热。"},
        {"title": "换热器", "body": "换热器用于不同介质之间的热量交换。"}
    ]

    # 写入文档（手动做中文分词）
    for item in docs:
        body_cut = chinese_preprocess(item["body"])
        writer.add_document(tantivy.Document(
            title=[item["title"]],
            body=[body_cut],
        ))

    writer.commit()
    print("索引创建完成")

# ========================
# 4. BM25 搜索
# ========================
index.reload()
searcher = index.searcher()

# 用户输入查询
user_query = "蒸发器是什么"
# 对查询做中文分词
query_cut = chinese_preprocess(user_query)

# 构建查询
query = index.parse_query(query_cut, ["body"])
top_docs = searcher.search(query, 10).hits

print("\n搜索结果：")
for score, doc_address in top_docs:
    doc = searcher.doc(doc_address)
    print(f"评分: {score:.4f}")
    print(f"标题: {doc['title'][0]}")
    print(f"内容: {doc['body'][0]}")
    print("-" * 50)
