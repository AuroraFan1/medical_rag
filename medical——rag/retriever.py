"""
高级检索策略
针对医疗文本优化
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from rank_bm25 import BM25Okapi
import jieba
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import RETRIEVAL_CONFIG, DEVICE

logger = logging.getLogger(__name__)

class MedicalRetriever:
    """医疗文本检索器"""

    def __init__(self,
                 vector_store,
                 enable_hybrid: bool = True,
                 enable_rerank: bool = True,
                 enable_diversity: bool = True):

        self.vector_store = vector_store
        self.enable_hybrid = enable_hybrid
        self.enable_rerank = enable_rerank
        self.enable_diversity = enable_diversity

        # BM25索引
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_doc_ids = []

        # 重排序模型
        self.reranker = None
        self.rerank_tokenizer = None

        # 初始化组件
        if enable_rerank:
            self._init_reranker()

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            model_name = RETRIEVAL_CONFIG.get("reranker_model", "BAAI/bge-reranker-large")

            logger.info(f"加载重排序模型: {model_name}")

            self.rerank_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(DEVICE)

            self.reranker.eval()

            logger.info("重排序模型加载成功")

        except Exception as e:
            logger.warning(f"加载重排序模型失败: {e}")
            logger.warning("将禁用重排序功能")
            self.enable_rerank = False

    # def build_bm25_index(self, documents: List[Dict]):
    #     """构建BM25索引"""
    #     if not documents:
    #         return
    #
    #     logger.info("构建BM25索引...")
    #
    #     # 提取文档内容
    #     self.bm25_docs = []
    #     self.bm25_doc_ids = []
    #
    #     for doc in documents:
    #         content = doc.get("content", "")
    #         if content:
    #             # 中文分词
    #             tokens = list(jieba.cut_for_search(content))
    #             self.bm25_docs.append(tokens)
    #             self.bm25_doc_ids.append(doc.get("id", len(self.bm25_docs)))
    #
    #     # 创建BM25索引
    #     if self.bm25_docs:
    #         self.bm25_index = BM25Okapi(self.bm25_docs)
    #         logger.info(f"BM25索引构建完成: {len(self.bm25_docs)} 个文档")
    #     else:
    #         logger.warning("没有有效文档构建BM25索引")

    def semantic_search(self,
                       query: str,
                       top_k: int = 10,
                       filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """语义搜索"""
        return self.vector_store.search(
            query,
            top_k=top_k * 2,  # 多取一些用于后续处理
            filter_conditions=filter_conditions
        )

    def keyword_search(self,
                      query: str,
                      top_k: int = 10) -> List[Dict]:
        """关键词搜索（BM25）"""
        if not self.bm25_index:
            return []

        # 分词查询
        query_tokens = list(jieba.cut_for_search(query))

        # 获取分数
        scores = self.bm25_index.get_scores(query_tokens)

        # 获取top_k索引
        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            if idx < len(self.bm25_docs) and scores[idx] > 0:
                # 注意：这里需要从原始文档获取完整信息
                # 简化处理，返回基本结构
                results.append({
                    "content": " ".join(self.bm25_docs[idx][:50]),  # 取前50个词
                    "similarity": float(scores[idx] / (scores[idx] + 1)),  # 转换为0-1范围
                    "method": "bm25"
                })

        return results[:top_k]

    def rerank_results(self,
                      query: str,
                      documents: List[Dict],
                      top_k: int = 10) -> List[Dict]:
        """重排序结果"""
        if not self.enable_rerank or not documents or len(documents) <= 1:
            return documents[:top_k]

        try:
            # 准备输入对
            pairs = []
            for doc in documents:
                content = doc.get("content", "")
                if len(content) > 500:  # 限制长度
                    content = content[:500]
                pairs.append((query, content))

            # 批量处理
            batch_size = 16
            rerank_scores = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]

                # 编码
                inputs = self.rerank_tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(DEVICE)

                # 推理
                with torch.no_grad():
                    outputs = self.reranker(**inputs)
                    batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                    rerank_scores.extend(batch_scores.tolist())

            # 更新文档分数
            for i, doc in enumerate(documents):
                if i < len(rerank_scores):
                    original_score = doc.get("similarity", 0.5)
                    rerank_score = rerank_scores[i]

                    # 组合分数
                    combined_score = 0.7 * rerank_score + 0.3 * original_score
                    doc["rerank_score"] = rerank_score
                    doc["combined_score"] = combined_score
                else:
                    doc["combined_score"] = doc.get("similarity", 0.5)

            # 按组合分数排序
            documents.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

            return documents[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]

    def diversity_filter(self,
                        documents: List[Dict],
                        threshold: float = 0.8) -> List[Dict]:
        """多样性过滤"""
        if not self.enable_diversity or len(documents) <= 1:
            return documents

        filtered_docs = []

        for doc in documents:
            is_similar = False

            # 检查与已选文档的相似度
            for selected_doc in filtered_docs:
                # 简单的文本相似度检查
                content1 = selected_doc.get("content", "")[:200]
                content2 = doc.get("content", "")[:200]

                if not content1 or not content2:
                    continue

                # 计算Jaccard相似度
                words1 = set(jieba.lcut(content1))
                words2 = set(jieba.lcut(content2))

                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))

                    if union > 0:
                        similarity = intersection / union
                        if similarity > threshold:
                            is_similar = True
                            break

            if not is_similar:
                filtered_docs.append(doc)

        return filtered_docs

    def hybrid_search(self,
                     query: str,
                     top_k: int = 10,
                     filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """混合搜索"""

        # 并行执行两种搜索
        semantic_results = self.semantic_search(query, top_k * 2, filter_conditions)
        keyword_results = self.keyword_search(query, top_k * 2) if self.bm25_index else []

        # 合并结果
        all_results = {}

        # 添加语义结果
        for result in semantic_results:
            content_key = result.get("content", "")[:100]
            if content_key and content_key not in all_results:
                all_results[content_key] = {
                    **result,
                    "semantic_score": result.get("similarity", 0),
                    "keyword_score": 0
                }

        # 添加关键词结果
        for result in keyword_results:
            content_key = result.get("content", "")[:100]
            if content_key:
                if content_key in all_results:
                    # 合并分数
                    all_results[content_key]["keyword_score"] = result.get("similarity", 0)
                else:
                    all_results[content_key] = {
                        **result,
                        "semantic_score": 0,
                        "keyword_score": result.get("similarity", 0)
                    }

        # 计算混合分数
        combined_results = []
        for content_key, data in all_results.items():
            semantic_weight = RETRIEVAL_CONFIG["hybrid_search"]["semantic_weight"]
            keyword_weight = RETRIEVAL_CONFIG["hybrid_search"]["bm25_weight"]

            hybrid_score = (
                data["semantic_score"] * semantic_weight +
                data["keyword_score"] * keyword_weight
            )

            result = {
                "content": data.get("content", ""),
                "metadata": data.get("metadata", {}),
                "similarity": hybrid_score,
                "semantic_score": data["semantic_score"],
                "keyword_score": data["keyword_score"],
                "method": "hybrid"
            }

            if "id" in data:
                result["id"] = data["id"]

            combined_results.append(result)

        # 按混合分数排序
        combined_results.sort(key=lambda x: x["similarity"], reverse=True)

        return combined_results

    def retrieve(self,
                query: str,
                top_k: int = 10,
                filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """
        主要检索方法

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_conditions: 过滤条件

        Returns:
            检索结果列表
        """

        # 选择检索策略
        if self.enable_hybrid and self.bm25_index:
            results = self.hybrid_search(query, top_k * 2, filter_conditions)
        else:
            results = self.semantic_search(query, top_k * 2, filter_conditions)

        # 重排序
        if self.enable_rerank and results:
            results = self.rerank_results(query, results, top_k * 2)

        # 多样性过滤
        if self.enable_diversity and results:
            diversity_threshold = RETRIEVAL_CONFIG.get("diversity_threshold", 0.8)
            results = self.diversity_filter(results, diversity_threshold)

        # 返回top_k
        return results[:top_k]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计"""
        stats = {
            "enable_hybrid": self.enable_hybrid,
            "enable_rerank": self.enable_rerank,
            "enable_diversity": self.enable_diversity,
            "bm25_index_size": len(self.bm25_docs) if self.bm25_index else 0,
            "has_reranker": self.reranker is not None
        }

        return stats

def test_retrieval():
    """测试检索功能"""
    # 创建模拟向量数据库
    class MockVectorStore:
        def search(self, query, top_k=10, filter_conditions=None):
            return [
                {
                    "content": f"文档{i}：关于{query}的信息",
                    "metadata": {"id": i, "source": "test"},
                    "similarity": 0.9 - i * 0.1
                }
                for i in range(top_k)
            ]

    # 创建检索器
    vector_store = MockVectorStore()
    retriever = MedicalRetriever(
        vector_store,
        enable_hybrid=True,
        enable_rerank=False,
        enable_diversity=True
    )

    # 构建模拟BM25索引
    mock_docs = [
        {"id": i, "content": f"医疗文档{i}：关于糖尿病的信息"}
        for i in range(100)
    ]
    retriever.build_bm25_index(mock_docs)

    # 测试检索
    query = "糖尿病饮食"
    results = retriever.retrieve(query, top_k=5)

    print(f"查询: {query}")
    print(f"返回结果数: {len(results)}")

    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  内容: {result.get('content', '')[:50]}...")
        print(f"  相似度: {result.get('similarity', 0):.3f}")
        print(f"  方法: {result.get('method', 'unknown')}")

    # 获取统计信息
    stats = retriever.get_retrieval_stats()
    print(f"\n检索统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_retrieval()