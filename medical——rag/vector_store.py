"""
向量数据库管理器
针对医疗文本优化
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
import pickle

from config import VECTOR_STORE_DIR, EMBEDDING_MODELS, DEFAULT_EMBEDDER
from embedder import EmbeddingManager

logger = logging.getLogger(__name__)

class MedicalVectorStore:
    """医疗向量数据库"""

    def __init__(self,
                 store_name: str = "medical_cases",
                 embedding_model: str = DEFAULT_EMBEDDER,
                 persist: bool = True):
        self.store_name = store_name
        self.embedding_model = embedding_model
        self.persist = persist

        # 嵌入管理器
        self.embedder = EmbeddingManager(embedding_model)

        # 存储路径
        if persist:
            self.store_path = VECTOR_STORE_DIR / store_name
            self.store_path.mkdir(parents=True, exist_ok=True)
        else:
            self.store_path = None

        # ChromaDB客户端
        self.client = None
        self.collection = None

        # 统计数据
        self.stats = {
            "total_documents": 0,
            "last_update": None,
            "embedding_model": embedding_model
        }

        self._init_chromadb()

    def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            print(self.store_path)
            if self.persist and self.store_path:
                # 持久化模式
                self.client = chromadb.PersistentClient(
                    path=str(self.store_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # 内存模式
                self.client = chromadb.EphemeralClient()

            # 尝试获取现有集合
            try:
                self.collection = self.client.get_collection(self.store_name)
                self.stats["total_documents"] = self.collection.count()
                logger.info(f"加载现有集合: {self.store_name} ({self.stats['total_documents']} 文档)")

            except (ValueError, chromadb.errors.NotFoundError):
                # 创建新集合
                self.collection = self.client.create_collection(
                    name=self.store_name,
                    metadata={
                        "description": "医疗病例向量数据库",
                        "created": datetime.now().isoformat(),
                        "embedding_model": self.embedding_model,
                        "source": "MedDialog文本文件"
                    }
                )
                logger.info(f"创建新集合: {self.store_name}")

            self.stats["last_update"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"初始化ChromaDB失败: {e}")
            raise

    def add_documents(self,
                      documents: List[Dict[str, Any]],
                      batch_size: int = 2000,  # 增大到2000
                      show_progress: bool = True):
        """快速添加文档，GPU优化版本"""
        if not documents:
            logger.warning("没有文档可添加")
            return

        total_docs = len(documents)
        logger.info(f"开始添加 {total_docs} 个文档到向量数据库")

        import time
        import uuid
        import torch  # 添加torch用于GPU监控
        start_time = time.time()

        # 检查GPU是否可用
        if torch.cuda.is_available():
            logger.info(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.warning("❌ GPU不可用，使用CPU会非常慢")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:min(i + batch_size, total_docs)]

            # 准备数据
            batch_ids = [str(uuid.uuid4()) for _ in range(len(batch))]  # UUID生成ID
            batch_contents = [doc.get("content", "") for doc in batch]
            batch_metadatas = [doc.get("metadata", {}) for doc in batch]

            # 🔥 关键修改：增大嵌入批次大小，充分利用GPU
            try:
                # 根据GPU内存动态调整嵌入批次大小
                if torch.cuda.is_available():
                    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    if free_memory > 6 * 1024 ** 3:  # 6GB以上
                        embed_batch_size = 512
                    elif free_memory > 3 * 1024 ** 3:  # 3GB以上
                        embed_batch_size = 256
                    else:
                        embed_batch_size = 128
                else:
                    embed_batch_size = 64  # CPU用小批次

                # 生成嵌入 - 使用大批次
                embeddings = self.embedder.encode(batch_contents, batch_size=1024)
                batch_embeddings = embeddings.tolist()

                # 添加到集合
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                # 更新统计
                self.stats["total_documents"] += len(batch)

            except Exception as e:
                logger.error(f"批次 {i} 添加失败: {e}")
                # 如果失败，尝试更小的批次
                self._add_with_smaller_batches(batch_contents, batch_metadatas, batch_ids)

            # 进度报告
            if show_progress and (i // batch_size) % 5 == 0:
                elapsed_time = time.time() - start_time
                docs_per_second = (i + len(batch)) / elapsed_time if elapsed_time > 0 else 0
                remaining_time = (total_docs - (i + len(batch))) / docs_per_second if docs_per_second > 0 else 0
                progress = (i + len(batch)) / total_docs * 100

                # GPU内存监控
                gpu_info = ""
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    gpu_info = f" | GPU内存: {allocated:.1f}/{reserved:.1f}GB"

                logger.info(f"进度: {i + len(batch)}/{total_docs} ({progress:.1f}%) | "
                            f"速度: {docs_per_second:.1f} doc/s | "
                            f"预计剩余: {remaining_time / 3600:.1f}小时{gpu_info}")

        total_elapsed = time.time() - start_time
        logger.info(f"✅ 添加完成！总计 {self.stats['total_documents']} 个文档")
        logger.info(f"   总耗时: {total_elapsed:.1f}秒，平均速度: {total_docs / total_elapsed:.1f} doc/s")



    def _generate_doc_id(self, doc: Dict) -> str:
        """生成文档ID - 确保唯一性"""
        import hashlib
        import uuid

        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        # 使用更多信息生成唯一ID
        # 1. 文档内容哈希
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        # 2. 元数据信息
        metadata_str = ""
        if metadata:
            # 提取关键元数据字段
            key_fields = ['disease', 'year', 'case_id', 'source_file']
            for field in key_fields:
                if field in metadata:
                    metadata_str += str(metadata[field]) + "_"

        # 3. 如果文档有自带ID，使用它
        if doc.get("id"):
            return f"doc_{doc['id']}_{content_hash}"

        # 4. 如果没有足够信息，添加UUID
        if not metadata_str and len(content) < 10:
            # 内容太短，使用UUID
            unique_id = str(uuid.uuid4())[:12]
            return f"doc_{unique_id}"

        # 组合生成ID
        if metadata_str:
            # 使用元数据和内容哈希
            combined = metadata_str + content_hash
            doc_id = f"doc_{hashlib.md5(combined.encode()).hexdigest()[:16]}"
        else:
            # 仅使用内容哈希
            doc_id = f"doc_{content_hash}"

        return doc_id

    def search(self,
               query: str,
               top_k: int = 10,
               score_threshold: float = 0.5,
               filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件

        Returns:
            搜索结果列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.embedder.encode([query])

            # 构建查询参数
            query_kwargs = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": top_k * 2,  # 多取一些用于过滤
                "include": ["documents", "metadatas", "distances"]
            }

            # 添加过滤条件
            if filter_conditions:
                query_kwargs["where"] = filter_conditions

            # 执行查询
            results = self.collection.query(**query_kwargs)

            # 处理结果
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # 转换为相似度

                    if similarity >= score_threshold:
                        doc = {
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "similarity": similarity,
                            "distance": distance,
                            "source": "vector_store"
                        }
                        retrieved_docs.append(doc)

            # 按相似度排序
            retrieved_docs.sort(key=lambda x: x["similarity"], reverse=True)

            # 返回top_k
            return retrieved_docs[:top_k]

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def similarity_search_with_score(self,
                                    query: str,
                                    k: int = 5) -> List[tuple]:
        """带分数的相似度搜索"""
        results = self.search(query, top_k=k, score_threshold=0.0)

        return [(doc["content"], doc["similarity"]) for doc in results]

    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """根据ID获取文档"""
        try:
            results = self.collection.get(ids=ids)

            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    "id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i],
                    "source": "vector_store"
                }
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"根据ID获取文档失败: {e}")
            return []

    def delete_by_ids(self, ids: List[str]):
        """根据ID删除文档"""
        try:
            self.collection.delete(ids=ids)
            self.stats["total_documents"] = self.collection.count()
            logger.info(f"删除了 {len(ids)} 个文档")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")

    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict = None):
        """更新文档"""
        try:
            # 生成新嵌入
            new_embedding = self.embedder.encode([new_content])

            # 更新元数据
            metadata = new_metadata or {}
            metadata["updated_time"] = datetime.now().isoformat()

            # 更新文档
            self.collection.update(
                ids=[doc_id],
                embeddings=new_embedding.tolist(),
                documents=[new_content],
                metadatas=[metadata]
            )

            logger.info(f"文档 {doc_id} 更新成功")

        except Exception as e:
            logger.error(f"更新文档失败: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.collection.count()

            # 获取一些统计信息
            metadata = self.collection.metadata or {}

            info = {
                "collection_name": self.store_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "metadata": metadata,
                "storage_path": str(self.store_path) if self.store_path else "内存",
                "last_update": self.stats["last_update"]
            }

            return info

        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}

    def create_index(self):
        """创建索引（ChromaDB自动管理）"""
        logger.info("ChromaDB自动管理索引，无需手动创建")
        return True

    def save_stats(self, file_path: Optional[Path] = None):
        """保存统计信息"""
        if not file_path:
            file_path = self.store_path / "stats.json" if self.store_path else Path("vector_store_stats.json")

        stats_data = {
            **self.stats,
            "collection_info": self.get_collection_info(),
            "save_time": datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        logger.info(f"统计信息已保存到 {file_path}")

    def clear_collection(self):
        """清空集合"""
        try:
            self.client.reset()
            self.stats["total_documents"] = 0
            self.stats["last_update"] = datetime.now().isoformat()
            logger.warning("集合已清空")
        except Exception as e:
            logger.error(f"清空集合失败: {e}")

def create_vector_store_from_documents(
    documents: List[Dict],
    store_name: str = "medical_cases_v1",
    embedding_model: str = DEFAULT_EMBEDDER,
    batch_size: int = 100,
    rebuild: bool = False
) -> MedicalVectorStore:
    """
    从文档创建向量数据库

    Args:
        documents: 文档列表
        store_name: 存储名称
        embedding_model: 嵌入模型
        batch_size: 批量大小
        rebuild: 是否重新构建

    Returns:
        MedicalVectorStore实例
    """

    logger.info(f"创建向量数据库: {store_name}")

    # 初始化向量数据库
    vector_store = MedicalVectorStore(
        store_name=store_name,
        embedding_model=embedding_model,
        persist=True
    )

    # 检查是否已有数据
    info = vector_store.get_collection_info()
    has_existing_data = info.get("document_count", 0) > 0

    if has_existing_data and not rebuild:
        logger.info(f"向量数据库已存在，包含 {info['document_count']} 个文档")
        choice = input("是否重新构建？(y/n): ").lower()
        if choice != 'y':
            return vector_store

    # 清空现有数据（如果需要）
    if rebuild and has_existing_data:
        logger.info("清空现有数据...")
        vector_store.clear_collection()

    # 添加文档
    vector_store.add_documents(documents, batch_size=batch_size)

    # 保存统计信息
    vector_store.save_stats()

    logger.info("✅ 向量数据库创建完成")

    return vector_store

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗向量数据库管理")
    parser.add_argument("--store-name", type=str, default="medical_cases", help="存储名称")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDER, help="嵌入模型")
    parser.add_argument("--rebuild", action="store_true", help="重新构建")
    parser.add_argument("--search", type=str, help="测试搜索查询")
    parser.add_argument("--top-k", type=int, default=5, help="搜索结果数量")

    args = parser.parse_args()

    # 创建向量数据库
    vector_store = MedicalVectorStore(
        store_name=args.store_name,
        embedding_model=args.embedding_model
    )

    # 显示信息
    info = vector_store.get_collection_info()
    print("\n" + "="*60)
    print("📊 向量数据库信息")
    print("="*60)
    for key, value in info.items():
        if key != "metadata":
            print(f"{key}: {value}")

    # 测试搜索
    if args.search:
        print(f"\n🔍 搜索测试: '{args.search}'")
        results = vector_store.search(args.search, top_k=args.top_k)

        for i, result in enumerate(results):
            print(f"\n结果 {i+1} (相似度: {result['similarity']:.3f}):")
            content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"内容: {content_preview}")

            if result['metadata']:
                disease = result['metadata'].get('disease', '未知')
                print(f"疾病: {disease}")