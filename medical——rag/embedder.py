"""
嵌入模型管理器
支持多种嵌入模型和GPU优化
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import chromadb
from chromadb.utils import embedding_functions

from config import EMBEDDING_MODELS, DEVICE, DEFAULT_EMBEDDER, MODELS_DIR

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入模型管理器"""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER):
        self.model_name = model_name
        self.model_config = EMBEDDING_MODELS.get(model_name, EMBEDDING_MODELS[DEFAULT_EMBEDDER])
        self.model = None
        self.tokenizer = None
        self.dimension = self.model_config["dimension"]

    # def load_model(self):
    #     """加载嵌入模型"""
    #     logger.info(f"加载嵌入模型: {self.model_name}")
    #
    #     try:
    #         # 使用sentence-transformers（支持更多模型）
    #         self.model = SentenceTransformer(
    #             self.model_config["name"],
    #             device=DEVICE,
    #             cache_folder=str(MODELS_DIR / "embedding_models")
    #         )
    #
    #         # 测试嵌入
    #         test_embedding = self.encode(["测试句子"])
    #         logger.info(f"模型加载成功，维度: {test_embedding.shape[1]}")
    #
    #     except Exception as e:
    #         logger.error(f"加载模型失败: {e}")
    #
    #         # 尝试使用transformers
    #         try:
    #             self.tokenizer = AutoTokenizer.from_pretrained(
    #                 self.model_config["name"],
    #                 cache_dir=str(MODELS_DIR / "embedding_models")
    #             )
    #             self.model = AutoModel.from_pretrained(
    #                 self.model_config["name"],
    #                 cache_dir=str(MODELS_DIR / "embedding_models")
    #             ).to(DEVICE)
    #
    #             logger.info("使用transformers加载模型成功")
    #         except Exception as e2:
    #             logger.error(f"备选加载也失败: {e2}")
    #             raise

    def load_model(self):
        """加载嵌入模型 - 支持本地路径"""
        logger.info(f"加载嵌入模型: {self.model_name}")

        model_config = self.model_config

        # 1. 优先尝试本地路径
        local_path = model_config.get("local_path")
        if local_path and Path(local_path).exists():
            logger.info(f"从本地路径加载模型: {local_path}")
            try:
                self.model = SentenceTransformer(
                    local_path,
                    device=DEVICE
                )
                # 测试一下确保模型工作
                test_embedding = self.encode(["测试句子"])
                logger.info(f"✅ 本地模型加载成功，维度: {test_embedding.shape[1]}")
                return  # 成功则直接返回
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")
                # 失败后继续尝试其他方法
        else:
            logger.info(f"未找到本地模型路径或路径不存在: {local_path}")

        # 2. 尝试在线加载（使用sentence-transformers）
        try:
            logger.info(f"尝试在线加载模型: {model_config['name']}")
            self.model = SentenceTransformer(
                model_config["name"],
                device=DEVICE,
                cache_folder=str(MODELS_DIR / "embedding_models")
            )

            # 测试嵌入
            test_embedding = self.encode(["测试句子"])
            logger.info(f"✅ 在线模型加载成功，维度: {test_embedding.shape[1]}")
            return  # 成功则返回

        except Exception as e:
            logger.error(f"sentence-transformers在线加载失败: {e}")
            # 继续尝试transformers

        # 3. 尝试使用transformers加载
        try:
            logger.info("尝试使用transformers加载...")
            from transformers import AutoTokenizer, AutoModel

            # 如果本地路径存在但sentence-transformers加载失败，可以尝试transformers
            model_name = local_path if local_path and Path(local_path).exists() else model_config["name"]

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(MODELS_DIR / "embedding_models")
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(MODELS_DIR / "embedding_models")
            ).to(DEVICE)

            logger.info("✅ 使用transformers加载模型成功")
            return

        except Exception as e2:
            logger.error(f"所有加载方式都失败: {e2}")
            raise RuntimeError(f"无法加载模型 '{self.model_name}'，请检查网络连接或本地模型文件")

    def encode(self, texts: List[str],
               batch_size: int = 512,
               normalize: bool = True) -> np.ndarray:
        """编码文本为向量"""
        if self.model is None:
            self.load_model()

        # 分批处理避免内存溢出
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if isinstance(self.model, SentenceTransformer):
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )
            else:
                # 使用transformers模型
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.model_config["max_length"],
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用平均池化
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                if normalize:
                    batch_embeddings = batch_embeddings / np.linalg.norm(
                        batch_embeddings, axis=1, keepdims=True
                    )

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_documents(self, documents: List[Dict],
                         batch_size: int = 32) -> List[Dict]:
        """编码文档列表"""
        texts = [doc['content'] for doc in documents]
        embeddings = self.encode(texts, batch_size)

        # 添加嵌入到文档
        for i, doc in enumerate(documents):
            doc['embedding'] = embeddings[i].tolist()

        return documents

    def get_similarity(self, query_embedding: np.ndarray,
                       doc_embeddings: np.ndarray) -> np.ndarray:
        """计算相似度"""
        # 余弦相似度
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarities

    def update_model(self, new_model_name: str):
        """更新嵌入模型"""
        if new_model_name in EMBEDDING_MODELS:
            self.model_name = new_model_name
            self.model_config = EMBEDDING_MODELS[new_model_name]
            self.model = None  # 强制重新加载
            self.load_model()
            logger.info(f"嵌入模型已更新为: {new_model_name}")
        else:
            logger.error(f"未知的嵌入模型: {new_model_name}")


    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "dimension": self.dimension,
            "is_loaded": self.model is not None,
            "device": str(DEVICE),
            "embedding_type": "sentence_transformers" if isinstance(self.model,
                                                                    SentenceTransformer) else "transformers"
        }

