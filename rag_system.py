"""
RAG系统核心模块
"""

from typing import List, Dict, Any, Optional
import logging

# LangChain 相关
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

from config import MODEL_NAME, SYSTEM_PROMPT, LLM_TEMPERATURE
from data_processing import VectorStoreManager

logger = logging.getLogger(__name__)


class MedicalRAGSystem:
    """医疗RAG系统"""

    def __init__(self, vector_store_path: str = "chroma_db_medical"):
        """
        初始化RAG系统

        Args:
            vector_store_path: 向量存储路径
        """
        self.vector_store_path = vector_store_path
        self.vector_manager = None
        self.rag_chain = None
        self.llm = None

    def initialize(self, search_k: int = 5) -> bool:
        """
        初始化RAG系统

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("正在初始化医疗RAG系统...")

            # 1. 初始化向量存储管理器
            self.vector_manager = VectorStoreManager(self.vector_store_path)
            vector_exists = self.vector_manager.initialize()

            if not vector_exists:
                logger.error("向量数据库不存在，请先运行 data_processing.py 构建向量数据库")
                return False

            # 2. 获取检索器
            retriever = self.vector_manager.get_retriever({"k": search_k})

            # 3. 初始化语言模型
            self.llm = ChatOpenAI(
                model_name=MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                streaming=True
            )

            # 4. 创建提示词模板
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

            # 5. 构建RAG链
            self.rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )

            logger.info("医疗RAG系统初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化RAG系统时出错: {e}")
            return False

    def query(self, question: str, use_streaming: bool = False) -> Any:
        """
        查询RAG系统

        Args:
            question: 用户问题
            use_streaming: 是否使用流式输出

        Returns:
            模型回答或流式生成器
        """
        if self.rag_chain is None:
            raise ValueError("RAG系统未初始化，请先调用 initialize()")

        try:
            if use_streaming:
                return self.rag_chain.stream(question)
            else:
                return self.rag_chain.invoke(question)

        except Exception as e:
            logger.error(f"查询时出错: {e}")
            return f"抱歉，查询时出现错误: {str(e)}"

    def search_similar_cases(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似病例

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似病例列表
        """
        if self.vector_manager is None or self.vector_manager.vectorstore is None:
            raise ValueError("向量存储未初始化")

        try:
            # 直接使用向量存储进行搜索
            docs = self.vector_manager.vectorstore.similarity_search(query, k=k)

            # 格式化结果
            results = []
            for i, doc in enumerate(docs):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"搜索相似病例时出错: {e}")
            return []

    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        """
        info = {
            "status": "已初始化" if self.rag_chain else "未初始化",
            "model": MODEL_NAME,
            "vector_store": self.vector_store_path
        }

        if self.vector_manager:
            vector_info = self.vector_manager.get_vectorstore_info()
            info.update(vector_info)

        return info


# 单例模式，方便全局使用
_rag_system_instance = None


def get_rag_system(force_reinitialize: bool = False) -> MedicalRAGSystem:
    """
    获取RAG系统实例（单例模式）

    Args:
        force_reinitialize: 是否强制重新初始化

    Returns:
        MedicalRAGSystem实例
    """
    global _rag_system_instance

    if _rag_system_instance is None or force_reinitialize:
        _rag_system_instance = MedicalRAGSystem()
        success = _rag_system_instance.initialize()
        if not success:
            logger.error("无法初始化RAG系统")
            return None

    return _rag_system_instance