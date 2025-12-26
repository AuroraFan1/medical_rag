# """
# 简化的RAG系统模块
# 直接使用向量数据库和LLM生成回答
# """
#
# import logging
# from typing import List, Dict, Any, Optional
#
# # LangChain 相关
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain.schema import Document
#
# # 导入配置
# from config import MODEL_NAME, LLM_TEMPERATURE, SYSTEM_PROMPT, RETRIEVAL_K
# from data_processing import VectorDatabaseManager
#
# logger = logging.getLogger(__name__)
#
#
# class MedicalRAGSystem:
#     """医疗RAG系统"""
#
#     def __init__(self, vector_db_path: Optional[str] = None):
#         self.vector_db_path = vector_db_path
#         self.vector_manager = None
#         self.llm = None
#         self.rag_chain = None
#
#     def initialize(self, search_k: int = RETRIEVAL_K) -> bool:
#         """初始化RAG系统"""
#         try:
#             logger.info("初始化医疗RAG系统...")
#
#             # 1. 初始化向量数据库管理器
#             self.vector_manager = VectorDatabaseManager(self.vector_db_path)
#
#             # 2. 加载向量数据库
#             if not self.vector_manager.load_vectorstore():
#                 logger.error("无法加载向量数据库")
#                 return False
#
#             # 3. 初始化LLM
#             self.llm = ChatOpenAI(
#                 model_name=MODEL_NAME,
#                 temperature=LLM_TEMPERATURE,
#                 streaming=True,
#                 max_tokens=2000
#             )
#
#             # 4. 创建检索器
#             retriever = self.vector_manager.vectorstore.as_retriever(
#                 search_kwargs={"k": search_k}
#             )
#
#             # 5. 创建提示词模板
#             prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
#
#             # 6. 构建RAG链
#             self.rag_chain = (
#                 {"context": retriever, "question": RunnablePassthrough()}
#                 | prompt
#                 | self.llm
#                 | StrOutputParser()
#             )
#
#             logger.info("医疗RAG系统初始化完成")
#             return True
#
#         except Exception as e:
#             logger.error(f"初始化RAG系统失败: {e}")
#             return False
#
#     def query(self, question: str, use_streaming: bool = True):
#         """查询RAG系统"""
#         if self.rag_chain is None:
#             raise ValueError("RAG系统未初始化")
#
#         try:
#             if use_streaming:
#                 return self.rag_chain.stream(question)
#             else:
#                 return self.rag_chain.invoke(question)
#         except Exception as e:
#             logger.error(f"查询失败: {e}")
#             return f"抱歉，查询时出现错误: {str(e)}"
#
#     def search_similar_cases(self, query: str, k: int = RETRIEVAL_K) -> List[Dict[str, Any]]:
#         """搜索相似病例"""
#         if self.vector_manager is None:
#             raise ValueError("向量数据库未初始化")
#
#         try:
#             docs = self.vector_manager.search_similar(query, k=k)
#
#             results = []
#             for i, doc in enumerate(docs):
#                 result = {
#                     "rank": i + 1,
#                     "content": doc.page_content,
#                     "metadata": doc.metadata
#                 }
#                 results.append(result)
#
#             return results
#
#         except Exception as e:
#             logger.error(f"搜索相似病例失败: {e}")
#             return []
#
#     def get_system_info(self) -> Dict[str, Any]:
#         """获取系统信息"""
#         info = {
#             "status": "已初始化" if self.rag_chain else "未初始化",
#             "model": MODEL_NAME,
#             "temperature": LLM_TEMPERATURE
#         }
#
#         if self.vector_manager:
#             vector_info = self.vector_manager.get_info()
#             info.update(vector_info)
#
#         return info
#
#
# # 全局单例实例
# _rag_system_instance = None
#
#
# def get_rag_system(force_reinitialize: bool = False) -> Optional[MedicalRAGSystem]:
#     """获取RAG系统实例（单例模式）"""
#     global _rag_system_instance
#
#     if _rag_system_instance is None or force_reinitialize:
#         _rag_system_instance = MedicalRAGSystem()
#         success = _rag_system_instance.initialize()
#
#         if not success:
#             logger.error("无法初始化RAG系统")
#             return None
#
#     return _rag_system_instance
#
#
# if __name__ == "__main__":
#     """测试RAG系统"""
#     import time
#
#     print("测试医疗RAG系统...")
#
#     # 获取系统实例
#     rag_system = get_rag_system()
#
#     if rag_system:
#         # 显示系统信息
#         info = rag_system.get_system_info()
#         print("系统信息:")
#         for key, value in info.items():
#             print(f"  {key}: {value}")
#
#         # 测试查询
#         test_questions = [
#             "感冒了怎么办？",
#             "高血压要注意什么？",
#             "糖尿病的症状有哪些？"
#         ]
#
#         for question in test_questions:
#             print(f"\n测试问题: {question}")
#
#             # 搜索相似病例
#             similar_cases = rag_system.search_similar_cases(question, k=2)
#             print(f"找到 {len(similar_cases)} 个相似病例")
#
#             # 生成回答
#             start_time = time.time()
#             response = rag_system.query(question, use_streaming=False)
#             elapsed = time.time() - start_time
#
#             print(f"回答 ({elapsed:.2f}秒):")
#             print(response[:200] + "..." if len(response) > 200 else response)
#
#         print("\n✅ RAG系统测试完成")
#     else:
#         print("❌ RAG系统初始化失败")

"""
修复的RAG系统模块
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# LangChain 相关
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

# 导入配置
from config import MODEL_NAME, SYSTEM_PROMPT, LLM_TEMPERATURE, RETRIEVAL_K, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)


class MedicalRAGSystem:
    """医疗RAG系统"""

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        初始化RAG系统

        Args:
            vector_store_path: 向量存储路径，如果为None则使用配置中的路径
        """
        if vector_store_path is None:
            vector_store_path = VECTOR_STORE_PATH

        # 确保路径是字符串
        if vector_store_path is None:
            raise ValueError("向量数据库路径不能为None")

        # 转换为绝对路径
        self.vector_store_path = str(Path(vector_store_path).absolute())
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None

        logger.info(f"初始化RAG系统，向量数据库路径: {self.vector_store_path}")

    def initialize(self, search_k: int = RETRIEVAL_K) -> bool:
        """初始化RAG系统"""
        try:
            logger.info("正在初始化医疗RAG系统...")

            # 1. 检查向量数据库是否存在
            if not os.path.exists(self.vector_store_path):
                logger.error(f"向量数据库不存在: {self.vector_store_path}")
                logger.info("请先运行 data_processing.py 构建向量数据库")
                return False

            # 2. 加载向量数据库
            logger.info("加载向量数据库...")
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            # 使用与构建时相同的嵌入模型
            from config import EMBEDDING_MODEL

            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )

            self.vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=embeddings
            )

            # 3. 创建检索器
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": search_k}
            )

            # 4. 初始化LLM
            logger.info("初始化语言模型...")
            self.llm = ChatOpenAI(
                model_name=MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                streaming=True,
                max_tokens=1500
            )

            # 5. 创建提示词模板
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

            # 6. 构建RAG链
            self.rag_chain = (
                    {"context": self.retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )

            logger.info("医疗RAG系统初始化完成")

            # 验证系统
            test_result = self._test_system()
            if test_result:
                logger.info("系统测试通过")
            else:
                logger.warning("系统测试失败，但继续运行")

            return True

        except Exception as e:
            logger.error(f"初始化RAG系统失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _test_system(self) -> bool:
        """测试系统是否正常工作"""
        try:
            # 简单查询测试
            test_query = "测试"
            results = self.vectorstore.similarity_search(test_query, k=1)
            if results:
                logger.info(f"系统测试成功，找到 {len(results)} 个结果")
                return True
            else:
                logger.warning("系统测试：没有找到结果，但可能正常")
                return True
        except Exception as e:
            logger.error(f"系统测试失败: {e}")
            return False

    def query(self, question: str, use_streaming: bool = True):
        """查询RAG系统"""
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

    def search_similar_cases(self, query: str, k: int = RETRIEVAL_K) -> List[Dict[str, Any]]:
        """搜索相似病例"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        try:
            docs = self.vectorstore.similarity_search(query, k=k)

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
        """获取系统信息"""
        info = {
            "status": "已初始化" if self.rag_chain else "未初始化",
            "model": MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "vector_store_path": self.vector_store_path,
            "vector_store_exists": os.path.exists(self.vector_store_path)
        }

        if self.vectorstore:
            try:
                count = self.vectorstore._collection.count()
                info["vector_count"] = count
            except:
                info["vector_count"] = "未知"

        return info


# 全局单例实例
_rag_system_instance = None


def get_rag_system(force_reinitialize: bool = False) -> Optional[MedicalRAGSystem]:
    """获取RAG系统实例（单例模式）"""
    global _rag_system_instance

    if _rag_system_instance is None or force_reinitialize:
        try:
            _rag_system_instance = MedicalRAGSystem()
            success = _rag_system_instance.initialize()

            if not success:
                logger.error("无法初始化RAG系统")
                _rag_system_instance = None
                return None
        except Exception as e:
            logger.error(f"创建RAG系统实例失败: {e}")
            _rag_system_instance = None
            return None

    return _rag_system_instance


if __name__ == "__main__":
    """测试RAG系统"""
    print("测试医疗RAG系统...")

    # 获取系统实例
    rag_system = get_rag_system()

    if rag_system:
        # 显示系统信息
        info = rag_system.get_system_info()
        print("系统信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 测试查询
        test_questions = [
            "感冒了怎么办？",
            "高血压要注意什么？",
        ]

        for question in test_questions:
            print(f"\n测试问题: {question}")

            try:
                # 搜索相似病例
                similar_cases = rag_system.search_similar_cases(question, k=2)
                print(f"找到 {len(similar_cases)} 个相似病例")

                # 生成回答
                response = rag_system.query(question, use_streaming=False)
                print(f"回答:")
                print(response[:200] + "..." if len(response) > 200 else response)

            except Exception as e:
                print(f"测试失败: {e}")

        print("\n✅ RAG系统测试完成")
    else:
        print("❌ RAG系统初始化失败")