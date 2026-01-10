"""
完整的医疗RAG系统
集成所有组件
"""

import json
from typing import Dict, Any, Optional, List, Generator
import logging
from pathlib import Path
from datetime import datetime

from config import (
    SYSTEM_CONFIG, PROMPT_TEMPLATES, MODEL_CONFIGS,
    DEFAULT_LLM, DEFAULT_EMBEDDER, RETRIEVAL_CONFIG
)

# 导入其他模块
from data_processing import MedicalCase, TextFileProcessor, DocumentSplitter
from embedder import EmbeddingManager
from vector_store import MedicalVectorStore, create_vector_store_from_documents
from retriever import MedicalRetriever
from llm_manager import LLMManager
from chat_session import ChatSession

logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    """医疗RAG主系统"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # 系统组件
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.chat_session = None

        # 系统状态
        self.is_initialized = False
        self.system_info = {
            "initialized": False,
            "components": {},
            "metrics": {
                "queries_processed": 0,
                "avg_response_time": 0,
                "success_rate": 1.0
            }
        }

    def initialize(self,
                   embedding_model: str = DEFAULT_EMBEDDER,
                   llm_model: str = DEFAULT_LLM,
                   use_api: bool = False,
                   vector_store_name: str = "medical_cases",
                   enable_hybrid: bool = True,
                   enable_rerank: bool = True,
                   session_id: Optional[str] = None) -> bool:
        """初始化系统"""

        logger.info("初始化医疗RAG系统...")

        try:
            # 1. 初始化嵌入管理器
            self.embedder = EmbeddingManager(embedding_model)
            self.system_info["components"]["embedder"] = self.embedder.get_model_info()
            print("嵌入管理器")
            # 2. 初始化向量数据库
            self.vector_store = MedicalVectorStore(
                store_name=vector_store_name,
                embedding_model=embedding_model,
                persist=True
            )
            self.system_info["components"]["vector_store"] = self.vector_store.get_collection_info()

            # 3. 初始化检索器
            self.retriever = MedicalRetriever(
                vector_store=self.vector_store,
                enable_hybrid=enable_hybrid,
                enable_rerank=enable_rerank,
                enable_diversity=RETRIEVAL_CONFIG.get("enable_diversity", True)
            )
            self.system_info["components"]["retriever"] = self.retriever.get_retrieval_stats()

            # 4. 初始化LLM
            self.llm = LLMManager(
                model_name=llm_model,
                use_api=use_api,
                load_in_4bit=True
            )
            self.system_info["components"]["llm"] = self.llm.get_model_info()

            # 5. 初始化聊天会话
            self.chat_session = ChatSession(
                session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                max_turns=SYSTEM_CONFIG['max_conversation_turns']
            )
            self.system_info["components"]["chat_session"] = self.chat_session.to_dict()

            # 标记为已初始化
            self.is_initialized = True
            self.system_info["initialized"] = True
            self.system_info["initialized_time"] = datetime.now().isoformat()

            logger.info("✅ 系统初始化完成")
            return True

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False

    def query(self,
              question: str,
              use_streaming: bool = SYSTEM_CONFIG['enable_streaming'],
              top_k: int = RETRIEVAL_CONFIG['top_k'],
              filter_conditions: Optional[Dict] = None,
              session_id: Optional[str] = None) -> Dict[str, Any]:
        """处理查询"""

        if not self.is_initialized:
            raise ValueError("系统未初始化，请先调用initialize()")

        start_time = datetime.now()

        try:
            # 1. 检索相关文档
            retrieved_docs = self.retriever.retrieve(
                question,
                top_k=top_k,
                filter_conditions=filter_conditions
            )

            # 2. 构建上下文
            context = self._build_context(retrieved_docs)

            # 3. 获取对话历史（如果指定了session_id）
            history = None
            if self.chat_session and (session_id is None or session_id == self.chat_session.session_id):
                history = self.chat_session.get_recent_history()

            # 4. 生成回答
            if use_streaming:
                response_generator = self.llm.generate(
                    prompt=question,
                    context=context,
                    history=history,
                    streaming=True
                )

                # 流式响应处理
                full_response = ""
                for chunk in response_generator:
                    full_response += chunk
                    # 这里可以实时发送chunk到前端

                response = full_response
            else:
                response = self.llm.generate(
                    prompt=question,
                    context=context,
                    history=history,
                    streaming=False
                )

            # 5. 检查不确定性
            is_uncertain = self.llm.check_uncertainty(response, context)

            if is_uncertain:
                # 使用不确定性模板
                context_summary = self._summarize_context(retrieved_docs)
                response = PROMPT_TEMPLATES["uncertain"].format(
                    context_summary=context_summary,
                    question=question
                )

            # 6. 添加引用
            response_with_citations = self.llm.format_with_citations(response, retrieved_docs)

            # 7. 更新会话
            if self.chat_session and (session_id is None or session_id == self.chat_session.session_id):
                self.chat_session.add_message("user", question)
                self.chat_session.add_message(
                    "assistant",
                    response_with_citations,
                    metadata={
                        "sources": [doc.get('metadata', {}) for doc in retrieved_docs],
                        "retrieved_count": len(retrieved_docs),
                        "is_uncertain": is_uncertain,
                        "response_time": None  # 稍后更新
                    }
                )

                if is_uncertain:
                    self.chat_session.mark_uncertain_answer()

            # 8. 计算响应时间
            response_time = (datetime.now() - start_time).total_seconds()

            # 9. 更新指标
            self._update_metrics(response_time)

            # 10. 构建返回结果
            result = {
                "response": response_with_citations,
                "sources": retrieved_docs,
                "context": context[:500] + "..." if len(context) > 500 else context,
                "metadata": {
                    "response_time": response_time,
                    "sources_count": len(retrieved_docs),
                    "is_uncertain": is_uncertain,
                    "session_id": session_id or self.chat_session.session_id if self.chat_session else None,
                    "query_length": len(question),
                    "response_length": len(response_with_citations)
                }
            }

            # 更新会话中的响应时间
            if self.chat_session:
                last_message = self.chat_session.history[-1] if self.chat_session.history else None
                if last_message and last_message.get("role") == "assistant":
                    last_message["metadata"]["response_time"] = response_time

            logger.info(f"查询处理完成: {response_time:.2f}s, 返回 {len(retrieved_docs)} 个来源")

            return result

        except Exception as e:
            logger.error(f"查询处理失败: {e}")

            # 更新指标（失败）
            self._update_metrics(0, success=False)

            return {
                "response": f"抱歉，处理查询时出现错误: {str(e)}",
                "sources": [],
                "context": "",
                "metadata": {
                    "error": str(e),
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "success": False
                }
            }

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建上下文"""

        if not retrieved_docs:
            return "无相关病例信息"

        context_parts = ["相关病例信息："]

        for i, doc in enumerate(retrieved_docs[:5]):  # 限制前5个文档
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 提取关键信息
            disease = metadata.get("disease", "未知疾病")
            hospital = metadata.get("hospital", "未知医院")

            # 格式化
            doc_info = f"[病例{i+1}]"
            if disease != "未知疾病":
                doc_info += f" 疾病：{disease}"
            if hospital != "未知医院":
                doc_info += f" 医院：{hospital}"

            # 添加内容摘要
            content_preview = content[:300] + "..." if len(content) > 300 else content
            doc_info += f"\n{content_preview}\n"

            context_parts.append(doc_info)

        return "\n".join(context_parts)

    def _summarize_context(self, retrieved_docs: List[Dict]) -> str:
        """摘要上下文"""

        if not retrieved_docs:
            return "无相关病例信息"

        # 提取疾病和症状关键词
        diseases = set()
        symptoms_keywords = []

        for doc in retrieved_docs[:3]:
            metadata = doc.get("metadata", {})
            disease = metadata.get("disease", "")
            if disease:
                diseases.add(disease)

            # 简单提取症状关键词
            content = doc.get("content", "").lower()
            symptom_words = ["症状", "表现", "感觉", "疼痛", "不适", "发烧", "咳嗽"]
            for word in symptom_words:
                if word in content:
                    symptoms_keywords.append(word)

        summary_parts = []
        if diseases:
            summary_parts.append(f"相关疾病：{', '.join(list(diseases)[:3])}")

        if symptoms_keywords:
            unique_symptoms = list(set(symptoms_keywords))[:5]
            summary_parts.append(f"涉及症状：{', '.join(unique_symptoms)}")

        return "；".join(summary_parts) if summary_parts else "医疗咨询病例"

    def _update_metrics(self, response_time: float, success: bool = True):
        """更新系统指标"""

        self.system_info["metrics"]["queries_processed"] += 1

        # 更新平均响应时间
        current_avg = self.system_info["metrics"]["avg_response_time"]
        total_queries = self.system_info["metrics"]["queries_processed"]

        if total_queries == 1:
            self.system_info["metrics"]["avg_response_time"] = response_time
        else:
            self.system_info["metrics"]["avg_response_time"] = (
                (current_avg * (total_queries - 1) + response_time) / total_queries
            )

        # 更新成功率
        if not success:
            success_count = self.system_info["metrics"]["success_rate"] * (total_queries - 1)
            self.system_info["metrics"]["success_rate"] = success_count / total_queries

    def update_embedding_model(self, new_model: str) -> bool:
        """更新嵌入模型"""

        if not self.embedder:
            logger.error("嵌入管理器未初始化")
            return False

        success = self.embedder.update_model(new_model)

        if success:
            # 更新系统信息
            self.system_info["components"]["embedder"] = self.embedder.get_model_info()

            # 需要重新构建向量数据库或更新现有向量的嵌入
            logger.warning("⚠️ 嵌入模型已更新，建议重新构建向量数据库")

        return success

    def update_retrieval_strategy(self,
                                 enable_hybrid: Optional[bool] = None,
                                 enable_rerank: Optional[bool] = None,
                                 enable_diversity: Optional[bool] = None):
        """更新检索策略"""

        if not self.retriever:
            logger.error("检索器未初始化")
            return

        if enable_hybrid is not None:
            self.retriever.enable_hybrid = enable_hybrid

        if enable_rerank is not None:
            self.retriever.enable_rerank = enable_rerank

        if enable_diversity is not None:
            self.retriever.enable_diversity = enable_diversity

        # 更新系统信息
        self.system_info["components"]["retriever"] = self.retriever.get_retrieval_stats()

        logger.info("检索策略已更新")

    def update_prompt_template(self, template_type: str, new_template: str):
        """更新prompt模板"""

        if template_type in PROMPT_TEMPLATES:
            PROMPT_TEMPLATES[template_type] = new_template
            logger.info(f"Prompt模板 '{template_type}' 已更新")
        else:
            logger.warning(f"未知的模板类型: {template_type}")

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""

        return self.system_info

    def export_conversation(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """导出对话"""

        if not self.chat_session:
            return {"error": "无活跃会话"}

        if session_id and session_id != self.chat_session.session_id:
            return {"error": "会话ID不匹配"}

        return self.chat_session.to_dict()

    def clear_conversation(self, session_id: Optional[str] = None):
        """清空对话"""

        if not self.chat_session:
            logger.warning("无活跃会话可清空")
            return

        if session_id and session_id != self.chat_session.session_id:
            logger.warning("会话ID不匹配")
            return

        self.chat_session.clear_history()
        logger.info("对话已清空")

    def search_similar_cases(self,
                            query: str,
                            top_k: int = 10,
                            with_scores: bool = True) -> List[Dict]:
        """搜索相似病例（直接检索，不生成回答）"""

        if not self.retriever:
            raise ValueError("检索器未初始化")

        return self.retriever.retrieve(query, top_k=top_k)


def build_complete_rag_pipeline(
        data_dir: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDER,
        llm_model: str = DEFAULT_LLM,
        vector_store_name: str = "medical_cases_v1",
        test_mode: bool = False,
        rebuild: bool = False
) -> MedicalRAGSystem:
    """
    构建完整的RAG管道

    Args:
        data_dir: 数据目录
        embedding_model: 嵌入模型
        llm_model: LLM模型
        vector_store_name: 向量数据库名称
        test_mode: 测试模式
        rebuild: 重新构建

    Returns:
        MedicalRAGSystem实例
    """

    from data_processing import process_medical_texts

    logger.info("=" * 60)
    logger.info(f"构建医疗RAG系统管道 - LLM模型: {llm_model}")
    logger.info("=" * 60)

    # 1. 数据处理
    logger.info("步骤1: 数据处理")
    cases, documents = process_medical_texts(
        data_dir=data_dir,
        test_mode=test_mode,
        rebuild=rebuild
    )

    # 转换为向量数据库格式
    vector_docs = []
    for doc in documents:
        vector_docs.append({
            "id": doc.metadata.get("id", f"doc_{hash(doc.page_content) % 1000000}"),
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    # 2. 向量数据库构建
    logger.info("步骤2: 向量数据库构建")
    vector_store = create_vector_store_from_documents(
        documents=vector_docs,
        store_name=vector_store_name,
        embedding_model=embedding_model,
        batch_size=100,
        rebuild=rebuild
    )

    # 3. 系统初始化
    logger.info("步骤3: 系统初始化")
    rag_system = MedicalRAGSystem()

    # ✅ 打印日志确认传入的模型参数
    logger.info(f"将使用LLM模型: {llm_model}")

    success = rag_system.initialize(
        embedding_model=embedding_model,
        llm_model=llm_model,  # ✅ 直接传递参数
        use_api=False,
        vector_store_name=vector_store_name,
        enable_hybrid=True,
        enable_rerank=True
    )

    if not success:
        logger.error("系统初始化失败")
        return None

    # 4. 跳过BM25索引构建
    logger.info("步骤4: 跳过BM25索引构建")

    logger.info("✅ RAG系统构建完成")

    # 显示系统信息
    system_info = rag_system.get_system_info()
    print("\n" + "=" * 60)
    print("📊 系统构建完成")
    print("=" * 60)
    print(f"嵌入模型: {system_info['components']['embedder']['model_name']}")

    # 获取实际的LLM模型名称
    llm_info = system_info['components']['llm']
    actual_model = llm_info.get('model_name', 'Unknown')
    print(f"LLM模型: {actual_model}")

    # 检查是否是期望的模型
    expected_model = llm_model
    if actual_model != expected_model:
        print(f"⚠️ 警告: 期望模型 '{expected_model}'，实际加载模型 '{actual_model}'")

    print(f"向量数据库: {system_info['components']['vector_store']['document_count']} 文档")
    print("=" * 60)

    return rag_system

#
# def build_complete_rag_pipeline(
#     data_dir: Optional[str] = None,
#     embedding_model: str = DEFAULT_EMBEDDER,
#     llm_model: str = DEFAULT_LLM,
#     vector_store_name: str = "medical_cases_v1",
#     test_mode: bool = False,
#     rebuild: bool = False
# ) -> MedicalRAGSystem:
#     """
#     构建完整的RAG管道
#
#     Args:
#         data_dir: 数据目录
#         embedding_model: 嵌入模型
#         llm_model: LLM模型
#         vector_store_name: 向量数据库名称
#         test_mode: 测试模式
#         rebuild: 重新构建
#
#     Returns:
#         MedicalRAGSystem实例
#     """
#
#     from data_processing import process_medical_texts
#
#     logger.info("="*60)
#     logger.info("构建医疗RAG系统管道")
#     logger.info("="*60)
#
#     # 1. 数据处理
#     logger.info("步骤1: 数据处理")
#     cases, documents = process_medical_texts(
#         data_dir=data_dir,
#         test_mode=test_mode,
#         rebuild=rebuild
#     )
#
#     # 转换为向量数据库格式
#     vector_docs = []
#     for doc in documents:
#         vector_docs.append({
#             "id": doc.metadata.get("id", f"doc_{hash(doc.page_content) % 1000000}"),
#             "content": doc.page_content,
#             "metadata": doc.metadata
#         })
#
#     # 2. 向量数据库构建
#     logger.info("步骤2: 向量数据库构建")
#     vector_store = create_vector_store_from_documents(
#         documents=vector_docs,
#         store_name=vector_store_name,
#         embedding_model=embedding_model,
#         batch_size=100,
#         rebuild=rebuild
#     )
#
#     # 3. 系统初始化
#     logger.info("步骤3: 系统初始化")
#     rag_system = MedicalRAGSystem()
#
#     success = rag_system.initialize(
#         embedding_model=embedding_model,
#         llm_model=llm_model,
#         use_api=False,
#         vector_store_name=vector_store_name,
#         enable_hybrid=True,
#         enable_rerank=True
#     )
#
#     if not success:
#         logger.error("系统初始化失败")
#         return None
#
#     # 4. 构建BM25索引
#     # logger.info("步骤4: 构建BM25索引")
#     # rag_system.retriever.build_bm25_index(vector_docs)
#
#     logger.info("✅ RAG系统构建完成")
#
#     # 显示系统信息
#     system_info = rag_system.get_system_info()
#     print("\n" + "="*60)
#     print("📊 系统构建完成")
#     print("="*60)
#     print(f"嵌入模型: {system_info['components']['embedder']['model_name']}")
#     print(f"LLM模型: {system_info['components']['llm']['model_name']}")
#     print(f"向量数据库: {system_info['components']['vector_store']['document_count']} 文档")
#     # print(f"BM25索引: {system_info['components']['retriever']['bm25_index_size']} 文档")
#     print("="*60)
#
#     return rag_system

def test_rag_system():
    """测试RAG系统"""

    print("测试医疗RAG系统...")

    # 创建测试系统（简化版本）
    rag_system = MedicalRAGSystem()

    # 使用模拟组件初始化
    rag_system.is_initialized = True
    rag_system.system_info["initialized"] = True

    # 测试查询
    test_queries = [
        "糖尿病患者应该注意什么饮食？",
        "高血压怎么治疗？",
        "感冒吃什么药？"
    ]

    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")

        try:
            result = rag_system.query(
                question=query,
                use_streaming=False,
                top_k=3
            )

            print(f"响应: {result['response'][:100]}...")
            print(f"来源数: {result['metadata']['sources_count']}")
            print(f"响应时间: {result['metadata']['response_time']:.2f}s")

        except Exception as e:
            print(f"查询失败: {e}")

    # 系统信息
    print(f"\n📊 系统信息:")
    info = rag_system.get_system_info()
    for key, value in info.items():
        if key != "components":
            print(f"  {key}: {value}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗RAG系统")
    parser.add_argument("--build", action="store_true", help="构建完整管道")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDER, help="嵌入模型")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM, help="LLM模型")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--rebuild", action="store_true", help="重新构建")
    parser.add_argument("--query", type=str, help="测试查询")

    args = parser.parse_args()

    if args.build:
        # 构建完整管道
        rag_system = build_complete_rag_pipeline(
            data_dir=args.data_dir,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            test_mode=args.test,
            rebuild=args.rebuild
        )

        if rag_system and args.query:
            # 测试查询
            print(f"\n🔍 测试查询: {args.query}")
            result = rag_system.query(args.query, use_streaming=False)

            print(f"\n响应: {result['response']}")
            print(f"\n来源数: {result['metadata']['sources_count']}")
            print(f"响应时间: {result['metadata']['response_time']:.2f}s")

            if result['sources']:
                print(f"\n参考来源:")
                for i, source in enumerate(result['sources'][:3]):
                    print(f"{i+1}. {source.get('content', '')[:100]}...")

    elif args.query:
        # 直接测试查询
        test_rag_system()

    else:
        parser.print_help()