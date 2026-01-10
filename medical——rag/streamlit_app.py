"""
Streamlit医疗RAG应用
优化版，支持多轮对话和引用显示
"""

import streamlit as st
import time
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# 导入配置
from config import (
    APP_TITLE, APP_ICON, EXAMPLE_QUESTIONS, DISCLAIMER_TEXT,
    MODEL_CONFIGS, EMBEDDING_MODELS, SYSTEM_CONFIG, PROMPT_TEMPLATES
)

# 页面配置
st.set_page_config(
    page_title="🏥 MedAssist - 医疗智能问答系统",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
.stChatMessage {
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.user-message {
    background-color: #f0f8ff;
    border-left-color: #2196F3;
}
.assistant-message {
    background-color: #f9f9f9;
    border-left-color: #4CAF50;
}
.warning-box {
    background-color: #fff3cd;
    border: 2px solid #ffeaa7;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    color: #856404;
    font-size: 0.95rem;
}
.citation-badge {
    display: inline-block;
    background-color: #e3f2fd;
    color: #1565c0;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.source-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system(_config):
    """初始化RAG系统（缓存）"""
    try:
        from medical_rag import MedicalRAGSystem

        rag_system = MedicalRAGSystem(_config)

        # 从session state获取配置
        embedding_model = st.session_state.get("embedding_model", "bge-large-zh")
        llm_model = st.session_state.get("llm_model", "qwen2.5-1.5b")
        vector_store_name = st.session_state.get("vector_store_name", "medical_cases_v1")

        success = rag_system.initialize(
            embedding_model=embedding_model,
            llm_model=llm_model,
            use_api=False,
            vector_store_name=vector_store_name,
            enable_hybrid=True,
            enable_rerank=True
        )

        if success:
            return rag_system
        else:
            st.error("系统初始化失败")
            return None

    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def display_message(role: str, content: str, metadata: dict = None):
    """显示消息"""

    if role == "user":
        st.markdown(f'''
        <div class="stChatMessage user-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="background-color: #2196F3; color: white; width: 32px; height: 32px; border-radius: 50%; 
                          display: flex; align-items: center; justify-content: center; margin-right: 0.5rem;">
                    👤
                </div>
                <strong style="font-size: 1.1rem;">用户</strong>
            </div>
            <div style="font-size: 1rem; line-height: 1.5;">{content}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        # 提取引用标记
        citations = []
        if metadata and metadata.get("sources"):
            for i, source in enumerate(metadata["sources"][:5]):
                citations.append(f'<span class="citation-badge">[来源{i+1}]</span>')

        citation_html = "".join(citations) if citations else ""

        st.markdown(f'''
        <div class="stChatMessage assistant-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="background-color: #4CAF50; color: white; width: 32px; height: 32px; border-radius: 50%; 
                          display: flex; align-items: center; justify-content: center; margin-right: 0.5rem;">
                    🤖
                </div>
                <strong style="font-size: 1.1rem;">MedAssist助手</strong>
            </div>
            <div style="font-size: 1rem; line-height: 1.5; margin-bottom: 1rem;">{content}</div>
            {citation_html}
        </div>
        ''', unsafe_allow_html=True)

        # 显示元数据（可展开）
        if metadata:
            with st.expander("📊 详细信息", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    response_time = metadata.get("response_time", 0)
                    st.metric("⏱️ 响应时间", f"{response_time:.2f}s")

                with col2:
                    sources_count = metadata.get("sources_count", 0)
                    st.metric("📚 参考病例", sources_count)

                with col3:
                    if metadata.get("is_uncertain"):
                        st.metric("⚠️ 确定性", "低")
                    else:
                        st.metric("✅ 确定性", "高")

                # 显示警告（如果不确定）
                if metadata.get("is_uncertain"):
                    st.warning("⚠️ 此回答基于有限信息，建议咨询专业医生获取准确诊断")

                # 显示来源详情
                if metadata.get("sources") and st.checkbox("显示参考病例详情", value=False):
                    st.subheader("📋 参考病例")

                    for i, source in enumerate(metadata["sources"][:3]):
                        with st.container():
                            st.markdown(f'<div class="source-card">', unsafe_allow_html=True)

                            source_metadata = source.get("metadata", {})
                            disease = source_metadata.get("disease", "未知疾病")
                            hospital = source_metadata.get("hospital", "未知医院")
                            year = source_metadata.get("year", "未知年份")

                            st.markdown(f"**病例 {i+1}**")
                            st.markdown(f"**疾病**: {disease}")
                            st.markdown(f"**医院**: {hospital}")
                            st.markdown(f"**年份**: {year}")

                            content = source.get("content", "")
                            if len(content) > 300:
                                content = content[:300] + "..."
                            st.markdown(f"**内容**: {content}")

                            st.markdown('</div>', unsafe_allow_html=True)

def main():
    """主函数"""

    st.title("🏥 MedAssist - 医疗智能问答系统")
    st.markdown("基于120万真实医疗病例的智能问答助手")
    st.markdown("---")

    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "conversation_stats" not in st.session_state:
        st.session_state.conversation_stats = {
            "total_queries": 0,
            "avg_response_time": 0,
            "total_sources": 0
        }

    # ====== 侧边栏 ======
    with st.sidebar:
        st.header("⚙️ 系统控制面板")

        # 模型配置
        st.subheader("🤖 模型配置")

        # 嵌入模型选择
        embedding_options = list(EMBEDDING_MODELS.keys())
        embedding_model = st.selectbox(
            "嵌入模型",
            embedding_options,
            index=embedding_options.index("bge-large-zh") if "bge-large-zh" in embedding_options else 0,
            help="用于病例嵌入和检索的模型"
        )

        # LLM模型选择
        llm_options = list(MODEL_CONFIGS.keys())
        llm_model = st.selectbox(
            "LLM模型",
            llm_options,
            index=llm_options.index("qwen2.5-7b") if "qwen2.5-7b" in llm_options else 0,
            help="用于生成回答的语言模型"
        )

        # 向量数据库名称
        vector_store_name = st.text_input(
            "向量数据库名称",
            value="medical_cases_v1",
            help="向量数据库的存储名称"
        )

        # 保存配置到session state
        st.session_state.embedding_model = embedding_model
        st.session_state.llm_model = llm_model
        st.session_state.vector_store_name = vector_store_name

        # 检索配置
        st.subheader("🔍 检索配置")

        top_k = st.slider("检索病例数量", 1, 20, 5,
                         help="每次检索返回的相似病例数量")

        score_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05,
                                  help="病例相似度最低阈值")

        col1, col2, col3 = st.columns(3)
        with col1:
            use_hybrid = st.checkbox("混合搜索", value=True,
                                   help="结合语义和关键词搜索")
        with col2:
            use_rerank = st.checkbox("结果重排序", value=True,
                                   help="使用重排序模型优化结果")
        with col3:
            use_diversity = st.checkbox("结果去重", value=True,
                                      help="确保检索结果的多样性")

        # 系统初始化按钮
        st.markdown("---")

        if st.button("🔄 初始化/更新系统", type="primary", use_container_width=True):
            with st.spinner("正在初始化系统..."):
                # 清除缓存并重新初始化
                st.cache_resource.clear()

                rag_system = initialize_rag_system({})
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.success("✅ 系统初始化成功")

                    # 更新检索策略
                    rag_system.update_retrieval_strategy(
                        enable_hybrid=use_hybrid,
                        enable_rerank=use_rerank,
                        enable_diversity=use_diversity
                    )
                else:
                    st.error("❌ 系统初始化失败")

        st.markdown("---")

        # 系统状态
        if st.session_state.rag_system:
            with st.expander("📊 系统状态", expanded=False):
                system_info = st.session_state.rag_system.get_system_info()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("嵌入模型", embedding_model)
                    st.metric("LLM模型", llm_model)
                with col2:
                    doc_count = system_info.get("components", {}).get("vector_store", {}).get("document_count", 0)
                    st.metric("病例数量", f"{doc_count:,}")
                    st.metric("查询次数", st.session_state.conversation_stats["total_queries"])

                if st.checkbox("显示详细配置"):
                    st.json(system_info)

        st.markdown("---")

        # 示例问题
        st.subheader("💡 常见问题示例")

        for question in EXAMPLE_QUESTIONS:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                st.session_state.user_input = question
                st.rerun()

        st.markdown("---")

        # 免责声明
        st.markdown(f'<div class="warning-box">{DISCLAIMER_TEXT}</div>',
                   unsafe_allow_html=True)

        st.markdown("---")

        # 操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_conversation(st.session_state.session_id)
                st.session_state.conversation_stats = {
                    "total_queries": 0,
                    "avg_response_time": 0,
                    "total_sources": 0
                }
                st.rerun()

        with col2:
            if st.button("💾 导出对话", use_container_width=True):
                if st.session_state.rag_system:
                    conversation = st.session_state.rag_system.export_conversation(st.session_state.session_id)

                    # 提供下载
                    json_str = json.dumps(conversation, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="下载JSON",
                        data=json_str,
                        file_name=f"conversation_{st.session_state.session_id}.json",
                        mime="application/json"
                    )

        # 对话统计
        st.markdown("---")
        st.subheader("📈 对话统计")

        stats = st.session_state.conversation_stats
        st.metric("总查询数", stats["total_queries"])
        st.metric("平均响应时间", f"{stats['avg_response_time']:.2f}s")
        st.metric("总参考病例", stats["total_sources"])

    # ====== 主界面 ======

    # 初始化系统（如果尚未初始化）
    if st.session_state.rag_system is None:
        st.info("⚠️ 系统尚未初始化，请在侧边栏配置并初始化系统")

        # 显示系统要求
        with st.expander("📋 系统要求", expanded=True):
            st.markdown("""
            **系统配置要求：**
            
            1. **数据准备**：
               - 医疗文本文件（.txt格式）
               - 按年份命名的文件，如：2010.txt, 2011.txt等
            
            2. **模型准备**：
               - 嵌入模型：BAAI/bge-large-zh-v1.5
               - LLM模型：Qwen/Qwen2.5-7B-Instruct
            
            3. **硬件要求**：
               - GPU：RTX 5090 32GB（推荐）
               - 内存：32GB以上
               - 存储：100GB以上可用空间
            
            **首次运行步骤：**
            1. 在侧边栏配置模型参数
            2. 点击"初始化/更新系统"按钮
            3. 系统将自动处理数据并构建索引
            """)

        st.stop()

    # 显示欢迎消息
    if len(st.session_state.messages) == 0:
        welcome_message = """
        <div class="stChatMessage assistant-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="background-color: #4CAF50; color: white; width: 40px; height: 40px; border-radius: 50%; 
                          display: flex; align-items: center; justify-content: center; margin-right: 0.5rem;">
                    🏥
                </div>
                <div>
                    <strong style="font-size: 1.2rem;">MedAssist医疗助手</strong><br>
                    <span style="font-size: 0.9rem; color: #666;">基于120万真实医疗病例</span>
                </div>
            </div>
            <div style="font-size: 1rem; line-height: 1.6; margin: 1rem 0;">
                您好！我是MedAssist医疗智能助手，基于120万真实医疗病例为您提供参考信息。
                
                **我能为您提供：**
                - 疾病症状咨询
                - 治疗方案参考
                - 医疗建议分析
                - 病例相似性检索
                
                **使用提示：**
                1. 描述具体的症状或问题
                2. 提供相关病史信息
                3. 系统会检索相似病例并提供参考建议
                
                ⚠️ **重要提醒**：以下信息基于历史病例数据，仅供参考，不能替代专业医疗诊断。
            </div>
        </div>
        """

        st.markdown(welcome_message, unsafe_allow_html=True)

        # 添加到消息历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": "您好！我是MedAssist医疗智能助手，基于120万真实医疗病例为您提供参考信息。",
            "metadata": {"is_welcome": True}
        })

    # 显示对话历史
    for message in st.session_state.messages:
        if not message.get("metadata", {}).get("is_welcome", False):
            display_message(
                message["role"],
                message["content"],
                message.get("metadata")
            )

    # 处理用户输入
    user_input = ""

    # 检查预设问题
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input

    if not user_input:
        # 聊天输入框
        user_input = st.chat_input("请输入您的医疗问题...", key="chat_input")

    if user_input and st.session_state.rag_system:
        # 显示用户消息
        display_message("user", user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # 生成助手回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 显示思考动画
            thinking_html = """
            <div style='text-align: center; padding: 2rem;'>
                <div style='display: inline-block; width: 40px; height: 40px; border: 3px solid #f3f3f3; 
                          border-top: 3px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite;'></div>
                <p style='margin-top: 1rem; color: #666; font-size: 0.9rem;'>正在检索医疗病例并生成回答...</p>
            </div>
            <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            </style>
            """
            message_placeholder.markdown(thinking_html, unsafe_allow_html=True)

            try:
                # 处理查询
                start_time = time.time()

                result = st.session_state.rag_system.query(
                    question=user_input,
                    use_streaming=False,  # 暂时禁用流式，简化处理
                    top_k=top_k,
                    session_id=st.session_state.session_id
                )

                response_time = time.time() - start_time

                # 显示结果
                display_message(
                    "assistant",
                    result["response"],
                    result["metadata"]
                )

                # 更新对话统计
                stats = st.session_state.conversation_stats
                stats["total_queries"] += 1
                stats["total_sources"] += result["metadata"]["sources_count"]

                # 更新平均响应时间
                if stats["total_queries"] == 1:
                    stats["avg_response_time"] = response_time
                else:
                    stats["avg_response_time"] = (
                        (stats["avg_response_time"] * (stats["total_queries"] - 1) + response_time)
                        / stats["total_queries"]
                    )

                # 添加到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": result["metadata"]
                })

                # 更新侧边栏统计
                st.rerun()

            except Exception as e:
                st.error(f"处理查询时出错: {str(e)}")

                error_message = f"抱歉，处理您的查询时出现错误。请稍后重试或联系管理员。\n\n错误详情: {str(e)}"
                display_message("assistant", error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

    # ====== 页脚 ======
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**版本信息**")
        st.markdown("MedAssist v2.0")
        st.markdown("基于Qwen2.5-7B")

    with col2:
        st.markdown("**数据来源**")
        st.markdown("MedDialog数据库")
        st.markdown("120万医疗病例")

    with col3:
        st.markdown("**技术支持**")
        st.markdown("AI医疗实验室")
        st.markdown("📧 contact@medical-ai.com")

    st.markdown("---")

    # 紧急提醒
    st.markdown("""
    <div style='text-align: center; color: #d32f2f; padding: 1rem; background-color: #ffebee; border-radius: 8px;'>
        <strong>🚨 紧急情况提醒</strong><br>
        如有紧急医疗情况，请立即就医或拨打急救电话 120
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()