"""
Streamlit Web界面
"""

import streamlit as st
import time
import logging
from typing import Optional

from config import APP_TITLE, APP_ICON, EXAMPLE_QUESTIONS, DISCLAIMER_TEXT, EMBEDDING_MODEL
from rag_system import get_rag_system

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedAssistUI:
    """MedAssist用户界面类"""

    def __init__(self):
        self.rag_system = None
        self.setup_page()

    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # 自定义CSS
        st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def setup_sidebar(self):
        """设置侧边栏"""
        with st.sidebar:
            st.title("⚙️ 系统控制面板")

            # 免责声明
            st.markdown(f'<div class="warning-box">{DISCLAIMER_TEXT}</div>',
                       unsafe_allow_html=True)

            # 系统初始化
            with st.spinner("正在初始化医疗知识库..."):
                self.rag_system = get_rag_system()

            if self.rag_system:
                st.success("✅ 医疗知识库已加载")

                # 系统信息
                with st.expander("📊 系统信息", expanded=False):
                    system_info = self.rag_system.get_system_info()
                    for key, value in system_info.items():
                        st.text(f"{key}: {value}")
                    st.text(f"嵌入模型：{EMBEDDING_MODEL}")

                # 搜索设置
                st.subheader("🔍 搜索设置")
                search_k = st.slider("检索结果数量", 1, 10, 5,
                                   help="每次搜索返回的相似病例数量")

                if 'search_k' not in st.session_state:
                    st.session_state.search_k = search_k
                elif st.session_state.search_k != search_k:
                    st.session_state.search_k = search_k
                    st.rerun()

            else:
                st.error("❌ 系统初始化失败")
                st.info("请先运行 `python data_processing.py` 构建向量数据库")
                st.stop()

            st.markdown("---")

            # 快速问题示例
            st.subheader("💡 常见问题示例")
            for i, question in enumerate(EXAMPLE_QUESTIONS):
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    if "user_input" not in st.session_state:
                        st.session_state.user_input = question
                    else:
                        st.session_state.user_input = question
                    st.rerun()

            st.markdown("---")

            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ 清除对话", type="secondary", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

            with col2:
                if st.button("🔄 重新初始化", type="secondary", use_container_width=True):
                    get_rag_system(force_reinitialize=True)
                    st.success("系统已重新初始化")
                    st.rerun()

    def display_chat_history(self):
        """显示对话历史"""
        # 初始化消息历史
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "您好！我是MedAssist医疗助手。我可以基于真实的医疗病例数据为您提供参考信息。请问您有什么医疗健康方面的疑问？"
                }
            ]

        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def process_user_query(self, query: str):
        """处理用户查询"""
        if not query or not query.strip():
            return

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(query)

        # 添加到历史
        st.session_state.messages.append({"role": "user", "content": query})

        # 生成AI回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 显示正在思考
            message_placeholder.markdown("🤔 正在检索相关病例并生成回答...")

            try:
                start_time = time.time()

                # 检索相似病例
                similar_cases = self.rag_system.search_similar_cases(
                    query,
                    k=st.session_state.get("search_k", 5)
                )

                # 生成回答
                response = ""
                for chunk in self.rag_system.query(query, use_streaming=True):
                    response += chunk
                    message_placeholder.markdown(response + "▌")

                full_response = response

                # 计算响应时间
                response_time = time.time() - start_time

                # 更新显示
                message_placeholder.markdown(full_response)

                # 显示参考病例
                if similar_cases:
                    with st.expander("📚 参考病例（点击展开）", expanded=False):
                        for i, case in enumerate(similar_cases[:3]):  # 只显示前3个
                            disease = case["metadata"].get("disease", "未知疾病")
                            hospital = case["metadata"].get("hospital", "未知医院")
                            year = case["metadata"].get("year", "未知年份")

                            st.markdown(f"**病例 {i+1}**")
                            st.markdown(f"**疾病**: {disease}")
                            st.markdown(f"**医院**: {hospital} ({year}年)")
                            st.markdown(f"**相关内容**: {case['content']}")
                            st.markdown("---")

                # 显示统计信息
                st.caption(f"⏱️ 响应时间: {response_time:.2f}秒 | "
                          f"🔍 检索病例: {len(similar_cases)}个")

            except Exception as e:
                error_msg = "抱歉，系统在处理您的请求时遇到问题。请稍后重试。"
                logger.error(f"处理查询时出错: {e}")
                message_placeholder.error(error_msg)
                full_response = error_msg

        # 添加到历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    def display_footer(self):
        """显示页脚"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center; color: #666;">
                <p>🏥 <strong>MedAssist v2.0</strong> | 基于真实医疗病例的智能问答系统</p>
                <p><small>⚠️ 本系统信息仅供参考，不能替代专业医疗诊断</small></p>
                <p><small>🚨 如有紧急情况，请立即就医或拨打急救电话 120</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )

    def run(self):
        """运行应用"""
        # 设置侧边栏
        self.setup_sidebar()

        # 主界面标题
        st.title(APP_TITLE)
        st.markdown("---")

        # 简介
        st.info("""
        💡 **系统特点**：
        - 基于2011-2020年真实医疗病例数据
        - 智能检索相似病例提供参考
        - 结合医学知识生成专业回答
        - **重要提醒**：所有回答仅供参考，不能替代专业医疗建议
        """)

        # 显示对话历史
        self.display_chat_history()

        # 处理用户输入
        user_input = ""

        # 检查是否有预设问题
        if "user_input" in st.session_state:
            user_input = st.session_state.user_input
            del st.session_state.user_input

        # 聊天输入框
        if user_input:
            # 如果有预设问题，直接处理
            self.process_user_query(user_input)
        else:
            # 否则显示输入框
            if prompt := st.chat_input("请输入您的医疗问题..."):
                self.process_user_query(prompt)

        # 显示页脚
        self.display_footer()

def main():
    """主函数"""
    app = MedAssistUI()
    app.run()

if __name__ == "__main__":
    main()