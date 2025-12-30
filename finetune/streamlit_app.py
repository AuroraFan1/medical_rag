"""
修复的Streamlit应用
"""

import streamlit as st
import time
import os
import sys
from pathlib import Path
import json

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 页面设置
from config import APP_TITLE, APP_ICON, EXAMPLE_QUESTIONS, DISCLAIMER_TEXT, MODEL_NAME

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
    border-radius: 10px;
    margin-bottom: 1rem;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)


# @st.cache_resource
# def initialize_rag_system():
#     """初始化RAG系统"""
#     from rag_system import EnhancedMedicalRAGSystem
#     from config import VECTOR_STORE_PATH
#
#     # 确保使用绝对路径
#     vector_db_path = Path(VECTOR_STORE_PATH)
#     if not vector_db_path.is_absolute():
#         vector_db_path = current_dir / vector_db_path
#
#     st.info(f"向量数据库路径: {vector_db_path}")
#
#     if not vector_db_path.exists():
#         st.error(f"向量数据库不存在: {vector_db_path}")
#         st.info("请先运行 data_processing.py 构建向量数据库")
#         return None, "向量数据库不存在"
#
#     try:
#         rag_system = EnhancedMedicalRAGSystem(str(vector_db_path))
#         success = rag_system.initialize()
#
#         if success:
#             return rag_system, "✅ 医疗知识库已加载"
#         else:
#             return None, "❌ 系统初始化失败"
#
#     except Exception as e:
#         return None, f"❌ 初始化失败: {str(e)}"


"""
修改后的Streamlit应用，支持微调模型
"""
# 修改initialize_rag_system函数
@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统，支持基础模型和微调模型"""
    from config import VECTOR_STORE_PATH
    import os

    import sys

    # 添加项目根目录到系统路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from rag_system import EnhancedMedicalRAGSystem


    # 模型选择
    model_mode = st.sidebar.selectbox(
        "选择模型模式",
        ["openai", "local"],
        index=0,
        help="openai: 使用API模型\nlocal: 使用本地模型"
    )

    # 检查微调模型是否存在
    finetuned_model_path = "./models/finetuned_medical"
    use_finetuned = False

    if model_mode == "local" and os.path.exists(finetuned_model_path):
        use_finetuned = st.sidebar.checkbox(
            "使用微调模型",
            value=True,
            help="使用经过医疗数据微调的模型"
        )

    # 确保使用绝对路径
    vector_db_path = Path(VECTOR_STORE_PATH)
    if not vector_db_path.is_absolute():
        vector_db_path = current_dir / vector_db_path

    if not vector_db_path.exists():
        st.error(f"向量数据库不存在: {vector_db_path}")
        st.info("请先运行 data_processing.py 构建向量数据库")
        return None, "向量数据库不存在", None

    try:
        # 初始化RAG系统
        rag_system = EnhancedMedicalRAGSystem(
            model_mode=model_mode,
            local_model_path=finetuned_model_path if use_finetuned else None,
            use_finetuned=use_finetuned,
            vector_store_path=str(vector_db_path)
        )

        success = rag_system.initialize()

        if success:
            model_info = f"{model_mode}模型"
            if use_finetuned:
                model_info += " (微调版)"
            return rag_system, f"✅ {model_info}已加载", model_info
        else:
            return None, "❌ 系统初始化失败", None

    except Exception as e:
        return None, f"❌ 初始化失败: {str(e)}", None


# 修改主函数中的模型信息显示
def main():
    """主函数"""
    st.title(APP_TITLE)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 系统控制面板")

        # 模型选择
        with st.spinner("正在初始化系统..."):
            rag_system, init_message, model_info = initialize_rag_system()

        if rag_system:
            st.success(init_message)

            # 显示模型信息
            st.info(f"当前模型: {model_info}")

            # 系统信息
            with st.expander("📊 系统信息", expanded=False):
                system_info = rag_system.get_system_info()
                for key, value in system_info.items():
                    st.text(f"{key}: {value}")

                # 显示评估结果（如果存在）
                eval_file = "./evaluation_results/evaluation_metrics.json"
                if os.path.exists(eval_file):
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        st.subheader("📈 评估指标")

                        if system_info.get('use_finetuned'):
                            metrics = eval_data.get('finetuned_model', {})
                        else:
                            metrics = eval_data.get('base_model', {})

                        if metrics.get('generation'):
                            st.metric("BLEU分数", f"{metrics['generation']['avg_bleu']:.3f}")
                            st.metric("ROUGE分数", f"{metrics['generation']['avg_rouge']:.3f}")

                    except:
                        pass

            # ... 其余侧边栏代码 ...

        # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "您好！我是MedAssist医疗助手。基于120万真实医疗病例为您提供参考信息。\n\n⚠️ **重要提醒**：以下信息仅供参考，不能替代专业医疗诊断。"
            }
        ]

        # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # 处理用户输入
    user_input = ""

    # 检查是否有预设问题
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input

    if user_input:
        # 处理预设问题
        prompt = user_input
    else:
        # 聊天输入
        prompt = st.chat_input("请输入您的医疗问题...")

    if prompt and rag_system:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 添加到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 生成助手回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # 显示思考状态
                message_placeholder.markdown("🤔 正在检索相关病例...")

                start_time = time.time()

                # 检索相似病例
                similar_cases = rag_system.search_similar_cases(
                    prompt,
                    k=st.session_state.get("search_k", 5)
                )

                # 生成回答
                message_placeholder.markdown("💭 正在生成回答...")

                # 使用流式输出
                response = ""
                for chunk in rag_system.query(prompt, use_streaming=True):
                    response += chunk
                    message_placeholder.markdown(response + "▌")
                    time.sleep(0.01)

                full_response = response

                # 显示最终回答
                message_placeholder.markdown(full_response)

                # 显示相似病例（可展开）
                if similar_cases:
                    with st.expander(f"📚 参考病例 ({len(similar_cases)}个)", expanded=False):
                        for i, case in enumerate(similar_cases):
                            disease = case["metadata"].get("disease", "未知疾病")
                            hospital = case["metadata"].get("hospital", "未知医院")
                            year = case["metadata"].get("year", "未知年份")

                            st.markdown(f"**病例 {i + 1}**")
                            if disease:
                                st.markdown(f"**疾病**: {disease}")
                            if hospital:
                                st.markdown(f"**医院**: {hospital}")
                            if year:
                                st.markdown(f"**年份**: {year}")

                            content = case["content"]
                            if len(content) > 200:
                                content = content[:200] + "..."
                            st.markdown(f"**内容**: {content}")
                            st.markdown("---")

                # 显示统计信息
                response_time = time.time() - start_time
                st.caption(f"⏱️ 响应时间: {response_time:.2f}秒 | 🔍 检索病例: {len(similar_cases)}个")

            except Exception as e:
                st.error(f"处理查询时出错: {e}")
                full_response = f"抱歉，系统遇到问题: {str(e)}"
                message_placeholder.markdown(full_response)

        # 添加到历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🏥 <strong>MedAssist v2.0</strong> | 基于1,202,613个医疗病例文档块</p>
        <p><small>⚠️ 本系统信息仅供参考，不能替代专业医疗诊断</small></p>
        <p><small>🚨 如有紧急情况，请立即就医或拨打急救电话 120</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )



# def main():
#     """主函数"""
#     st.title(APP_TITLE)
#     st.markdown("---")
#
#     # 侧边栏
#     with st.sidebar:
#         st.header("⚙️ 系统控制面板")
#
#         # 初始化RAG系统
#         with st.spinner("正在初始化系统..."):
#             rag_system, init_message = initialize_rag_system()
#
#         if rag_system:
#             st.success(init_message)
#
#             # 系统信息
#             with st.expander("📊 系统信息", expanded=False):
#                 system_info = rag_system.get_system_info()
#                 for key, value in system_info.items():
#                     st.text(f"{key}: {value}")
#
#             # 检索设置
#             st.subheader("🔍 检索设置")
#             search_k = st.slider("检索结果数量", 1, 10, 5)
#             st.session_state.search_k = search_k
#
#         else:
#             st.error(init_message)
#
#             # 提供修复选项
#             if st.button("🔄 尝试重新初始化"):
#                 st.cache_resource.clear()
#                 st.rerun()
#
#             st.stop()
#
#         # 示例问题
#         st.subheader("💡 常见问题示例")
#         for i, question in enumerate(EXAMPLE_QUESTIONS):
#             if st.button(question, key=f"example_{i}", use_container_width=True):
#                 st.session_state.user_input = question
#                 st.rerun()
#
#         st.markdown("---")
#
#         # 免责声明
#         st.markdown(f'<div class="warning-box">{DISCLAIMER_TEXT}</div>',
#                    unsafe_allow_html=True)
#
#         st.markdown("---")
#
#         # 操作按钮
#         if st.button("🗑️ 清除对话历史", use_container_width=True):
#             if "messages" in st.session_state:
#                 st.session_state.messages = []
#             st.rerun()
#
#     # 初始化聊天历史
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {
#                 "role": "assistant",
#                 "content": "您好！我是MedAssist医疗助手。基于120万真实医疗病例为您提供参考信息。\n\n⚠️ **重要提醒**：以下信息仅供参考，不能替代专业医疗诊断。"
#             }
#         ]
#
#     # 显示聊天历史
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     # 处理用户输入
#     user_input = ""
#
#     # 检查是否有预设问题
#     if "user_input" in st.session_state:
#         user_input = st.session_state.user_input
#         del st.session_state.user_input
#
#     if user_input:
#         # 处理预设问题
#         prompt = user_input
#     else:
#         # 聊天输入
#         prompt = st.chat_input("请输入您的医疗问题...")
#
#     if prompt and rag_system:
#         # 显示用户消息
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         # 添加到历史
#         st.session_state.messages.append({"role": "user", "content": prompt})
#
#         # 生成助手回复
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""
#
#             try:
#                 # 显示思考状态
#                 message_placeholder.markdown("🤔 正在检索相关病例...")
#
#                 start_time = time.time()
#
#                 # 检索相似病例
#                 similar_cases = rag_system.search_similar_cases(
#                     prompt,
#                     k=st.session_state.get("search_k", 5)
#                 )
#
#                 # 生成回答
#                 message_placeholder.markdown("💭 正在生成回答...")
#
#                 # 使用流式输出
#                 response = ""
#                 for chunk in rag_system.query(prompt, use_streaming=True):
#                     response += chunk
#                     message_placeholder.markdown(response + "▌")
#                     time.sleep(0.01)
#
#                 full_response = response
#
#                 # 显示最终回答
#                 message_placeholder.markdown(full_response)
#
#                 # 显示相似病例（可展开）
#                 if similar_cases:
#                     with st.expander(f"📚 参考病例 ({len(similar_cases)}个)", expanded=False):
#                         for i, case in enumerate(similar_cases):
#                             disease = case["metadata"].get("disease", "未知疾病")
#                             hospital = case["metadata"].get("hospital", "未知医院")
#                             year = case["metadata"].get("year", "未知年份")
#
#                             st.markdown(f"**病例 {i+1}**")
#                             if disease:
#                                 st.markdown(f"**疾病**: {disease}")
#                             if hospital:
#                                 st.markdown(f"**医院**: {hospital}")
#                             if year:
#                                 st.markdown(f"**年份**: {year}")
#
#                             content = case["content"]
#                             if len(content) > 200:
#                                 content = content[:200] + "..."
#                             st.markdown(f"**内容**: {content}")
#                             st.markdown("---")
#
#                 # 显示统计信息
#                 response_time = time.time() - start_time
#                 st.caption(f"⏱️ 响应时间: {response_time:.2f}秒 | 🔍 检索病例: {len(similar_cases)}个")
#
#             except Exception as e:
#                 st.error(f"处理查询时出错: {e}")
#                 full_response = f"抱歉，系统遇到问题: {str(e)}"
#                 message_placeholder.markdown(full_response)
#
#         # 添加到历史
#         st.session_state.messages.append({"role": "assistant", "content": full_response})
#
#     # 页脚
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style='text-align: center; color: #666;'>
#         <p>🏥 <strong>MedAssist v2.0</strong> | 基于1,202,613个医疗病例文档块</p>
#         <p><small>⚠️ 本系统信息仅供参考，不能替代专业医疗诊断</small></p>
#         <p><small>🚨 如有紧急情况，请立即就医或拨打急救电话 120</small></p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


if __name__ == "__main__":
    main()