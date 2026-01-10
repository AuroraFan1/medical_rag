"""
多轮对话管理模块
支持长上下文和会话状态管理
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from config import SYSTEM_CONFIG

logger = logging.getLogger(__name__)


class ChatSession:
    """聊天会话管理器"""

    def __init__(self,
                 session_id: Optional[str] = None,
                 max_turns: int = SYSTEM_CONFIG['max_conversation_turns'],
                 persist_path: Optional[Path] = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_turns = max_turns
        self.persist_path = persist_path

        # 对话历史
        self.history: List[Dict] = []
        # 会话状态
        self.state: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "turn_count": 0,
            "topics": [],
            "uncertain_answers": 0
        }

        # 加载已有会话
        if persist_path and persist_path.exists():
            self.load_session()

    def add_message(self,
                    role: str,
                    content: str,
                    metadata: Optional[Dict] = None):
        """添加消息到历史"""

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": len(self.history) + 1
        }

        if metadata:
            message["metadata"] = metadata

        self.history.append(message)

        # 更新状态
        self.state["turn_count"] = len(self.history)

        # 限制历史长度
        if len(self.history) > self.max_turns * 2:  # user + assistant
            self.history = self.history[-(self.max_turns * 2):]

        # 自动保存
        if self.persist_path:
            self.save_session()

    def get_recent_history(self,
                           max_turns: int = 5,
                           include_metadata: bool = False) -> List[Dict]:
        """获取最近的对话历史"""

        recent = self.history[-(max_turns * 2):] if len(self.history) > max_turns * 2 else self.history

        if not include_metadata:
            # 移除元数据以节省token
            recent = [
                {k: v for k, v in msg.items() if k not in ['metadata', 'timestamp']}
                for msg in recent
            ]

        return recent

    def get_context_summary(self) -> str:
        """获取上下文摘要"""

        if not self.history:
            return "无对话历史"

        # 提取关键信息
        topics = set()
        for msg in self.history[-10:]:  # 最近10条消息
            content = msg['content'][:100]  # 前100字符
            # 简单的关键词提取（可根据需要增强）
            if "症状" in content:
                topics.add("症状")
            if "治疗" in content:
                topics.add("治疗")
            if "药物" in content:
                topics.add("药物")
            if "检查" in content:
                topics.add("检查")

        summary = f"对话主题: {', '.join(topics) if topics else '未识别特定主题'}\n"
        summary += f"对话轮次: {len(self.history) // 2}\n"

        return summary

    def check_repetition(self, new_message: str, threshold: float = 0.8) -> bool:
        """检查消息是否重复"""

        if not self.history:
            return False

        # 检查与最近消息的相似度
        recent_messages = [msg['content'] for msg in self.history[-3:]]

        for msg in recent_messages:
            # 简单的重叠检查
            overlap = len(set(new_message) & set(msg)) / max(len(set(new_message)), 1)
            if overlap > threshold:
                return True

        return False

    def mark_uncertain_answer(self):
        """标记不确定回答"""
        self.state["uncertain_answers"] += 1

    def should_suggest_clarification(self) -> bool:
        """是否应该要求澄清"""

        # 如果连续多个不确定回答，建议澄清
        recent_uncertain = 0
        for msg in self.history[-4:]:
            if msg.get('metadata', {}).get('uncertain', False):
                recent_uncertain += 1

        return recent_uncertain >= 2

    def get_suggested_clarification(self) -> str:
        """获取澄清建议"""

        # 分析最近的问题
        recent_questions = [
            msg['content'] for msg in self.history[-4:]
            if msg['role'] == 'user'
        ]

        if not recent_questions:
            return "请提供更多详细信息。"

        last_question = recent_questions[-1]

        # 根据问题类型提供澄清建议
        clarification_suggestions = {
            "症状": "请描述症状的持续时间、严重程度和具体表现。",
            "治疗": "请说明已经尝试过的治疗方法和效果。",
            "药物": "请提供药物名称、剂量和使用时间。",
            "检查": "请提供检查结果和医生诊断意见。",
            "一般": "请提供更多详细信息，包括年龄、性别、既往病史等。"
        }

        # 确定问题类型
        question_type = "一般"
        for key in clarification_suggestions:
            if key in last_question:
                question_type = key
                break

        return clarification_suggestions[question_type]

    def save_session(self, path: Optional[Path] = None):
        """保存会话"""

        save_path = path or self.persist_path
        if not save_path:
            logger.warning("未指定保存路径")
            return

        session_data = {
            "session_id": self.session_id,
            "history": self.history,
            "state": self.state,
            "save_time": datetime.now().isoformat()
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        logger.info(f"会话已保存: {save_path}")

    def load_session(self, path: Optional[Path] = None):
        """加载会话"""

        load_path = path or self.persist_path
        if not load_path or not load_path.exists():
            logger.warning("加载路径不存在")
            return

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.session_id = session_data.get("session_id", self.session_id)
            self.history = session_data.get("history", [])
            self.state = session_data.get("state", self.state)

            logger.info(f"会话已加载: {load_path}")

        except Exception as e:
            logger.error(f"加载会话失败: {e}")

    def clear_history(self):
        """清空历史"""
        self.history = []
        self.state["turn_count"] = 0
        self.state["uncertain_answers"] = 0
        self.state["topics"] = []

        logger.info("对话历史已清空")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "history": self.history,
            "state": self.state,
            "history_length": len(self.history)
        }