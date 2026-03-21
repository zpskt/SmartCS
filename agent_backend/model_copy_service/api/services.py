#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：services.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/22 00:14 
@Description： 
'''
import logging
import uuid
from typing import Dict, Optional
from datetime import datetime

from agents.model_copy_agent import get_model_copy_agent

logger = logging.getLogger(__name__)

# 内存存储会话映射（生产环境建议用 Redis）
# 格式: {session_id: agent_instance}
_session_store: Dict[str, object] = {}

# 内存存储会话历史（生产环境建议用数据库）
# 格式: {session_id: [{"role": "user/assistant", "content": "...", "timestamp": "..."}]}
_history_store: Dict[str, list] = {}


class ModelCopyService:
    """型号复制服务类"""

    def __init__(self):
        self.agent = get_model_copy_agent()

    def create_session(self, user_id: str = None) -> str:
        """
        创建新会话

        :param user_id: 用户ID（可选）
        :return: 会话ID
        """
        session_id = str(uuid.uuid4())
        # 创建新的智能体实例（每个会话独立）
        from agents.model_copy_agent import ModelCopyAgent
        agent = ModelCopyAgent()
        agent.set_session_id(session_id)

        _session_store[session_id] = agent
        _history_store[session_id] = []

        logger.info(f"创建会话: {session_id}, 用户: {user_id}")
        return session_id

    def chat(self, message: str, session_id: str = None, user_id: str = None) -> dict:
        """
        对话处理

        :param message: 用户消息
        :param session_id: 会话ID
        :param user_id: 用户ID
        :return: 响应结果
        """
        # 如果没有 session_id，创建新会话
        if not session_id:
            session_id = self.create_session(user_id)

        # 获取或创建 Agent 实例
        agent = _session_store.get(session_id)
        if not agent:
            # 会话不存在，创建新会话
            session_id = self.create_session(user_id)
            agent = _session_store[session_id]

        # 保存用户消息到历史
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        _history_store[session_id].append(user_message)

        # 调用智能体
        try:
            logger.info(f"处理消息: session={session_id}, message={message[:50]}...")
            response_content = agent.invoke(message)

            # 保存 AI 响应到历史
            ai_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            }
            _history_store[session_id].append(ai_message)

            return {
                "success": True,
                "session_id": session_id,
                "message": response_content
            }

        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            return {
                "success": False,
                "session_id": session_id,
                "message": "",
                "error": str(e)
            }

    def get_history(self, session_id: str) -> dict:
        """
        获取会话历史

        :param session_id: 会话ID
        :return: 历史记录
        """
        messages = _history_store.get(session_id, [])
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        :param session_id: 会话ID
        :return: 是否删除成功
        """
        if session_id in _session_store:
            del _session_store[session_id]
        if session_id in _history_store:
            del _history_store[session_id]
        logger.info(f"删除会话: {session_id}")
        return True

    def list_sessions(self, limit: int = 20) -> list:
        """
        列出最近的会话

        :param limit: 限制数量
        :return: 会话列表
        """
        sessions = []
        for session_id in list(_session_store.keys())[-limit:]:
            messages = _history_store.get(session_id, [])
            last_message = messages[-1]["content"][:50] if messages else ""
            sessions.append({
                "session_id": session_id,
                "message_count": len(messages),
                "last_message": last_message,
                "last_updated": messages[-1]["timestamp"] if messages else None
            })
        return sessions


# 单例服务实例
_service_instance = None


def get_model_copy_service() -> ModelCopyService:
    """获取服务单例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelCopyService()
    return _service_instance