"""
会话管理服务
管理用户会话的创建、切换、删除等
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from models.schemas import Session, Message


class SessionManager:
    """会话管理器类"""
    
    def __init__(self):
        """初始化会话管理器"""
        # 内存存储（实际应该用数据库）
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
    
    def create_session(
        self,
        user_id: str,
        title: str = "新会话",
        metadata: Optional[Dict] = None
    ) -> Session:
        """
        创建新会话
        
        :param user_id: 用户 ID
        :param title: 会话标题
        :param metadata: 额外元数据
        :return: Session 对象
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            user_id=user_id,
            title=title,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        
        # 关联到用户
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话
        
        :param session_id: 会话 ID
        :return: Session 对象
        """
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        获取用户的所有会话
        
        :param user_id: 用户 ID
        :return: 会话列表
        """
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        添加消息到会话
        
        :param session_id: 会话 ID
        :param role: 角色（user/assistant/system）
        :param content: 消息内容
        :param metadata: 额外元数据
        :return: Message 对象
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话 {session_id} 不存在")
        
        message = Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        session.messages.append(message.dict())
        session.updated_at = datetime.now()
        
        # 自动更新会话标题（如果是第一条消息）
        if len(session.messages) == 1 and role == "user":
            session.title = content[:50] + "..." if len(content) > 50 else content
        
        return message
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        :param session_id: 会话 ID
        :return: 是否成功删除
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # 从用户会话列表中移除
        user_id = session.user_id
        if user_id in self.user_sessions:
            self.user_sessions[user_id].remove(session_id)
        
        # 删除会话
        del self.sessions[session_id]
        return True
    
    def clear_session_messages(self, session_id: str) -> bool:
        """
        清空会话消息
        
        :param session_id: 会话 ID
        :return: 是否成功清空
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.messages = []
        session.updated_at = datetime.now()
        return True
    
    def export_session(self, session_id: str, format: str = "json") -> str:
        """
        导出会话
        
        :param session_id: 会话 ID
        :param format: 导出格式（json/txt）
        :return: 导出的字符串
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话 {session_id} 不存在")
        
        if format == "json":
            import json
            return json.dumps(session.dict(), ensure_ascii=False, indent=2)
        elif format == "txt":
            lines = [f"=== 会话：{session.title} ==="]
            for msg in session.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            return "\n".join(lines)
        else:
            raise ValueError(f"不支持的导出格式：{format}")
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取会话历史（用于 RAG 上下文）
        
        :param session_id: 会话 ID
        :return: 历史消息列表
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in session.messages[-10:]  # 只返回最近 10 条
        ]
