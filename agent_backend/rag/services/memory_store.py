"""
记忆存储服务
管理短期和长期记忆
"""
from typing import Dict, List, Optional
from datetime import datetime
from models.schemas import Message


class MemoryStore:
    """记忆存储服务类"""
    
    def __init__(self, max_short_term_memory: int = 10):
        """
        初始化记忆存储服务
        
        :param max_short_term_memory: 短期记忆最大条数
        """
        self.max_short_term_memory = max_short_term_memory
        
        # 短期记忆（当前会话）
        self.short_term_memories: Dict[str, List[Message]] = {}
        
        # 长期记忆（重要信息）
        self.long_term_memories: Dict[str, List[Dict]] = {}
    
    def add_short_term_memory(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        添加短期记忆
        
        :param session_id: 会话 ID
        :param role: 角色
        :param content: 内容
        :param metadata: 元数据
        """
        if session_id not in self.short_term_memories:
            self.short_term_memories[session_id] = []
        
        message = Message(
            message_id=f"mem_{datetime.now().timestamp()}",
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self.short_term_memories[session_id].append(message)
        
        # 限制记忆长度
        if len(self.short_term_memories[session_id]) > self.max_short_term_memory:
            self.short_term_memories[session_id] = self.short_term_memories[session_id][-self.max_short_term_memory:]
    
    def get_short_term_memory(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取短期记忆
        
        :param session_id: 会话 ID
        :return: 消息列表
        """
        messages = self.short_term_memories.get(session_id, [])
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def add_long_term_memory(
        self,
        session_id: str,
        content: str,
        category: str = "general",
        importance: float = 0.5
    ):
        """
        添加长期记忆
        
        :param session_id: 会话 ID
        :param content: 记忆内容
        :param category: 分类
        :param importance: 重要程度（0-1）
        """
        if session_id not in self.long_term_memories:
            self.long_term_memories[session_id] = []
        
        memory = {
            "content": content,
            "category": category,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
        self.long_term_memories[session_id].append(memory)
    
    def get_long_term_memory(
        self,
        session_id: str,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        获取长期记忆
        
        :param session_id: 会话 ID
        :param category: 分类过滤（可选）
        :return: 记忆列表
        """
        memories = self.long_term_memories.get(session_id, [])
        if category:
            return [m for m in memories if m.get("category") == category]
        return memories
    
    def clear_short_term_memory(self, session_id: str):
        """
        清空短期记忆
        
        :param session_id: 会话 ID
        """
        if session_id in self.short_term_memories:
            self.short_term_memories[session_id] = []
    
    def search_long_term_memory(
        self,
        session_id: str,
        query: str
    ) -> List[Dict]:
        """
        搜索长期记忆（简单关键词匹配）
        
        :param session_id: 会话 ID
        :param query: 查询关键词
        :return: 匹配的记忆列表
        """
        memories = self.long_term_memories.get(session_id, [])
        return [
            m for m in memories
            if query.lower() in m.get("content", "").lower()
        ]
