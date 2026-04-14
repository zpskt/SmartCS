"""
记忆存储服务
管理短期和长期记忆
"""
import json
from typing import Dict, List, Optional
from datetime import datetime
import pymysql
from pymysql.cursors import DictCursor
from models.schemas import Message
from config.settings import settings


class MemoryStore:
    """记忆存储服务类"""
    
    def __init__(self, max_short_term_memory: int = 10):
        """
        初始化记忆存储服务
        
        :param max_short_term_memory: 短期记忆最大条数
        """
        self.max_short_term_memory = max_short_term_memory
        
        # MySQL 数据库配置
        self.db_config = {
            'host': settings.MYSQL_HOST,
            'port': settings.MYSQL_PORT,
            'user': settings.MYSQL_USER,
            'password': settings.MYSQL_PASSWORD,
            'database': settings.MYSQL_DATABASE,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
            'autocommit': True
        }
        
        # 初始化数据库表
        self._init_database()
        
        # 短期记忆缓存（用于快速访问）
        self.short_term_cache: Dict[str, List[Message]] = {}
        
        # 长期记忆缓存
        self.long_term_cache: Dict[str, List[Dict]] = {}
    
    def _get_connection(self) -> pymysql.Connection:
        """获取数据库连接"""
        return pymysql.connect(**self.db_config)
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 创建短期记忆表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS short_term_memories (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        message_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session_id (session_id),
                        INDEX idx_timestamp (timestamp)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                ''')
                
                # 创建长期记忆表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        category VARCHAR(100) DEFAULT 'general',
                        importance FLOAT DEFAULT 0.5,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session_id (session_id),
                        INDEX idx_category (category),
                        INDEX idx_importance (importance)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                ''')
        finally:
            conn.close()
    
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
        message = Message(
            message_id=f"mem_{datetime.now().timestamp()}",
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        # 存入数据库
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO short_term_memories (session_id, message_id, role, content, metadata) VALUES (%s, %s, %s, %s, %s)',
                    (session_id, message.message_id, role, content, json.dumps(metadata or {}, ensure_ascii=False))
                )
                
                # 限制记忆长度，删除最旧的记录
                cursor.execute(
                    'SELECT COUNT(*) as count FROM short_term_memories WHERE session_id = %s',
                    (session_id,)
                )
                result = cursor.fetchone()
                count = result['count'] if result else 0
                
                if count > self.max_short_term_memory:
                    delete_count = count - self.max_short_term_memory
                    cursor.execute(
                        'DELETE FROM short_term_memories WHERE session_id = %s ORDER BY timestamp ASC LIMIT %s',
                        (session_id, delete_count)
                    )
        finally:
            conn.close()
        
        # 更新缓存
        if session_id not in self.short_term_cache:
            self.short_term_cache[session_id] = []
        self.short_term_cache[session_id].append(message)
        
        # 限制缓存长度
        if len(self.short_term_cache[session_id]) > self.max_short_term_memory:
            self.short_term_cache[session_id] = self.short_term_cache[session_id][-self.max_short_term_memory:]
    
    def get_short_term_memory(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取短期记忆
        
        :param session_id: 会话 ID
        :return: 消息列表
        """
        # 先从缓存获取
        if session_id in self.short_term_cache and self.short_term_cache[session_id]:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in self.short_term_cache[session_id]
            ]
        
        # 缓存未命中，从数据库加载
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT role, content FROM short_term_memories WHERE session_id = %s ORDER BY timestamp ASC LIMIT %s',
                    (session_id, self.max_short_term_memory)
                )
                messages = cursor.fetchall()
                
                # 更新缓存
                self.short_term_cache[session_id] = [
                    Message(
                        message_id=f"cached_{i}",
                        session_id=session_id,
                        role=msg['role'],
                        content=msg['content'],
                        metadata={},
                        timestamp=datetime.now()
                    )
                    for i, msg in enumerate(messages)
                ]
                
                return [
                    {"role": msg['role'], "content": msg['content']}
                    for msg in messages
                ]
        finally:
            conn.close()
    
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
        memory = {
            "content": content,
            "category": category,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
        # 存入数据库
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO long_term_memories (session_id, content, category, importance) VALUES (%s, %s, %s, %s)',
                    (session_id, content, category, importance)
                )
        finally:
            conn.close()
        
        # 更新缓存
        if session_id not in self.long_term_cache:
            self.long_term_cache[session_id] = []
        self.long_term_cache[session_id].append(memory)
    
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
        # 先从缓存获取
        if session_id in self.long_term_cache and self.long_term_cache[session_id]:
            memories = self.long_term_cache[session_id]
            if category:
                return [m for m in memories if m.get("category") == category]
            return memories
        
        # 缓存未命中，从数据库加载
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if category:
                    cursor.execute(
                        'SELECT content, category, importance, timestamp FROM long_term_memories WHERE session_id = %s AND category = %s ORDER BY timestamp DESC',
                        (session_id, category)
                    )
                else:
                    cursor.execute(
                        'SELECT content, category, importance, timestamp FROM long_term_memories WHERE session_id = %s ORDER BY timestamp DESC',
                        (session_id,)
                    )
                memories = cursor.fetchall()
                
                result = [
                    {
                        "content": m['content'],
                        "category": m['category'],
                        "importance": m['importance'],
                        "timestamp": m['timestamp'].isoformat() if isinstance(m['timestamp'], datetime) else str(m['timestamp'])
                    }
                    for m in memories
                ]
                
                # 更新缓存
                self.long_term_cache[session_id] = result
                
                if category:
                    return [m for m in result if m.get("category") == category]
                return result
        finally:
            conn.close()
    
    def clear_short_term_memory(self, session_id: str):
        """
        清空短期记忆
        
        :param session_id: 会话 ID
        """
        # 删除数据库记录
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM short_term_memories WHERE session_id = %s', (session_id,))
        finally:
            conn.close()
        
        # 清空缓存
        if session_id in self.short_term_cache:
            self.short_term_cache[session_id] = []
    
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
        # 从数据库搜索
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT content, category, importance, timestamp FROM long_term_memories WHERE session_id = %s AND content LIKE %s ORDER BY importance DESC',
                    (session_id, f'%{query}%')
                )
                memories = cursor.fetchall()
                
                return [
                    {
                        "content": m['content'],
                        "category": m['category'],
                        "importance": m['importance'],
                        "timestamp": m['timestamp'].isoformat() if isinstance(m['timestamp'], datetime) else str(m['timestamp'])
                    }
                    for m in memories
                ]
        finally:
            conn.close()
