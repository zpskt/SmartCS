"""
会话管理服务
管理用户会话的创建、切换、删除等
"""
import uuid
import json
import psycopg
from psycopg import rows
from datetime import datetime
from typing import Dict, List, Optional
from config.settings import settings
from models.schemas import Session, Message


class SessionManager:
    """会话管理器类"""
    
    def __init__(self):
        """初始化会话管理器"""
        # 内存缓存（提高读取性能）
        self.sessions_cache: Dict[str, Session] = {}
        self.user_sessions_cache: Dict[str, List[str]] = {}
        
        # PostgreSQL 数据库配置
        self.db_config = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
            'dbname': settings.POSTGRES_DATABASE
        }
        
        # 初始化数据库
        self._init_db()
    
    def _get_connection(self):
        """获取数据库连接"""
        conn = psycopg.connect(**self.db_config, autocommit=True)
        return conn
    
    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                # 创建会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT DEFAULT '新会话',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}',
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # 创建消息表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
                
                conn.commit()
        finally:
            conn.close()
        
        # 加载现有数据到缓存
        self._load_cache()
    
    def _load_cache(self):
        """从数据库加载数据到缓存"""
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                # 加载所有活跃会话
                cursor.execute("SELECT * FROM sessions WHERE is_active = TRUE")
                db_rows = cursor.fetchall()
                
                for row in db_rows:
                    session = Session(
                        session_id=row['session_id'],
                        user_id=row['user_id'],
                        title=row['title'],
                        messages=[],  # 消息单独加载
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']),
                        is_active=row['is_active']
                    )
                    self.sessions_cache[session.session_id] = session
                    
                    if session.user_id not in self.user_sessions_cache:
                        self.user_sessions_cache[session.user_id] = []
                    self.user_sessions_cache[session.user_id].append(session.session_id)
        finally:
            conn.close()
    
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
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        # 插入数据库
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("""
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (session_id, user_id, title, now, now, metadata_json))
                conn.commit()
        finally:
            conn.close()
        
        # 创建 Session 对象
        session = Session(
            session_id=session_id,
            user_id=user_id,
            title=title,
            metadata=metadata or {}
        )
        
        # 更新缓存
        self.sessions_cache[session_id] = session
        if user_id not in self.user_sessions_cache:
            self.user_sessions_cache[user_id] = []
        self.user_sessions_cache[user_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话
        
        :param session_id: 会话 ID
        :return: Session 对象
        """
        # 先从缓存获取
        if session_id in self.sessions_cache:
            session = self.sessions_cache[session_id]
            # 加载消息
            session.messages = self.load_messages(session_id)
            return session
        
        # 从数据库获取
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("SELECT * FROM sessions WHERE session_id = %s AND is_active = TRUE", (session_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                session = Session(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    messages=self.load_messages(session_id),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']),
                    is_active=row['is_active']
                )
                
                # 更新缓存
                self.sessions_cache[session_id] = session
                if session.user_id not in self.user_sessions_cache:
                    self.user_sessions_cache[session.user_id] = []
                if session_id not in self.user_sessions_cache[session.user_id]:
                    self.user_sessions_cache[session.user_id].append(session_id)
                
                return session
        finally:
            conn.close()
    
    def load_messages(self, session_id: str) -> List[Dict[str, any]]:
        """加载会话的消息列表"""
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("""
                    SELECT * FROM messages 
                    WHERE session_id = %s 
                    ORDER BY timestamp ASC
                """, (session_id,))
                
                db_rows = cursor.fetchall()
                return [
                    {
                        'message_id': row['message_id'],
                        'session_id': row['session_id'],
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'metadata': json.loads(row['metadata'])
                    }
                    for row in db_rows
                ]
        finally:
            conn.close()
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        获取用户的所有会话
        
        :param user_id: 用户 ID
        :return: 会话列表
        """
        # 直接从数据库查询，确保数据一致性
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE user_id = %s AND is_active = TRUE 
                    ORDER BY updated_at DESC
                """, (user_id,))
                
                db_rows = cursor.fetchall()
                sessions = []
                
                for row in db_rows:
                    session = Session(
                        session_id=row['session_id'],
                        user_id=row['user_id'],
                        title=row['title'],
                        messages=[],  # 不加载消息，提高性能
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata=json.loads(row['metadata']),
                        is_active=row['is_active']
                    )
                    sessions.append(session)
                    
                    # 更新缓存
                    self.sessions_cache[session.session_id] = session
                
                # 更新用户会话缓存
                self.user_sessions_cache[user_id] = [s.session_id for s in sessions]
                
                return sessions
        finally:
            conn.close()
    
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
        
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        # 插入消息到数据库
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("""
                    INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (message_id, session_id, role, content, now, metadata_json))
                
                # 更新会话的更新时间
                cursor.execute("""
                    UPDATE sessions SET updated_at = %s WHERE session_id = %s
                """, (now, session_id))
                
                # 如果是第一条用户消息，自动更新会话标题
                if len(session.messages) == 0 and role == "user":
                    new_title = content[:50] + "..." if len(content) > 50 else content
                    cursor.execute("""
                        UPDATE sessions SET title = %s, updated_at = %s WHERE session_id = %s
                    """, (new_title, now, session_id))
                    session.title = new_title
                
                conn.commit()
        finally:
            conn.close()
        
        # 创建 Message 对象
        message = Message(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.fromisoformat(now),
            metadata=metadata or {}
        )
        
        # 更新缓存
        session.messages.append(message.dict())
        session.updated_at = datetime.fromisoformat(now)
        
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
        
        # 软删除：标记为不活跃
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("""
                    UPDATE sessions SET is_active = FALSE, updated_at = %s WHERE session_id = %s
                """, (datetime.now().isoformat(), session_id))
                conn.commit()
        finally:
            conn.close()
        
        # 从缓存中移除
        user_id = session.user_id
        if user_id in self.user_sessions_cache:
            if session_id in self.user_sessions_cache[user_id]:
                self.user_sessions_cache[user_id].remove(session_id)
        
        if session_id in self.sessions_cache:
            del self.sessions_cache[session_id]
        
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
        
        # 从数据库删除消息
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
                cursor.execute("""
                    UPDATE sessions SET updated_at = %s WHERE session_id = %s
                """, (datetime.now().isoformat(), session_id))
                conn.commit()
        finally:
            conn.close()
        
        # 更新缓存
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
