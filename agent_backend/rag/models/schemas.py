"""
数据模型定义
"""
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class User(BaseModel):
    """用户模型"""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = "user"  # admin, user, guest
    permissions: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True


class KnowledgeDocument(BaseModel):
    """知识文档模型"""
    doc_id: str
    title: str
    content: str
    source_type: str  # file, feishu, url, manual
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    chunks: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str
    status: str = "active"  # active, archived, deleted
    file_size: Optional[int] = None  # 文件大小（字节）


class Session(BaseModel):
    """会话模型"""
    session_id: str
    user_id: str
    title: str = "新会话"
    messages: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}
    is_active: bool = True


class Message(BaseModel):
    """消息模型"""
    message_id: str
    session_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """聊天请求模型"""
    session_id: str
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    """聊天响应模型"""
    session_id: str
    message_id: str
    content: str
    sources: List[Dict[str, Any]] = []  # 参考来源
    timestamp: datetime = Field(default_factory=datetime.now)
