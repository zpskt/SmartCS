"""
系统配置管理
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置类"""
    
    # ========== 基础配置 ==========
    APP_NAME: str = "企业知识库 RAG 系统"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, production
    
    # ========== 模型配置 ==========
    MODEL_PROVIDER: str = "dashscope"  # dashscope, ollama
    CHAT_MODEL: str = "qwen-max"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    DASHSCOPE_API_KEY: Optional[str] = None
    TEMPERATURE: float = 0.7
    
    # Ollama 配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:7b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # ========== 向量数据库配置 ==========
    VECTOR_DB_TYPE: str = "chroma"  # chroma, milvus, weaviate
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_NAME: str = "knowledge_base"
    
    # ========== Checkpointer 数据库配置 ==========
    CHECKPOINTER_DB_PATH: str = "./data/checkpointer.db"  # SQLite 数据库路径
    
    # ========== 文本分割配置 ==========
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_CHUNK_SIZE: int = 2000
    
    # ========== 会话配置 ==========
    DEFAULT_SESSION_ID: str = "default_session"
    SESSION_TTL_HOURS: int = 24  # 会话存活时间（小时）
    MAX_SHORT_TERM_MEMORY: int = 10  # 短期记忆最大条数
    
    # ========== 权限配置 ==========
    ENABLE_AUTH: bool = True
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 小时
    ADMIN_USERS: List[str] = ["admin", "zpaskt"]
    
    # ========== 飞书配置 ==========
    FEISHU_APP_ID: str = ""
    FEISHU_APP_SECRET: str = ""
    FEISHU_VERIFICATION_TOKEN: Optional[str] = None
    FEISHU_BASE_URL: str = "https://open.feishu.cn/open-apis"
    
    # ========== 检索配置 ==========
    SIMILARITY_TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.5
    
    # ========== 日志配置 ==========
    LOG_LEVEL: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    LOG_DIR: str = "./logs"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置对象"""
    return settings
