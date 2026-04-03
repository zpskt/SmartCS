"""
企业知识库 RAG 系统 - 主应用
整合所有服务，提供统一的调用接口
"""
from typing import Dict, Any, Optional, List
from services.auth import get_auth_service, AuthService
from services.knowledge_base import KnowledgeBaseService
from services.rag_engine import RAGEngine
from services.session_manager import SessionManager
from services.memory_store import MemoryStore
from integrations.feishu import FeishuClient
from models.schemas import ChatRequest, ChatResponse


class EnterpriseRAGSystem:
    """企业级 RAG 知识库系统类"""
    
    def __init__(self):
        """初始化企业 RAG 系统"""
        self.auth_service = get_auth_service()
        self.knowledge_base = KnowledgeBaseService()
        self.rag_engine = RAGEngine()
        self.session_manager = SessionManager()
        self.memory_store = MemoryStore()
        self.feishu_client = FeishuClient()
    
    # ========== 权限管理 ==========
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        用户登录
        
        :param username: 用户名
        :param password: 密码
        :return: 登录结果
        """
        user = self.auth_service.authenticate_user(username, password)
        if not user:
            return {"success": False, "message": "用户名或密码错误"}
        
        token = self.auth_service.create_access_token(user["user_id"])
        return {
            "success": True,
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "token": token
        }
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        检查权限
        
        :param user_id: 用户 ID
        :param permission: 权限名称
        :return: 是否有权限
        """
        return self.auth_service.has_permission(user_id, permission)
    
    # ========== 知识库管理 ==========
    def add_knowledge(
        self,
        title: str,
        content: str,
        source_type: str,
        created_by: str,
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加知识到知识库
        
        :param title: 文档标题
        :param content: 文档内容
        :param source_type: 来源类型
        :param created_by: 创建者
        :param source_url: 来源 URL
        :return: 添加结果
        """
        try:
            doc = self.knowledge_base.add_document(
                title=title,
                content=content,
                source_type=source_type,
                created_by=created_by,
                source_url=source_url
            )
            return {
                "success": True,
                "doc_id": doc.doc_id,
                "message": "知识添加成功"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def sync_feishu_documents(
        self,
        folder_token: str,
        created_by: str
    ) -> Dict[str, Any]:
        """
        同步飞书文档
        
        :param folder_token: 文件夹 token
        :param created_by: 创建者
        :return: 同步结果
        """
        try:
            feishu_docs = self.feishu_client.sync_folder_documents(
                folder_token=folder_token
            )
            docs = self.knowledge_base.add_documents_from_feishu(
                feishu_docs=feishu_docs,
                created_by=created_by
            )
            return {
                "success": True,
                "count": len(docs),
                "message": f"成功同步 {len(docs)} 个飞书文档"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def search_knowledge(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索知识
        
        :param query: 查询文本
        :param limit: 返回数量限制
        :return: 搜索结果列表
        """
        return self.knowledge_base.search_documents(query, limit)
    
    # ========== 会话管理 ==========
    def create_session(self, user_id: str, title: str = "新会话") -> Dict[str, Any]:
        """
        创建会话
        
        :param user_id: 用户 ID
        :param title: 会话标题
        :return: 创建结果
        """
        session = self.session_manager.create_session(
            user_id=user_id,
            title=title
        )
        return {
            "success": True,
            "session_id": session.session_id,
            "title": session.title
        }
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        对话
        
        :param request: 聊天请求
        :return: 聊天响应
        """
        # 验证会话
        session = self.session_manager.get_session(request.session_id)
        if not session:
            raise ValueError(f"会话 {request.session_id} 不存在")
        
        # 添加用户消息
        self.session_manager.add_message_to_session(
            session_id=request.session_id,
            role="user",
            content=request.message
        )
        
        # 获取会话历史
        chat_history = self.session_manager.get_session_history(request.session_id)
        
        # 执行 RAG 查询
        rag_response = self.rag_engine.query(
            question=request.message,
            chat_history=chat_history,
            include_sources=True
        )
        
        # 添加 AI 响应到会话
        ai_message = self.session_manager.add_message_to_session(
            session_id=request.session_id,
            role="assistant",
            content=rag_response["answer"]
        )
        
        # 构建响应
        response = ChatResponse(
            session_id=request.session_id,
            message_id=ai_message.message_id,
            content=rag_response["answer"],
            sources=rag_response["sources"]
        )
        
        return response
    
    def get_session_list(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的会话列表
        
        :param user_id: 用户 ID
        :return: 会话列表
        """
        sessions = self.session_manager.get_user_sessions(user_id)
        return [
            {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
            for session in sessions
        ]
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        删除会话
        
        :param session_id: 会话 ID
        :return: 删除结果
        """
        success = self.session_manager.delete_session(session_id)
        return {
            "success": success,
            "message": "会话已删除" if success else "会话不存在"
        }
    
    def export_session(self, session_id: str, format: str = "json") -> Dict[str, Any]:
        """
        导出会话
        
        :param session_id: 会话 ID
        :param format: 导出格式
        :return: 导出的内容
        """
        try:
            content = self.session_manager.export_session(session_id, format)
            return {
                "success": True,
                "content": content,
                "format": format
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def clear_session_messages(self, session_id: str) -> Dict[str, Any]:
        """
        清空会话消息
        
        :param session_id: 会话 ID
        :return: 清空结果
        """
        success = self.session_manager.clear_session_messages(session_id)
        return {
            "success": success,
            "message": "会话消息已清空" if success else "会话不存在"
        }
    
    # ========== 记忆管理 ==========
    def add_long_term_memory(
        self,
        session_id: str,
        content: str,
        category: str = "general",
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        添加长期记忆
        
        :param session_id: 会话 ID
        :param content: 记忆内容
        :param category: 分类
        :param importance: 重要程度
        :return: 添加结果
        """
        self.memory_store.add_long_term_memory(
            session_id=session_id,
            content=content,
            category=category,
            importance=importance
        )
        return {
            "success": True,
            "message": "记忆已保存"
        }
    
    def search_memories(
        self,
        session_id: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        :param session_id: 会话 ID
        :param query: 查询关键词
        :return: 匹配的记忆列表
        """
        return self.memory_store.search_long_term_memory(session_id, query)


# 单例模式
_system_instance: Optional[EnterpriseRAGSystem] = None


def get_enterprise_rag_system() -> EnterpriseRAGSystem:
    """
    获取企业 RAG 系统实例（单例）
    
    :return: EnterpriseRAGSystem 实例
    """
    global _system_instance
    if _system_instance is None:
        _system_instance = EnterpriseRAGSystem()
    return _system_instance
