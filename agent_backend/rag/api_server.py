"""
FastAPI 接口层（可选）
提供 RESTful API 服务
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel

from .app import get_enterprise_rag_system
from .models.schemas import ChatRequest, ChatResponse


# ========== FastAPI 应用初始化 ==========
app = FastAPI(
    title="企业知识库 RAG 系统 API",
    description="基于 LangChain 的企业级知识库问答系统",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 请求/响应模型 ==========
class LoginRequest(BaseModel):
    username: str
    password: str


class KnowledgeAddRequest(BaseModel):
    title: str
    content: str
    source_type: str
    source_url: Optional[str] = None


class SessionCreateRequest(BaseModel):
    user_id: str
    title: str = "新会话"


class FeishuSyncRequest(BaseModel):
    folder_token: str
    user_id: str


# ========== 认证依赖 ==========
async def verify_token(x_authorization: Optional[str] = Header(None)):
    """验证 Token（简化版，实际应该更复杂）"""
    if not x_authorization:
        raise HTTPException(status_code=401, detail="缺少认证令牌")
    
    rag_system = get_enterprise_rag_system()
    payload = rag_system.auth_service.verify_token(x_authorization.replace("Bearer ", ""))
    
    if not payload:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    
    return payload["user_id"]


# ========== API 路由 ==========

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """用户登录"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.login(request.username, request.password)
    
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    
    return result


@app.post("/api/knowledge/add")
async def add_knowledge(
    request: KnowledgeAddRequest,
    user_id: str = Depends(verify_token)
):
    """添加知识到知识库"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.add_knowledge(
        title=request.title,
        content=request.content,
        source_type=request.source_type,
        created_by=user_id,
        source_url=request.source_url
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.post("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    """搜索知识"""
    rag_system = get_enterprise_rag_system()
    results = rag_system.search_knowledge(query=query, limit=limit)
    return {"results": results}


@app.post("/api/knowledge/sync-feishu")
async def sync_feishu(
    request: FeishuSyncRequest,
    user_id: str = Depends(verify_token)
):
    """同步飞书文档"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.sync_feishu_documents(
        folder_token=request.folder_token,
        created_by=user_id
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.post("/api/session/create")
async def create_session(
    request: SessionCreateRequest,
    user_id: str = Depends(verify_token)
):
    """创建会话"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.create_session(
        user_id=request.user_id,
        title=request.title
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("message", "创建失败"))
    
    return result


@app.get("/api/session/list")
async def list_sessions(user_id: str = Depends(verify_token)):
    """获取会话列表"""
    rag_system = get_enterprise_rag_system()
    sessions = rag_system.get_session_list(user_id)
    return {"sessions": sessions}


@app.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(verify_token)
):
    """删除会话"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.delete_session(session_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return result


@app.post("/api/session/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = "json",
    user_id: str = Depends(verify_token)
):
    """导出会话"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.export_session(session_id, format)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result


@app.post("/api/chat")
async def chat(request: ChatRequest, user_id: str = Depends(verify_token)):
    """对话接口"""
    try:
        rag_system = get_enterprise_rag_system()
        response = rag_system.chat(request)
        return response.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误：{str(e)}")


@app.post("/api/memory/add")
async def add_memory(
    session_id: str,
    content: str,
    category: str = "general",
    importance: float = 0.5,
    user_id: str = Depends(verify_token)
):
    """添加长期记忆"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.add_long_term_memory(
        session_id=session_id,
        content=content,
        category=category,
        importance=importance
    )
    return result


@app.get("/api/memory/search")
async def search_memories(
    session_id: str,
    query: str,
    user_id: str = Depends(verify_token)
):
    """搜索记忆"""
    rag_system = get_enterprise_rag_system()
    memories = rag_system.search_memories(session_id, query)
    return {"memories": memories}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# ========== 主函数 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
