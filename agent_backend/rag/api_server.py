"""
FastAPI 接口层（可选）
提供 RESTful API 服务
"""
import time
import json
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from typing import List, Optional
from pydantic import BaseModel
import logging

from app import get_enterprise_rag_system
from models.schemas import ChatRequest, ChatResponse
from utils.logger import get_logger

# 获取 API 专用日志记录器
logger = get_logger("api_server")


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


# ========== 请求日志中间件 ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有 HTTP 请求和响应"""
    start_time = time.time()
    
    # 获取请求信息
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "unknown"
    
    # 检测是否为流式请求
    is_streaming_request = False
    if method == "POST" and "/api/chat" in url:
        try:
            body = await request.body()
            if body:
                import json as json_mod
                try:
                    body_data = json_mod.loads(body)
                    is_streaming_request = body_data.get("stream", False)
                except:
                    pass
                
                # 只有非流式请求才重写 receive
                if not is_streaming_request:
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
        except Exception as e:
            logger.warning(f"读取请求体失败: {e}")
    
    # 记录请求开始
    logger.info(f"📨 请求开始 | {method} {url} | 客户端: {client_host}")
    
    # 处理请求
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 记录响应信息
    status_code = response.status_code
    status_emoji = "✅" if status_code < 400 else "❌"
    
    # 检测是否为流式响应
    is_streaming_response = hasattr(response, 'body_iterator')
    
    logger.info(
        f"{status_emoji} 请求完成 | {method} {url} | "
        f"状态码: {status_code} | "
        f"耗时: {process_time:.3f}s | "
        f"{'流式' if is_streaming_request or is_streaming_response else '普通'}"
    )
    
    # 如果是错误响应，记录详细信息
    if status_code >= 400:
        logger.warning(
            f"⚠️ 错误响应 | {method} {url} | "
            f"状态码: {status_code} | "
            f"耗时: {process_time:.3f}s"
        )
    
    return response


# ========== 请求/响应模型 ==========
class LoginRequest(BaseModel):
    username: str
    password: str


class KnowledgeAddRequest(BaseModel):
    title: str
    content: str
    source_type: str
    source_url: Optional[str] = None


class KnowledgeUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[str] = None


class KnowledgeDeleteRequest(BaseModel):
    doc_id: str


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
    logger.info(f"🔐 用户登录尝试 | 用户名: {request.username}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.login(request.username, request.password)
    
    if not result["success"]:
        logger.warning(f"❌ 登录失败 | 用户名: {request.username} | 原因: {result['message']}")
        raise HTTPException(status_code=401, detail=result["message"])
    
    logger.info(f"✅ 登录成功 | 用户ID: {result.get('user_id')} | 角色: {result.get('role')}")
    return result


@app.post("/api/knowledge/add")
async def add_knowledge(
    request: KnowledgeAddRequest,
    user_id: str = Depends(verify_token)
):
    """添加知识到知识库"""
    logger.info(f"📝 添加知识 | 用户: {user_id} | 标题: {request.title} | 类型: {request.source_type}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.add_knowledge(
        title=request.title,
        content=request.content,
        source_type=request.source_type,
        created_by=user_id,
        source_url=request.source_url
    )
    
    if not result["success"]:
        logger.error(f"❌ 添加知识失败 | 用户: {user_id} | 原因: {result['message']}")
        raise HTTPException(status_code=400, detail=result["message"])
    
    logger.info(f"✅ 知识添加成功 | 文档ID: {result.get('doc_id')}")
    return result


@app.post("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    """搜索知识"""
    rag_system = get_enterprise_rag_system()
    results = rag_system.search_knowledge(query=query, limit=limit)
    return {"results": results}


@app.post("/api/knowledge/delete")
async def delete_knowledge(
    request: KnowledgeDeleteRequest,
    user_id: str = Depends(verify_token)
):
    """删除知识库文档"""
    logger.info(f"🗑️ 删除知识库文档 | 用户: {user_id} | 文档ID: {request.doc_id}")
    rag_system = get_enterprise_rag_system()
    
    success = rag_system.knowledge_base.delete_document(request.doc_id)
    
    if not success:
        logger.warning(f"⚠️ 删除失败 | 文档不存在: {request.doc_id}")
        raise HTTPException(status_code=404, detail="文档不存在")
    
    logger.info(f"✅ 文档删除成功 | 文档ID: {request.doc_id}")
    return {"success": True, "message": "文档已删除"}


@app.put("/api/knowledge/{doc_id}")
async def update_knowledge(
    doc_id: str,
    request: KnowledgeUpdateRequest,
    user_id: str = Depends(verify_token)
):
    """更新知识库文档"""
    logger.info(f"✏️ 更新知识库文档 | 用户: {user_id} | 文档ID: {doc_id}")
    rag_system = get_enterprise_rag_system()
    
    try:
        # 获取原文档信息（需要从向量库查询）
        all_data = rag_system.knowledge_base.vector_store.get_all_documents()
        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])
        
        # 查找原文档元数据
        original_metadata = None
        original_content = None
        for meta, doc in zip(metadatas, documents):
            if meta and meta.get("doc_id") == doc_id:
                original_metadata = meta
                original_content = doc
                break
        
        if not original_metadata:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 构建新文档内容
        new_title = request.title or original_metadata.get("title", "")
        new_content = request.content or original_content
        new_source_url = request.source_url if request.source_url is not None else original_metadata.get("source_url", "")
        
        # 删除旧文档
        rag_system.knowledge_base.delete_document(doc_id)
        
        # 添加新文档（保持原有元数据）
        result = rag_system.add_knowledge(
            title=new_title,
            content=new_content,
            source_type=original_metadata.get("source_type", "manual"),
            created_by=original_metadata.get("created_by", user_id),
            source_url=new_source_url if new_source_url else None
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"✅ 文档更新成功 | 文档ID: {doc_id}")
        return {"success": True, "message": "文档已更新", "doc_id": result.get("doc_id")}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新文档失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新失败：{str(e)}")


@app.get("/api/knowledge/list")
async def list_knowledge(
    page: int = 1,
    page_size: int = 10,
    user_id: str = Depends(verify_token)
):
    """获取知识库文档列表（分页）"""
    logger.info(f"📚 获取知识库列表 | 用户: {user_id} | 页码: {page} | 每页: {page_size}")
    rag_system = get_enterprise_rag_system()
    
    # 获取所有文档
    all_docs = rag_system.knowledge_base.get_all_documents()
    total = len(all_docs)
    
    # 计算分页
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = all_docs[start_idx:end_idx]
    
    logger.info(f"✅ 返回知识库列表 | 总数: {total} | 当前页: {len(paginated_docs)}")
    
    return {
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        "documents": paginated_docs
    }


@app.get("/api/knowledge/{doc_id}")
async def get_knowledge_detail(
    doc_id: str,
    user_id: str = Depends(verify_token)
):
    """获取单个文档详情"""
    logger.info(f"📄 获取文档详情 | 用户: {user_id} | 文档ID: {doc_id}")
    rag_system = get_enterprise_rag_system()
    
    all_docs = rag_system.knowledge_base.get_all_documents()
    doc = next((d for d in all_docs if d["doc_id"] == doc_id), None)
    
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    return {"success": True, "document": doc}


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
    logger.info(f"🆕 创建会话 | 用户: {user_id} | 标题: {request.title}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.create_session(
        user_id=request.user_id,
        title=request.title
    )
    
    if not result["success"]:
        logger.error(f"❌ 创建会话失败 | 用户: {user_id} | 原因: {result.get('message', '未知错误')}")
        raise HTTPException(status_code=400, detail=result.get("message", "创建失败"))
    
    logger.info(f"✅ 会话创建成功 | 会话ID: {result.get('session_id')}")
    return result


@app.get("/api/session/list")
async def list_sessions(user_id: str = Depends(verify_token)):
    """获取会话列表"""
    logger.info(f"📋 获取会话列表 | 用户: {user_id}")
    rag_system = get_enterprise_rag_system()
    sessions = rag_system.get_session_list(user_id)
    logger.info(f"✅ 返回 {len(sessions)} 个会话")
    return {"sessions": sessions}


@app.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(verify_token)
):
    """删除会话"""
    logger.info(f"🗑️ 删除会话 | 用户: {user_id} | 会话ID: {session_id}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.delete_session(session_id)
    
    if not result["success"]:
        logger.warning(f"⚠️ 删除会话失败 | 会话不存在: {session_id}")
        raise HTTPException(status_code=404, detail="会话不存在")
    
    logger.info(f"✅ 会话删除成功 | 会话ID: {session_id}")
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
    logger.info(f"💬 对话请求 | 用户: {user_id} | 会话: {request.session_id} | 流式: {request.stream}")
    logger.debug(f"问题: {request.message[:200]}")
    
    try:
        rag_system = get_enterprise_rag_system()
        
        # 根据 stream 参数选择响应方式
        if request.stream:
            # 流式响应 - 使用异步生成器
            return StreamingResponse(
                rag_system.chat_stream(request),
                media_type="text/event-stream"
            )
        else:
            # 一次性响应
            response = rag_system.chat(request)
            logger.info(f"✅ 对话成功 | 响应长度: {len(response.content)} 字符")
            return response.model_dump()
    except ValueError as e:
        logger.warning(f"⚠️ 对话失败 - 参数错误 | 用户: {user_id} | 错误: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 对话失败 - 服务器错误 | 用户: {user_id} | 错误: {str(e)}", exc_info=True)
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
