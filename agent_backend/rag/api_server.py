"""
FastAPI 接口层（可选）
提供 RESTful API 服务
"""
import string
import time
import json
from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from app import get_enterprise_rag_system
from models.schemas import ChatRequest, ChatResponse, IntentRecognitionRequest, IntentRecognitionResponse
from utils.logger import get_logger
from utils.permissions import (
    require_roles,
    require_permissions,
    require_role_and_permissions,
    require_admin,
    require_read_permission,
    require_write_permission,
    require_delete_permission,
    require_manage_users_permission,
    require_knowledge_management
)

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
    if method == "POST" and ("/api/chat" in url or "/api/model-adapter/chat" in url):
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
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[str] = None


class KnowledgeDeleteRequest(BaseModel):
    doc_id: str


class SessionCreateRequest(BaseModel):
    user_id: str
    title: str = "新会话"


class SessionUpdateRequest(BaseModel):
    """更新会话请求"""
    title: str


class FeishuSyncRequest(BaseModel):
    folder_token: str
    user_id: str


class UserCreateRequest(BaseModel):
    """创建用户请求"""
    user_id: str
    username: str
    password: str
    role: str = "user"  # admin, user


class UserUpdateRequest(BaseModel):
    """更新用户请求"""
    password: Optional[str] = None
    role: Optional[str] = None


class UserDeleteRequest(BaseModel):
    """删除用户请求"""
    user_id: str


class ExecuteToolRequest(BaseModel):
    """执行工具请求"""
    tool_name: str
    parameters: Dict[str, Any]


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
    user_id: str = Depends(require_write_permission())
):
    """添加知识到知识库（需要写入权限）"""
    logger.info(f"📝 添加知识 | 用户: {user_id} | 标题: {request.title} | 类型: {request.source_type}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.add_knowledge(
        title=request.title,
        content=request.content,
        source_type=request.source_type,
        created_by=user_id,
        source_url=request.source_url,
        metadata=request.metadata
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
    user_id: str = Depends(require_delete_permission())
):
    """删除知识库文档（需要删除权限）"""
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
    user_id: str = Depends(require_write_permission())
):
    """更新知识库文档（需要写入权限）"""
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
    source_type: Optional[str] = None,
    user_id: str = Depends(require_read_permission())
):
    """获取知识库文档列表（分页，需要读取权限）
    
    Args:
        page: 页码
        page_size: 每页数量
        source_type: 文档来源类型过滤 (manual, feishu, upload, web, other)，不传则返回全部
    """
    logger.info(f"📚 获取知识库列表 | 用户: {user_id} | 页码: {page} | 每页: {page_size} | 类型: {source_type or '全部'}")
    rag_system = get_enterprise_rag_system()
    
    # 获取所有文档
    all_docs = rag_system.knowledge_base.get_all_documents()
    
    # 根据 source_type 过滤
    if source_type:
        filtered_docs = [doc for doc in all_docs if doc.get("source_type") == source_type]
        logger.info(f"🔍 按类型过滤 | 类型: {source_type} | 过滤前: {len(all_docs)} | 过滤后: {len(filtered_docs)}")
    else:
        filtered_docs = all_docs
    
    total = len(filtered_docs)
    
    # 计算分页
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = filtered_docs[start_idx:end_idx]
    
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
    user_id: str = Depends(require_read_permission())
):
    """获取单个文档详情（需要读取权限）"""
    logger.info(f"📄 获取文档详情 | 用户: {user_id} | 文档ID: {doc_id}")
    rag_system = get_enterprise_rag_system()
    
    all_docs = rag_system.knowledge_base.get_all_documents()
    doc = next((d for d in all_docs if d["doc_id"] == doc_id), None)
    
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    return {"success": True, "document": doc}


@app.post("/api/knowledge/upload")
async def upload_knowledge_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = None,
    user_id: str = Depends(require_write_permission())
):
    """上传文件到知识库（支持 CSV、Excel、TXT、MD 等，需要写入权限）"""
    logger.info(f"📤 上传文件 | 用户: {user_id} | 文件名: {file.filename}")
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 解析元数据
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="元数据格式错误，请使用有效的 JSON")
        
        # 调用 RAG 系统处理文件（自动识别文件类型并选择合适的分割策略）
        rag_system = get_enterprise_rag_system()
        result = rag_system.upload_file_to_knowledge(
            filename=file.filename,
            content=content,  # 传递原始字节，让后端处理
            created_by=user_id,
            metadata=parsed_metadata
        )
        
        if not result["success"]:
            logger.error(f"❌ 文件上传失败 | 用户: {user_id} | 原因: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"✅ 文件上传成功 | 文档ID: {result.get('doc_id')}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 文件上传失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")


@app.post("/api/knowledge/sync-feishu")
async def sync_feishu(
    request: FeishuSyncRequest,
    user_id: str = Depends(require_role_and_permissions(
        required_roles=["admin"],
        required_permissions=["write", "manage_knowledge"]
    ))
):
    """同步飞书文档（需要管理员角色或知识库管理权限）"""
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
    user_id: str = Depends(require_write_permission())
):
    """创建会话（需要写入权限）"""
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
async def list_sessions(user_id: str = Depends(require_read_permission())):
    """获取会话列表（需要读取权限）"""
    logger.info(f"📋 获取会话列表 | 用户: {user_id}")
    rag_system = get_enterprise_rag_system()
    sessions = rag_system.get_session_list(user_id)
    logger.info(f"✅ 返回 {len(sessions)} 个会话")
    return {"sessions": sessions}


@app.get("/api/session/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user_id: str = Depends(require_read_permission())
):
    """获取会话的聊天记录（需要读取权限）"""
    logger.info(f"💬 获取会话消息 | 用户: {user_id} | 会话ID: {session_id}")
    
    rag_system = get_enterprise_rag_system()
    session_manager = rag_system.session_manager
    
    # 获取会话信息
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 验证权限：只能查看自己的会话
    if session.user_id != user_id:
        logger.warning(f"⚠️ 权限不足 | 用户: {user_id} 尝试查看会话: {session_id}")
        raise HTTPException(status_code=403, detail="无权查看此会话")
    
    # 加载消息
    messages = session_manager.load_messages(session_id)
    logger.info(f"✅ 返回 {len(messages)} 条消息")
    
    return {
        "success": True,
        "session_id": session_id,
        "title": session.title,
        "total_messages": len(messages),
        "messages": messages
    }


@app.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(require_delete_permission())
):
    """删除会话（软删除，需要删除权限）"""
    logger.info(f"🗑️ 删除会话 | 用户: {user_id} | 会话ID: {session_id}")
    rag_system = get_enterprise_rag_system()
    result = rag_system.delete_session(session_id)
    
    if not result["success"]:
        logger.warning(f"⚠️ 删除会话失败 | 会话不存在: {session_id}")
        raise HTTPException(status_code=404, detail="会话不存在")
    
    logger.info(f"✅ 会话删除成功 | 会话ID: {session_id}")
    return result


@app.post("/api/session/{session_id}/clear")
async def clear_session_messages(
    session_id: str,
    user_id: str = Depends(require_delete_permission())
):
    """删除会话（逻辑删除，需要删除权限）"""
    logger.info(f"🗑️ 删除会话 | 用户: {user_id} | 会话ID: {session_id}")
    
    rag_system = get_enterprise_rag_system()
    session_manager = rag_system.session_manager
    
    # 获取会话信息
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 验证权限：只能操作自己的会话
    if session.user_id != user_id:
        logger.warning(f"⚠️ 权限不足 | 用户: {user_id} 尝试删除会话: {session_id}")
        raise HTTPException(status_code=403, detail="无权操作此会话")
    
    # 检查是否至少保留一个会话
    all_sessions = session_manager.get_user_sessions(user_id)
    active_sessions = [s for s in all_sessions if s.is_active]
    if len(active_sessions) <= 1:
        raise HTTPException(status_code=400, detail="至少保留一个会话")
    
    # 逻辑删除会话（标记为不活跃）
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="删除会话失败")
    
    logger.info(f"✅ 会话删除成功 | 会话ID: {session_id}")
    return {
        "success": True,
        "message": "会话已删除",
        "session_id": session_id
    }


@app.put("/api/session/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    user_id: str = Depends(require_write_permission())
):
    """修改会话名称（需要写入权限）"""
    logger.info(f"✏️ 修改会话名称 | 用户: {user_id} | 会话ID: {session_id} | 新名称: {request.title}")
    
    rag_system = get_enterprise_rag_system()
    session_manager = rag_system.session_manager
    
    # 获取会话信息
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 验证权限：只能修改自己的会话
    if session.user_id != user_id:
        logger.warning(f"⚠️ 权限不足 | 用户: {user_id} 尝试修改会话: {session_id}")
        raise HTTPException(status_code=403, detail="无权操作此会话")
    
    # 更新会话标题
    import sqlite3
    from datetime import datetime
    
    conn = session_manager.conn
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?',
        (request.title, datetime.now().isoformat(), session_id)
    )
    conn.commit()
    
    # 更新缓存
    if session_id in session_manager.sessions_cache:
        session_manager.sessions_cache[session_id].title = request.title
        session_manager.sessions_cache[session_id].updated_at = datetime.now()
    
    logger.info(f"✅ 会话名称修改成功 | 会话ID: {session_id}")
    return {
        "success": True,
        "message": "会话名称已修改",
        "session_id": session_id,
        "title": request.title
    }


@app.post("/api/session/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = "json",
    user_id: str = Depends(require_read_permission())
):
    """导出会话（需要读取权限）"""
    rag_system = get_enterprise_rag_system()
    result = rag_system.export_session(session_id, format)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result


@app.post("/api/chat")
async def chat(request: ChatRequest, user_id: str = Depends(require_write_permission())):
    """对话接口（需要写入权限）"""
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


@app.post("/api/model-adapter/chat")
async def model_adapter_chat(request: ChatRequest, user_id: str = Depends(require_write_permission())):
    """型号适配专用对话接口（需要写入权限）"""
    logger.info(f"🔧 型号适配对话请求 | 用户: {user_id} | 会话: {request.session_id} | 流式: {request.stream}")
    logger.debug(f"问题: {request.message[:200]}")
    
    try:
        rag_system = get_enterprise_rag_system()
        
        # 根据 stream 参数选择响应方式
        if request.stream:
            # 流式响应
            return StreamingResponse(
                rag_system.model_adapter_chat_stream(request),
                media_type="text/event-stream"
            )
        else:
            # 一次性响应
            response = rag_system.model_adapter_chat(request)
            logger.info(f"✅ 型号适配对话成功 | 响应长度: {len(response.content)} 字符")
            return response.model_dump()
    except ValueError as e:
        logger.warning(f"⚠️ 型号适配对话失败 - 参数错误 | 用户: {user_id} | 错误: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 型号适配对话失败 - 服务器错误 | 用户: {user_id} | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器错误：{str(e)}")


@app.post("/api/memory/add")
async def add_memory(
    session_id: str,
    content: str,
    category: str = "general",
    importance: float = 0.5,
    user_id: str = Depends(require_write_permission())
):
    """添加长期记忆（需要写入权限）"""
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
    user_id: str = Depends(require_read_permission())
):
    """搜索记忆（需要读取权限）"""
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


# ========== 意图识别和工具执行 ==========

@app.post("/api/intent/recognize")
async def recognize_intent(
    request: IntentRecognitionRequest,
    user_id: str = Depends(verify_token)
):
    """识别用户意图，判断是否需要展示表单"""
    logger.info(f"🎯 意图识别 | 用户: {user_id} | 消息: {request.message[:100]}")
    
    try:
        from services.intent_recognizer import get_intent_recognizer
        
        recognizer = get_intent_recognizer()
        result = recognizer.recognize(request.message)
        
        logger.info(f"✅ 意图识别完成 | 类型: {result['intent_type']} | 置信度: {result['confidence']}")
        
        return result
    except Exception as e:
        logger.error(f"❌ 意图识别失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"意图识别失败：{str(e)}")


@app.post("/api/tools/execute")
async def execute_tool(
    request: ExecuteToolRequest,
    user_id: str = Depends(require_write_permission())
):
    """执行工具（直接调用，不经过Agent）"""
    logger.info(f"🔧 执行工具 | 用户: {user_id} | 工具: {request.tool_name}")
    logger.debug(f"参数: {request.parameters}")
    
    try:
        # 获取 RAG 系统
        rag_system = get_enterprise_rag_system()
        
        # 根据工具名称找到对应的工具函数
        tool_result = await _execute_tool_by_name(request.tool_name, request.parameters)
        
        if not tool_result["success"]:
            logger.error(f"❌ 工具执行失败 | 原因: {tool_result['message']}")
            raise HTTPException(status_code=400, detail=tool_result["message"])
        
        logger.info(f"✅ 工具执行成功")
        return tool_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 工具执行失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"工具执行失败：{str(e)}")


async def _execute_tool_by_name(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """根据工具名称执行对应的工具函数
    
    Args:
        tool_name: 工具名称
        parameters: 参数字典
        
    Returns:
        执行结果
    """
    # 导入工具函数
    from tools.model_adapter import generate_insert_sql
    
    # 工具映射表
    tool_functions = {
        "generate_insert_sql": generate_insert_sql,
        # 可以添加更多工具
    }
    
    # 查找工具函数
    tool_func = tool_functions.get(tool_name)
    if not tool_func:
        return {
            "success": False,
            "message": f"未找到工具: {tool_name}"
        }
    
    try:
        # 调用工具函数
        result = tool_func.invoke(parameters)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"工具执行失败: {str(e)}"
        }


# ========== 用户管理路由 ==========

@app.post("/api/users/create")
async def create_user(
    request: UserCreateRequest,
    current_user_id: str = Depends(require_manage_users_permission())
):
    """创建新用户（需要用户管理权限）"""
    logger.info(f"👤 创建用户 | 操作者: {current_user_id} | 新用户ID: {request.user_id}")
    
    rag_system = get_enterprise_rag_system()
    auth_service = rag_system.auth_service
    
    # 检查当前用户是否有管理权限
    if not auth_service.has_permission(current_user_id, "manage_users"):
        logger.warning(f"⚠️ 权限不足 | 用户: {current_user_id} 尝试创建用户")
        raise HTTPException(status_code=403, detail="权限不足，需要管理员权限")
    
    try:
        auth_service.add_user(
            user_id=request.user_id,
            username=request.username,
            password=request.password,
            role=request.role
        )
        logger.info(f"✅ 用户创建成功 | 用户ID: {request.user_id}")
        return {
            "success": True,
            "message": "用户创建成功",
            "user_id": request.user_id
        }
    except ValueError as e:
        logger.warning(f"⚠️ 创建用户失败 | 原因: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 创建用户失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建失败：{str(e)}")


@app.get("/api/users/list")
async def list_users(current_user_id: str = Depends(require_manage_users_permission())):
    """获取所有用户列表（需要用户管理权限）"""
    logger.info(f"📋 获取用户列表 | 操作者: {current_user_id}")
    
    rag_system = get_enterprise_rag_system()
    auth_service = rag_system.auth_service
    
    # 检查当前用户是否有管理权限
    if not auth_service.has_permission(current_user_id, "manage_users"):
        logger.warning(f"⚠️ 权限不足 | 用户: {current_user_id} 尝试查看用户列表")
        raise HTTPException(status_code=403, detail="权限不足，需要管理员权限")
    
    users = auth_service.list_users()
    logger.info(f"✅ 返回 {len(users)} 个用户")
    return {
        "success": True,
        "total": len(users),
        "users": users
    }


@app.get("/api/users/{user_id}")
async def get_user_detail(
    user_id: str,
    current_user_id: str = Depends(verify_token)
):
    """获取用户详情（需要管理员权限或查看自己的信息）"""
    logger.info(f"🔍 获取用户详情 | 操作者: {current_user_id} | 目标用户: {user_id}")
    
    rag_system = get_enterprise_rag_system()
    auth_service = rag_system.auth_service
    
    # 检查权限：管理员或查看自己的信息
    if current_user_id != user_id and not auth_service.has_permission(current_user_id, "manage_users"):
        logger.warning(f"⚠️ 权限不足 | 用户: {current_user_id} 尝试查看用户: {user_id}")
        raise HTTPException(status_code=403, detail="权限不足")
    
    user = auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return {
        "success": True,
        "user": user
    }

# 编辑
@app.put("/api/users/{user_id}")
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user_id: str = Depends(require_manage_users_permission())
):
    """更新用户信息（需要用户管理权限）"""
    logger.info(f"✏️ 更新用户 | 操作者: {current_user_id} | 目标用户: {user_id}")
    
    rag_system = get_enterprise_rag_system()
    auth_service = rag_system.auth_service
    
    # 检查当前用户是否有管理权限
    if not auth_service.has_permission(current_user_id, "manage_users"):
        logger.warning(f"⚠️ 权限不足 | 用户: {current_user_id} 尝试更新用户: {user_id}")
        raise HTTPException(status_code=403, detail="权限不足，需要管理员权限")
    
    # 检查用户是否存在
    existing_user = auth_service.get_user_by_id(user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    try:
        # 更新密码
        if request.password is not None:
            auth_service.update_user_password(user_id, request.password)
            logger.info(f"🔑 密码已更新 | 用户: {user_id}")
        
        # 更新角色
        if request.role is not None and request.role != existing_user['role']:
            auth_service.update_user_role(user_id, request.role)
            logger.info(f"🔄 角色更新成功 | 用户: {user_id} | 新角色: {request.role}")
        
        logger.info(f"✅ 用户更新成功 | 用户ID: {user_id}")
        return {
            "success": True,
            "message": "用户更新成功"
        }
    except Exception as e:
        logger.error(f"❌ 更新用户失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新失败：{str(e)}")


@app.post("/api/users/delete")
async def delete_user(
    request: UserDeleteRequest,
    current_user_id: str = Depends(require_manage_users_permission())
):
    """删除用户（需要用户管理权限）"""
    logger.info(f"🗑️ 删除用户 | 操作者: {current_user_id} | 目标用户: {request.user_id}")
    
    rag_system = get_enterprise_rag_system()
    auth_service = rag_system.auth_service
    
    # 检查当前用户是否有管理权限
    if not auth_service.has_permission(current_user_id, "manage_users"):
        logger.warning(f"⚠️ 权限不足 | 用户: {current_user_id} 尝试删除用户: {request.user_id}")
        raise HTTPException(status_code=403, detail="权限不足，需要管理员权限")
    
    # 不能删除自己
    if request.user_id == current_user_id:
        logger.warning(f"⚠️ 非法操作 | 用户: {current_user_id} 尝试删除自己")
        raise HTTPException(status_code=400, detail="不能删除当前登录用户")
    
    # 检查用户是否存在
    existing_user = auth_service.get_user_by_id(request.user_id)
    if not existing_user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    try:
        auth_service.remove_user(request.user_id)
        logger.info(f"✅ 用户删除成功 | 用户ID: {request.user_id}")
        return {
            "success": True,
            "message": "用户删除成功"
        }
    except Exception as e:
        logger.error(f"❌ 删除用户失败 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")


# ========== 主函数 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
