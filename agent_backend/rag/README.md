# 企业知识库 RAG 系统

基于 LangChain 构建的企业级知识库问答系统，支持权限校验、飞书文档同步、智能对话、会话管理等功能。

## 📁 项目结构

```
agent_backend/rag/
├── __init__.py              # 包初始化
├── app.py                   # 主应用入口
├── config/                  # 配置模块
│   ├── __init__.py
│   └── settings.py          # 系统配置
├── models/                  # 数据模型
│   ├── __init__.py
│   └── schemas.py           # Pydantic 模型
├── services/                # 业务服务层
│   ├── __init__.py
│   ├── auth.py             # 权限认证服务
│   ├── knowledge_base.py   # 知识库管理服务
│   ├── rag_engine.py       # RAG 对话引擎
│   ├── session_manager.py  # 会话管理器
│   └── memory_store.py     # 记忆存储
├── stores/                  # 数据存储层
│   ├── __init__.py
│   └── vector_store.py     # 向量存储服务
├── integrations/            # 第三方集成
│   ├── __init__.py
│   └── feishu.py           # 飞书集成
├── utils/                   # 工具类
│   ├── __init__.py
│   └── helpers.py          # 通用工具函数
├── example_usage.py         # 使用示例
├── requirements.txt         # 依赖包列表
└── .env.example            # 环境配置示例
```

## 🚀 核心功能

### 1. 权限校验系统
- ✅ JWT Token 认证
- ✅ 用户角色管理（admin/user/guest）
- ✅ 权限控制（read/write/delete/manage_users）
- ✅ Token 黑名单机制

### 2. 知识库管理
- ✅ 支持多种知识源（本地文件、飞书文档、URL、手动录入）
- ✅ 文本自动分块（可配置 chunk_size 和 overlap）
- ✅ MD5 去重检测
- ✅ 批量导入支持
- ✅ 飞书云文档同步

### 3. RAG 对话引擎
- ✅ 基于 Chroma 向量数据库的相似度搜索
- ✅ 结合上下文的智能问答
- ✅ 多轮对话支持
- ✅ 参考来源展示
- ✅ 流式输出支持（待实现）

### 4. 会话管理
- ✅ 会话创建/切换/删除
- ✅ 会话历史记录
- ✅ 自动更新会话标题
- ✅ 会话导出（JSON/TXT格式）
- ✅ 多会话并发支持

### 5. 记忆管理
- ✅ 短期记忆（当前会话，可配置最大条数）
- ✅ 长期记忆（重要信息永久保存）
- ✅ 记忆分类存储
- ✅ 记忆搜索功能

## 📦 安装依赖

```bash
cd agent_backend/rag
pip install -r requirements.txt
```

## ⚙️ 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

**必要配置项：**
- `DASHSCOPE_API_KEY`: 通义千问 API 密钥
- `FEISHU_APP_ID`: 飞书应用 ID（如使用飞书集成）
- `FEISHU_APP_SECRET`: 飞书应用密钥
- `JWT_SECRET_KEY`: JWT 加密密钥（生产环境请修改）

## 💡 快速开始

### 方式一：运行示例脚本

```bash
python example_usage.py
```

示例脚本会演示完整的使用流程：
1. 用户登录
2. 添加知识到知识库
3. 创建会话
4. 多轮对话
5. 搜索知识
6. 获取会话列表
7. 导出会话
8. 添加长期记忆
9. 搜索记忆
10. 清空会话

### 方式二：代码调用

```python
from rag.app import get_enterprise_rag_system
from rag.models.schemas import ChatRequest

# 获取系统实例
rag_system = get_enterprise_rag_system()

# 1. 用户登录
login_result = rag_system.login("admin", "admin123")
user_id = login_result["user_id"]

# 2. 添加知识
rag_system.add_knowledge(
    title="产品说明",
    content="这是产品详细介绍...",
    source_type="manual",
    created_by=user_id
)

# 3. 创建会话
session_result = rag_system.create_session(
    user_id=user_id,
    title="产品咨询"
)
session_id = session_result["session_id"]

# 4. 开始对话
request = ChatRequest(
    session_id=session_id,
    message="这个产品有什么特点？"
)
response = rag_system.chat(request)
print(f"AI 回答：{response.content}")
print(f"参考来源：{response.sources}")

# 5. 导出会话
export_result = rag_system.export_session(
    session_id=session_id,
    format="json"
)
```

## 🔧 主要组件说明

### 1. AuthService - 权限认证服务

```python
from rag.services.auth import get_auth_service

auth_service = get_auth_service()

# 用户认证
user = auth_service.authenticate_user("username", "password")

# 创建 Token
token = auth_service.create_access_token(user_id)

# 验证 Token
payload = auth_service.verify_token(token)

# 检查权限
has_permission = auth_service.has_permission(user_id, "write")
```

### 2. KnowledgeBaseService - 知识库管理

```python
from rag.services.knowledge_base import KnowledgeBaseService

kb_service = KnowledgeBaseService()

# 添加文档
doc = kb_service.add_document(
    title="文档标题",
    content="文档内容...",
    source_type="feishu",
    created_by="user_id"
)

# 搜索文档
results = kb_service.search_documents(
    query="搜索关键词",
    limit=5
)

# 删除文档
kb_service.delete_document(doc_id)
```

### 3. RAGEngine - RAG 对话引擎

```python
from rag.services.rag_engine import RAGEngine

rag_engine = RAGEngine()

# 执行查询
response = rag_engine.query(
    question="用户问题",
    chat_history=[{"role": "user", "content": "历史问题"}],
    include_sources=True
)

print(response["answer"])
print(response["sources"])
```

### 4. SessionManager - 会话管理

```python
from rag.services.session_manager import SessionManager

session_mgr = SessionManager()

# 创建会话
session = session_mgr.create_session(
    user_id="user_001",
    title="新会话"
)

# 添加消息
session_mgr.add_message_to_session(
    session_id=session.session_id,
    role="user",
    content="你好"
)

# 获取历史
history = session_mgr.get_session_history(session_id)

# 导出会话
json_content = session_mgr.export_session(session_id, format="json")
```

### 5. FeishuClient - 飞书集成

```python
from rag.integrations.feishu import FeishuClient

feishu = FeishuClient()

# 同步文件夹中的所有文档
docs = feishu.sync_folder_documents(
    folder_token="folder_token_here",
    created_by="user_id"
)

# 获取单个文档内容
doc_content = feishu.get_docx_content(docx_token)
```

## 🎯 实际应用场景示例

### 场景 1: 客服知识库问答
```python
# 添加产品 FAQ 到知识库
rag_system.add_knowledge(
    title="常见问题解答",
    content="Q: 如何退货？A: 七天内无理由退货...",
    source_type="manual",
    created_by="admin"
)

# 客户咨询
response = rag_system.chat(ChatRequest(
    session_id=session_id,
    message="我可以退货吗？"
))
```

### 场景 2: 企业内部文档助手
```python
# 同步飞书上的公司制度文档
rag_system.sync_feishu_documents(
    folder_token="company_policies_folder",
    created_by="hr_admin"
)

# 员工询问请假流程
response = rag_system.chat(ChatRequest(
    session_id=session_id,
    message="请假流程是什么？"
))
```

### 场景 3: 技术支持系统
```python
# 添加技术文档
rag_system.add_knowledge(
    title="产品技术手册",
    content="技术参数：... 使用方法：...",
    source_type="file",
    created_by="tech_team"
)

# 技术支持人员查询
response = rag_system.chat(ChatRequest(
    session_id=session_id,
    message="设备显示错误代码 E03 怎么处理？"
))
```

## 📝 注意事项

1. **API Key 配置**: 确保在 `.env` 文件中配置了正确的 `DASHSCOPE_API_KEY`
2. **飞书集成**: 如需使用飞书功能，需要先在飞书开放平台创建应用
3. **向量数据库**: Chroma 会自动在 `./data/chroma_db` 目录创建数据库
4. **会话持久化**: 当前会话数据存储在内存中，重启会丢失，生产环境建议使用数据库
5. **性能优化**: 大量文档时建议调整 `CHUNK_SIZE` 和 `SIMILARITY_TOP_K` 参数

## 🔮 后续优化方向

- [ ] 添加 FastAPI 接口层，提供 RESTful API
- [ ] 实现流式输出
- [ ] 接入真实数据库（PostgreSQL/MongoDB）
- [ ] 添加 Web 管理界面
- [ ] 支持更多文档格式（PDF、Word、Excel）
- [ ] 实现更复杂的权限系统（RBAC）
- [ ] 添加日志和监控
- [ ] 性能优化和缓存机制

## 📄 License

MIT License