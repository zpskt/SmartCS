# 快速开始指南

## 📋 前置要求

- Python 3.8+
- 通义千问 API Key（DashScope）
- 飞书应用配置（可选，如需飞书集成）

## 🚀 快速安装

### 1. 进入项目目录
```bash
cd agent_backend/rag
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制环境配置示例文件
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
vim .env  # 或使用其他编辑器
```

**必须配置的变量：**
```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 4. 运行示例
```bash
python example_usage.py
```

## 💡 使用方式

### 方式一：运行脚本（推荐新手）
```bash
# 使用启动脚本（有交互式菜单）
./start.sh

# 或直接运行示例
python example_usage.py
```

### 方式二：代码调用
```python
from rag.app import get_enterprise_rag_system
from rag.models.schemas import ChatRequest

# 初始化系统
rag = get_enterprise_rag_system()

# 登录
result = rag.login("admin", "admin123")
user_id = result["user_id"]

# 创建会话
session = rag.create_session(user_id, "我的会话")
session_id = session["session_id"]

# 对话
response = rag.chat(ChatRequest(
    session_id=session_id,
    message="你好"
))
print(response.content)
```

### 方式三：API 服务器
```bash
# 启动 FastAPI 服务器
python api_server.py

# 访问 http://localhost:8000/docs 查看 API 文档
```

## 📁 项目结构说明

```
rag/
├── app.py                    # 主应用入口，统一调用接口
├── config/                   # 配置管理
│   └── settings.py          # 系统配置类
├── models/                   # 数据模型定义
│   └── schemas.py           # Pydantic 模型
├── services/                 # 核心业务服务
│   ├── auth.py             # 权限认证
│   ├── knowledge_base.py   # 知识库管理
│   ├── rag_engine.py       # RAG 对话引擎
│   ├── session_manager.py  # 会话管理
│   └── memory_store.py     # 记忆存储
├── stores/                   # 数据存储层
│   └── vector_store.py     # 向量数据库
├── integrations/             # 第三方集成
│   └── feishu.py           # 飞书集成
├── utils/                    # 工具函数
│   └── helpers.py
├── example_usage.py          # 使用示例
├── test_app.py              # 单元测试
├── api_server.py            # FastAPI 接口层
└── requirements.txt         # 依赖列表
```

## 🔧 核心功能速览

### 1. 用户认证
```python
# 登录
login_result = rag.login("admin", "admin123")

# 检查权限
has_permission = rag.check_permission("admin", "write")
```

### 2. 知识库管理
```python
# 添加知识
rag.add_knowledge(
    title="产品说明",
    content="这是产品详细介绍...",
    source_type="manual",
    created_by="admin"
)

# 搜索知识
results = rag.search_knowledge("产品功能", limit=5)
```

### 3. 会话对话
```python
# 创建会话
session = rag.create_session("user_001", "产品咨询")

# 多轮对话
response = rag.chat(ChatRequest(
    session_id=session["session_id"],
    message="这个产品多少钱？"
))
```

### 4. 会话管理
```python
# 获取会话列表
sessions = rag.get_session_list("user_001")

# 导出会话
export = rag.export_session(session_id, format="json")

# 删除会话
rag.delete_session(session_id)
```

### 5. 记忆管理
```python
# 添加长期记忆
rag.add_long_term_memory(
    session_id=session_id,
    content="用户关心价格",
    category="preference",
    importance=0.8
)

# 搜索记忆
memories = rag.search_memories(session_id, "价格")
```

### 6. 飞书集成（可选）
```python
# 同步飞书文档
rag.sync_feishu_documents(
    folder_token="your_folder_token",
    created_by="admin"
)
```

## 🎯 典型应用场景

### 场景 1：客服自动问答
```python
# 1. 导入 FAQ 文档
rag.add_knowledge(
    title="常见问题解答",
    content=open("faq.txt").read(),
    source_type="file",
    created_by="admin"
)

# 2. 客户咨询
response = rag.chat(ChatRequest(
    session_id=session_id,
    message="如何退货？"
))
```

### 场景 2：企业文档助手
```python
# 1. 同步公司制度文档
rag.sync_feishu_documents(
    folder_token="company_policies",
    created_by="hr"
)

# 2. 员工查询
response = rag.chat(ChatRequest(
    session_id=session_id,
    message="请假流程是什么？"
))
```

### 场景 3：技术支持系统
```python
# 1. 导入技术手册
rag.add_knowledge(
    title="产品技术手册",
    content=technical_doc,
    source_type="manual",
    created_by="tech_team"
)

# 2. 技术支持
response = rag.chat(ChatRequest(
    session_id=session_id,
    message="错误代码 E03 怎么处理？"
))
```

## ⚠️ 常见问题

### Q1: 提示缺少依赖包
```bash
# 重新安装依赖
pip install -r requirements.txt --upgrade
```

### Q2: API Key 无效
- 检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确
- 确保没有多余的空格或引号
- 重启 Python 环境

### Q3: 飞书集成失败
- 检查是否正确配置飞书应用 ID 和密钥
- 确认文件夹 token 正确
- 检查网络连通性

### Q4: 向量数据库初始化失败
```bash
# 删除旧的向量数据库，重新创建
rm -rf data/chroma_db/*
python example_usage.py
```

## 📊 性能优化建议

1. **调整分块大小**: 根据文档类型调整 `CHUNK_SIZE`
   - 技术文档：500-800
   - FAQ: 200-400
   - 长文档：800-1000

2. **优化检索数量**: 调整 `SIMILARITY_TOP_K`
   - 精确问答：3-5
   - 综合查询：5-10

3. **内存管理**: 定期清理不需要的会话
   ```python
   rag.delete_session(session_id)
   ```

## 🔮 下一步

- [ ] 查看 `example_usage.py` 了解完整示例
- [ ] 运行 `test_app.py` 进行功能测试
- [ ] 启动 API 服务器开发前端界面
- [ ] 根据业务需求定制功能

## 📞 技术支持

如有问题，请查看：
1. README.md 详细文档
2. 各模块的代码注释
3. LangChain 官方文档
