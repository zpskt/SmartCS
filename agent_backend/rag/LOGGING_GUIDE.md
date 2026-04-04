# 日志系统使用指南

## 📋 概述

本项目已集成完善的日志系统，支持：
- ✅ 控制台实时输出
- ✅ 文件持久化存储（自动轮转）
- ✅ 分级日志记录（DEBUG/INFO/WARNING/ERROR）
- ✅ 独立的错误日志和 API 日志
- ✅ 详细的请求/响应追踪

## 📁 日志文件位置

所有日志文件存储在 `logs/` 目录下：

```
logs/
├── app.log          # 所有应用日志（INFO 及以上级别）
├── error.log        # 仅错误日志（ERROR 及以上级别）
└── api.log          # API 请求日志
```

## 🔧 配置说明

在 `.env` 文件中配置日志参数：

```bash
# 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# 日志目录
LOG_DIR=./logs

# 单个日志文件最大大小（字节），默认 10MB
LOG_FILE_MAX_BYTES=10485760

# 日志文件备份数量，默认保留 5 个
LOG_BACKUP_COUNT=5
```

## 📊 日志级别说明

| 级别 | 用途 | 示例场景 |
|------|------|----------|
| **DEBUG** | 调试信息 | 详细的处理步骤、变量值 |
| **INFO** | 一般信息 | 用户登录、创建会话、对话完成 |
| **WARNING** | 警告信息 | 登录失败、资源不存在 |
| **ERROR** | 错误信息 | 服务器异常、数据库错误 |
| **CRITICAL** | 严重错误 | 系统崩溃、关键服务不可用 |

## 🎯 日志格式

```
2024-04-04 10:30:15,123 - rag_system.api_server - INFO - [api_server.py:45] - 📨 请求开始 | POST http://localhost:8000/api/chat | 客户端: 127.0.0.1
```

格式组成：
- `时间戳` - `模块名` - `日志级别` - `[文件名:行号]` - `日志消息`

## 💡 使用示例

### 1. 在代码中使用日志

```python
from utils.logger import get_logger

# 获取模块专用的日志记录器
logger = get_logger("your_module_name")

# 不同级别的日志
logger.debug("调试信息：详细的数据处理过程")
logger.info("一般信息：用户登录成功")
logger.warning("警告信息：资源未找到")
logger.error("错误信息：数据库连接失败", exc_info=True)
logger.critical("严重错误：系统崩溃")
```

### 2. 查看实时日志

```bash
# 查看所有日志
tail -f logs/app.log

# 仅查看错误日志
tail -f logs/error.log

# 仅查看 API 请求日志
tail -f logs/api.log

# 过滤特定关键词
tail -f logs/app.log | grep "对话"
```

### 3. 调整日志级别

**开发环境**（显示更多详细信息）：
```bash
LOG_LEVEL=DEBUG
```

**生产环境**（仅记录重要信息）：
```bash
LOG_LEVEL=WARNING
```

## 🔍 日志内容示例

### API 请求日志
```
📨 请求开始 | POST /api/chat | 客户端: 127.0.0.1
💬 对话请求 | 用户: admin | 会话: sess_123
✅ 对话成功 | 响应长度: 256 字符
✅ 请求完成 | POST /api/chat | 状态码: 200 | 耗时: 1.234s
```

### 错误日志
```
❌ 登录失败 | 用户名: testuser | 原因: 用户名或密码错误
⚠️ 错误响应 | POST /api/auth/login | 状态码: 401 | 耗时: 0.012s
💥 服务器错误 | POST /api/chat | 耗时: 2.345s | 错误: Database connection timeout
```

### 业务日志
```
🚀 初始化企业 RAG 系统...
✅ 企业 RAG 系统初始化完成
🔐 用户登录尝试 | 用户名: admin
✅ 登录成功 | 用户ID: user_001 | 角色: admin
📝 添加知识 | 用户: admin | 标题: 产品手册 | 类型: document
🆕 创建会话 | 用户: admin | 标题: 新会话
```

## 🛠️ 故障排查

### 问题 1：看不到日志文件

**解决方案**：
```bash
# 检查日志目录是否存在
ls -la logs/

# 如果不存在，手动创建
mkdir -p logs

# 检查权限
chmod 755 logs/
```

### 问题 2：日志太多，磁盘空间不足

**解决方案**：
```bash
# 减小日志文件大小限制
LOG_FILE_MAX_BYTES=5242880  # 5MB

# 减少备份数量
LOG_BACKUP_COUNT=3

# 清理旧日志
rm logs/*.log.*
```

### 问题 3：想看到更详细的调试信息

**解决方案**：
```bash
# 修改 .env 文件
LOG_LEVEL=DEBUG

# 重启服务
```

## 📈 最佳实践

1. **开发环境**：使用 `DEBUG` 级别，便于排查问题
2. **测试环境**：使用 `INFO` 级别，平衡性能和可观测性
3. **生产环境**：使用 `WARNING` 或 `ERROR` 级别，减少 I/O 开销
4. **定期清理**：设置定时任务清理过期日志
5. **监控告警**：对 `error.log` 进行监控，及时发现异常

## 🔗 相关文档

- [Python logging 官方文档](https://docs.python.org/3/library/logging.html)
- [FastAPI 中间件文档](https://fastapi.tiangolo.com/tutorial/middleware/)
