# 权限管理系统 - 快速开始

## 🚀 已实现的功能

### 1. 核心组件

✅ **AuthService 增强** (`services/auth.py`)
- `has_any_permission()` - 检查是否拥有任意一个权限
- `has_all_permissions()` - 检查是否拥有所有权限
- `get_user_role()` - 获取用户角色
- `check_role_and_permission()` - 综合角色和权限校验

✅ **权限依赖注入** (`utils/permissions.py`)
- `require_roles()` - 基于角色的权限控制
- `require_permissions()` - 基于权限的控制
- `require_role_and_permissions()` - 综合控制
- 预定义常用依赖：`require_admin()`, `require_read_permission()` 等

✅ **API 接口权限保护** (`api_server.py`)
- 所有接口已应用相应的权限控制
- 详细的权限说明文档

### 2. 权限体系

| 角色 | 权限 | 适用场景 |
|------|------|----------|
| admin | read, write, delete, manage_users | 系统管理员 |
| user | read, write | 普通用户 |
| guest | read | 访客（只读） |

## 📝 使用示例

### 示例 1：简单的权限控制

```python
from fastapi import Depends
from utils.permissions import require_write_permission

@app.post("/api/resource")
async def create_resource(user_id: str = Depends(require_write_permission())):
    # 只有拥有 write 权限的用户才能访问
    return {"message": f"用户 {user_id} 创建成功"}
```

### 示例 2：多权限要求（满足任一）

```python
from utils.permissions import require_permissions

@app.get("/api/resource")
async def get_resource(
    user_id: str = Depends(require_permissions(["read", "write"]))
):
    # 拥有 read 或 write 权限都可以访问
    return {"message": "访问成功"}
```

### 示例 3：需要所有权限

```python
@app.post("/api/advanced")
async def advanced_operation(
    user_id: str = Depends(require_permissions(["read", "write"], require_all=True))
):
    # 必须同时拥有 read 和 write 权限
    return {"message": "高级操作成功"}
```

### 示例 4：角色限制

```python
from utils.permissions import require_roles

@app.delete("/api/admin-only")
async def admin_only(user_id: str = Depends(require_roles(["admin"]))):
    # 只有 admin 角色可以访问
    return {"message": "管理员操作成功"}
```

### 示例 5：综合权限控制

```python
from utils.permissions import require_role_and_permissions

@app.post("/api/sync-feishu")
async def sync_feishu(
    user_id: str = Depends(require_role_and_permissions(
        required_roles=["admin"],
        required_permissions=["write", "manage_knowledge"]
    ))
):
    # 需要 admin 角色，并且有 write 和 manage_knowledge 权限
    return {"message": "同步成功"}
```

## 🔧 扩展示例

### 添加新角色

在 `services/auth.py` 中修改 `_get_default_permissions`：

```python
def _get_default_permissions(self, role: str) -> List[str]:
    if role == "admin":
        return ["read", "write", "delete", "manage_users"]
    elif role == "editor":  # 新增编辑者角色
        return ["read", "write", "publish"]
    elif role == "viewer":  # 新增查看者角色
        return ["read"]
    else:
        return ["read"]
```

### 添加新权限类型

1. 在数据库中添加用户的权限列表
2. 创建新的权限依赖函数：

```python
# utils/permissions.py
def require_publish_permission():
    """发布权限"""
    return require_permissions(["publish"])
```

3. 在接口中使用：

```python
@app.post("/api/publish")
async def publish_content(user_id: str = Depends(require_publish_permission())):
    return {"message": "发布成功"}
```

## 🧪 测试

运行测试脚本验证权限系统：

```bash
# 1. 启动后端服务
python api_server.py

# 2. 在另一个终端运行测试
python test_permissions.py
```

## 📊 当前 API 权限分配

### 知识库管理
- 读取类接口 → `read` 权限
- 写入类接口 → `write` 权限
- 删除类接口 → `delete` 权限
- 飞书同步 → `admin` + (`write`, `manage_knowledge`)

### 会话管理
- 查看类 → `read` 权限
- 创建/修改 → `write` 权限
- 删除 → `delete` 权限

### 用户管理
- 所有管理接口 → `manage_users` 权限
- 查看自己信息 → 只需登录

### 对话功能
- 所有对话接口 → `write` 权限

## ⚠️ 注意事项

1. **Token 传递**：前端需要在请求头中携带 `X-Authorization: Bearer <token>`
2. **权限更新**：修改用户权限后，用户需要重新登录以刷新 token
3. **最小权限原则**：给用户分配完成任务所需的最小权限
4. **日志记录**：所有权限校验失败都会记录日志，便于审计

## 📖 更多文档

- 完整权限指南：[PERMISSION_GUIDE.md](PERMISSION_GUIDE.md)
- 认证服务源码：[services/auth.py](services/auth.py)
- 权限工具源码：[utils/permissions.py](utils/permissions.py)

## 🎯 下一步

1. 根据业务需求调整角色和权限配置
2. 在前端实现权限感知的 UI（隐藏无权访问的按钮）
3. 添加权限审计日志
4. 实现动态权限管理界面
