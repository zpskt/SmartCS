"""
权限管理依赖注入
提供 FastAPI 依赖函数，用于接口级别的权限校验
"""
from typing import List, Optional
from fastapi import Depends, HTTPException, Header
from functools import wraps
from services.auth import get_auth_service
from utils.logger import get_logger

logger = get_logger("permissions")


def require_roles(required_roles: List[str]):
    """
    角色校验装饰器/依赖
    
    :param required_roles: 允许的角色列表（满足任一即可）
    :return: 依赖函数
    
    使用示例:
        @app.get("/admin-only")
        async def admin_endpoint(user_id: str = Depends(require_roles(["admin"]))):
            ...
    """
    async def role_checker(x_authorization: Optional[str] = Header(None)):
        if not x_authorization:
            raise HTTPException(status_code=401, detail="缺少认证令牌")
        
        auth_service = get_auth_service()
        token = x_authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        
        user_id = payload["user_id"]
        
        # 检查角色
        user_role = auth_service.get_user_role(user_id)
        if not user_role or user_role not in required_roles:
            logger.warning(f"⚠️ 角色校验失败 | 用户: {user_id} | 当前角色: {user_role} | 需要角色: {required_roles}")
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，需要以下角色之一: {', '.join(required_roles)}"
            )
        
        logger.debug(f"✅ 角色校验通过 | 用户: {user_id} | 角色: {user_role}")
        return user_id
    
    return role_checker


def require_permissions(required_permissions: List[str], require_all: bool = False):
    """
    权限校验装饰器/依赖
    
    :param required_permissions: 需要的权限列表
    :param require_all: 是否需要所有权限（True=全部，False=任一）
    :return: 依赖函数
    
    使用示例:
        # 需要任意一个权限
        @app.get("/endpoint")
        async def endpoint(user_id: str = Depends(require_permissions(["read", "write"]))):
            ...
        
        # 需要所有权限
        @app.post("/endpoint")
        async def endpoint(user_id: str = Depends(require_permissions(["read", "write"], require_all=True))):
            ...
    """
    async def permission_checker(x_authorization: Optional[str] = Header(None)):
        if not x_authorization:
            raise HTTPException(status_code=401, detail="缺少认证令牌")
        
        auth_service = get_auth_service()
        token = x_authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        
        user_id = payload["user_id"]
        
        # 检查权限
        if require_all:
            has_permission = auth_service.has_all_permissions(user_id, required_permissions)
        else:
            has_permission = auth_service.has_any_permission(user_id, required_permissions)
        
        if not has_permission:
            logger.warning(
                f"⚠️ 权限校验失败 | 用户: {user_id} | "
                f"需要权限: {required_permissions} (全部={require_all})"
            )
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，需要以下权限: {', '.join(required_permissions)}"
            )
        
        logger.debug(f"✅ 权限校验通过 | 用户: {user_id} | 权限: {required_permissions}")
        return user_id
    
    return permission_checker


def require_role_and_permissions(
    required_roles: Optional[List[str]] = None,
    required_permissions: Optional[List[str]] = None,
    require_all_permissions: bool = False
):
    """
    综合角色和权限校验
    
    :param required_roles: 需要的角色列表（满足任一即可）
    :param required_permissions: 需要的权限列表
    :param require_all_permissions: 是否需要所有权限
    :return: 依赖函数
    
    使用示例:
        @app.post("/advanced-feature")
        async def advanced_feature(
            user_id: str = Depends(require_role_and_permissions(
                required_roles=["admin", "editor"],
                required_permissions=["write", "publish"]
            ))
        ):
            ...
    """
    async def combined_checker(x_authorization: Optional[str] = Header(None)):
        if not x_authorization:
            raise HTTPException(status_code=401, detail="缺少认证令牌")
        
        auth_service = get_auth_service()
        token = x_authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        
        user_id = payload["user_id"]
        
        # 综合检查
        has_access = auth_service.check_role_and_permission(
            user_id=user_id,
            required_roles=required_roles,
            required_permissions=required_permissions,
            require_all_permissions=require_all_permissions
        )
        
        if not has_access:
            logger.warning(
                f"⚠️ 综合权限校验失败 | 用户: {user_id} | "
                f"需要角色: {required_roles} | 需要权限: {required_permissions}"
            )
            
            error_parts = []
            if required_roles:
                error_parts.append(f"角色: {', '.join(required_roles)}")
            if required_permissions:
                error_parts.append(f"权限: {', '.join(required_permissions)}")
            
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，需要 {', '.join(error_parts)}"
            )
        
        logger.debug(f"✅ 综合权限校验通过 | 用户: {user_id}")
        return user_id
    
    return combined_checker


# ========== 预定义的常用权限依赖 ==========

def require_admin():
    """管理员权限"""
    return require_roles(["admin"])


def require_read_permission():
    """读取权限"""
    return require_permissions(["read"])


def require_write_permission():
    """写入权限"""
    return require_permissions(["write"])


def require_delete_permission():
    """删除权限"""
    return require_permissions(["delete"])


def require_manage_users_permission():
    """用户管理权限"""
    return require_permissions(["manage_users"])


def require_knowledge_management():
    """知识库管理权限（需要 write 或 manage_knowledge）"""
    return require_permissions(["write", "manage_knowledge"])
