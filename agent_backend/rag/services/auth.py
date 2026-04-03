"""
权限认证服务
实现用户认证、授权和访问控制
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from jose import jwt, JWTError
from config.settings import settings


class AuthService:
    """认证授权服务类"""
    
    def __init__(self):
        """初始化认证服务"""
        self.secret_key = settings.JWT_SECRET_KEY or secrets.token_urlsafe(32)
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        # 模拟用户数据库（实际应该连接数据库）
        self.users_db: Dict[str, dict] = {
            "admin": {
                "user_id": "admin",
                "username": "admin",
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "delete", "manage_users"]
            },
            "zpaskt": {
                "user_id": "zpaskt",
                "username": "zpaskt",
                "password_hash": self._hash_password("user123"),
                "role": "user",
                "permissions": ["read", "write"]
            }
        }
        
        # Token 黑名单
        self.token_blacklist: set = set()
    
    def _hash_password(self, password: str) -> str:
        """
        密码哈希
        
        :param password: 原始密码
        :return: 哈希后的密码
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        :param password: 待验证密码
        :param hashed_password: 已存储的哈希密码
        :return: 是否匹配
        """
        return self._hash_password(password) == hashed_password
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """
        认证用户
        
        :param username: 用户名
        :param password: 密码
        :return: 用户信息（认证失败返回 None）
        """
        user = self.users_db.get(username)
        if not user:
            return None
        if not self.verify_password(password, user["password_hash"]):
            return None
        return user
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        创建访问令牌
        
        :param user_id: 用户 ID
        :param expires_delta: 过期时间增量（可选）
        :return: JWT token
        """
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """
        验证令牌
        
        :param token: JWT token
        :return: payload 信息（验证失败返回 None）
        """
        if token in self.token_blacklist:
            return None
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            if not user_id:
                return None
            return {"user_id": user_id, "payload": payload}
        except JWTError:
            return None
    
    def revoke_token(self, token: str):
        """
        撤销令牌
        
        :param token: JWT token
        """
        self.token_blacklist.add(token)
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """
        获取用户权限列表
        
        :param user_id: 用户 ID
        :return: 权限列表
        """
        user = self.users_db.get(user_id)
        if not user:
            return []
        return user.get("permissions", [])
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        检查用户是否有指定权限
        
        :param user_id: 用户 ID
        :param permission: 权限名称
        :return: 是否有权限
        """
        permissions = self.get_user_permissions(user_id)
        return permission in permissions
    
    def add_user(self, user_id: str, username: str, password: str, role: str = "user"):
        """
        添加新用户
        
        :param user_id: 用户 ID
        :param username: 用户名
        :param password: 密码
        :param role: 角色
        """
        if user_id in self.users_db:
            raise ValueError(f"用户 {user_id} 已存在")
        
        self.users_db[user_id] = {
            "user_id": user_id,
            "username": username,
            "password_hash": self._hash_password(password),
            "role": role,
            "permissions": self._get_default_permissions(role)
        }
    
    def _get_default_permissions(self, role: str) -> List[str]:
        """
        获取角色的默认权限
        
        :param role: 角色名称
        :return: 权限列表
        """
        if role == "admin":
            return ["read", "write", "delete", "manage_users"]
        elif role == "user":
            return ["read", "write"]
        else:
            return ["read"]
    
    def remove_user(self, user_id: str):
        """
        删除用户
        
        :param user_id: 用户 ID
        """
        if user_id in self.users_db:
            del self.users_db[user_id]


# 单例模式
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """
    获取认证服务实例（单例）
    
    :return: AuthService 实例
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
