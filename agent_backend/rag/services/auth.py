"""
权限认证服务
实现用户认证、授权和访问控制
"""
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from jose import jwt, JWTError
import psycopg
from psycopg import rows
from config.settings import settings


class AuthService:
    """认证授权服务类"""
    
    def __init__(self):
        """初始化认证服务"""
        self.secret_key = settings.JWT_SECRET_KEY or secrets.token_urlsafe(32)
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        # PostgreSQL 数据库配置
        self.db_config = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
            'dbname': settings.POSTGRES_DATABASE
        }
        
        # 初始化数据库
        self._init_database()
        
        # Token 黑名单
        self.token_blacklist: set = set()
    
    def _get_connection(self):
        """获取数据库连接"""
        conn = psycopg.connect(**self.db_config, autocommit=True)
        return conn
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                # 创建用户表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(255) PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        role VARCHAR(50) DEFAULT 'user',
                        permissions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 检查是否存在默认用户，不存在则创建
                cursor.execute('SELECT COUNT(*) FROM users')
                result = cursor.fetchone()
                count = result['count'] if result else 0
                
                if count == 0:
                    # 插入默认管理员用户
                    cursor.execute(
                        'INSERT INTO users (user_id, username, password, role, permissions) VALUES (%s, %s, %s, %s, %s)',
                        ('admin', 'admin', 'admin123', 'admin', '["read", "write", "delete", "manage_users"]')
                    )
                    # 插入默认普通用户
                    cursor.execute(
                        'INSERT INTO users (user_id, username, password, role, permissions) VALUES (%s, %s, %s, %s, %s)',
                        ('zpaskt', 'zpaskt', 'user123', 'user', '["read", "write"]')
                    )
                    conn.commit()
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """
        认证用户
        
        :param username: 用户名
        :param password: 密码
        :return: 用户信息（认证失败返回 None）
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute(
                    'SELECT user_id, username, password, role, permissions FROM users WHERE username = %s',
                    (username,)
                )
                user = cursor.fetchone()
                
                if not user:
                    return None
                
                # 验证明文密码
                if user['password'] != password:
                    return None
                
                # 解析权限JSON字符串为列表
                permissions = json.loads(user['permissions'])
                
                return {
                    "user_id": user['user_id'],
                    "username": user['username'],
                    "role": user['role'],
                    "permissions": permissions
                }
        finally:
            conn.close()
    
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
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute('SELECT permissions FROM users WHERE user_id = %s', (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    return []
                
                return json.loads(user['permissions'])
        finally:
            conn.close()
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        检查用户是否有指定权限
        
        :param user_id: 用户 ID
        :param permission: 权限名称
        :return: 是否有权限
        """
        permissions = self.get_user_permissions(user_id)
        return permission in permissions
    
    def has_any_permission(self, user_id: str, permissions: List[str]) -> bool:
        """
        检查用户是否拥有任意一个指定权限
        
        :param user_id: 用户 ID
        :param permissions: 权限列表
        :return: 是否拥有任意权限
        """
        user_permissions = self.get_user_permissions(user_id)
        return any(p in user_permissions for p in permissions)
    
    def has_all_permissions(self, user_id: str, permissions: List[str]) -> bool:
        """
        检查用户是否拥有所有指定权限
        
        :param user_id: 用户 ID
        :param permissions: 权限列表
        :return: 是否拥有所有权限
        """
        user_permissions = self.get_user_permissions(user_id)
        return all(p in user_permissions for p in permissions)
    
    def get_user_role(self, user_id: str) -> Optional[str]:
        """
        获取用户角色
        
        :param user_id: 用户 ID
        :return: 角色名称
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute('SELECT role FROM users WHERE user_id = %s', (user_id,))
                user = cursor.fetchone()
                return user['role'] if user else None
        finally:
            conn.close()
    
    def check_role_and_permission(
        self,
        user_id: str,
        required_roles: Optional[List[str]] = None,
        required_permissions: Optional[List[str]] = None,
        require_all_permissions: bool = False
    ) -> bool:
        """
        综合检查用户角色和权限
        
        :param user_id: 用户 ID
        :param required_roles: 需要的角色列表（满足任一即可）
        :param required_permissions: 需要的权限列表
        :param require_all_permissions: 是否需要所有权限（True=全部，False=任一）
        :return: 是否通过检查
        """
        # 检查角色
        if required_roles:
            user_role = self.get_user_role(user_id)
            if not user_role or user_role not in required_roles:
                return False
        
        # 检查权限
        if required_permissions:
            if require_all_permissions:
                return self.has_all_permissions(user_id, required_permissions)
            else:
                return self.has_any_permission(user_id, required_permissions)
        
        return True
    
    def add_user(self, user_id: str, username: str, password: str, role: str = "user"):
        """
        添加新用户
        
        :param user_id: 用户 ID
        :param username: 用户名
        :param password: 密码（明文存储）
        :param role: 角色
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                # 检查用户是否已存在
                cursor.execute('SELECT user_id FROM users WHERE user_id = %s OR username = %s', (user_id, username))
                if cursor.fetchone():
                    raise ValueError(f"用户 {user_id} 或用户名 {username} 已存在")
                
                permissions = json.dumps(self._get_default_permissions(role), ensure_ascii=False)
                cursor.execute(
                    'INSERT INTO users (user_id, username, password, role, permissions) VALUES (%s, %s, %s, %s, %s)',
                    (user_id, username, password, role, permissions)
                )
        except psycopg.errors.UniqueViolation:
            raise ValueError(f"用户 {user_id} 或用户名 {username} 已存在")
        finally:
            conn.close()
    
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
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM users WHERE user_id = %s', (user_id,))
        finally:
            conn.close()
    
    def update_user_password(self, user_id: str, new_password: str):
        """
        更新用户密码
        
        :param user_id: 用户 ID
        :param new_password: 新密码（明文）
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'UPDATE users SET password = %s, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s',
                    (new_password, user_id)
                )
                return cursor.rowcount > 0
        finally:
            conn.close()
    
    def update_user_role(self, user_id: str, new_role: str):
        """
        更新用户角色
        
        :param user_id: 用户 ID
        :param new_role: 新角色
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                # 先检查用户是否存在
                cursor.execute('SELECT role, permissions FROM users WHERE user_id = %s', (user_id,))
                existing_user = cursor.fetchone()
                if not existing_user:
                    return False
                
                # 根据新角色获取权限列表
                new_permissions = self._get_default_permissions(new_role)
                # 将权限列表转换为JSON字符串
                permissions_json = json.dumps(new_permissions, ensure_ascii=False)
                
                print(f"DEBUG: 更新角色 | 用户: {user_id} | 旧角色: {existing_user['role']} | 新角色: {new_role}")
                print(f"DEBUG: 旧权限: {existing_user['permissions']} | 新权限: {permissions_json}")
                
                # 更新角色和权限
                cursor.execute(
                    'UPDATE users SET role = %s, permissions = %s, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s',
                    (new_role, permissions_json, user_id)
                )
                
                # 验证更新是否成功
                cursor.execute('SELECT role, permissions FROM users WHERE user_id = %s', (user_id,))
                updated_user = cursor.fetchone()
                print(f"DEBUG: 更新后 | 角色: {updated_user['role']} | 权限: {updated_user['permissions']}")
                
                return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """
        根据用户ID获取用户信息
        
        :param user_id: 用户 ID
        :return: 用户信息
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute(
                    'SELECT user_id, username, role, permissions, created_at, updated_at FROM users WHERE user_id = %s',
                    (user_id,)
                )
                user = cursor.fetchone()
                
                if not user:
                    return None
                
                return {
                    "user_id": user['user_id'],
                    "username": user['username'],
                    "role": user['role'],
                    "permissions": json.loads(user['permissions']),
                    "created_at": user['created_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(user['created_at'], datetime) else str(user['created_at']),
                    "updated_at": user['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(user['updated_at'], datetime) else str(user['updated_at'])
                }
        finally:
            conn.close()
    
    def list_users(self) -> List[dict]:
        """
        获取所有用户列表（不包含密码）
        
        :return: 用户列表
        """
        conn = self._get_connection()
        try:
            with conn.cursor(row_factory=rows.dict_row) as cursor:
                cursor.execute(
                    'SELECT user_id, username, role, permissions, created_at, updated_at FROM users'
                )
                users = cursor.fetchall()
                
                return [
                    {
                        "user_id": user['user_id'],
                        "username": user['username'],
                        "role": user['role'],
                        "permissions": json.loads(user['permissions']),
                        "created_at": user['created_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(user['created_at'], datetime) else str(user['created_at']),
                        "updated_at": user['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(user['updated_at'], datetime) else str(user['updated_at'])
                    }
                    for user in users
                ]
        finally:
            conn.close()


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
