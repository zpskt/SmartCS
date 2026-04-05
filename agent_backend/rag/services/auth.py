"""
权限认证服务
实现用户认证、授权和访问控制
"""
import secrets
import sqlite3
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
        
        # 数据库路径
        self.db_path = "./data/users.db"
        
        # 初始化数据库
        self._init_database()
        
        # Token 黑名单
        self.token_blacklist: set = set()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    permissions TEXT DEFAULT '["read"]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            
            # 检查是否存在默认用户，不存在则创建
            cursor.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            if count == 0:
                # 插入默认管理员用户
                cursor.execute(
                    'INSERT INTO users (user_id, username, password, role, permissions) VALUES (?, ?, ?, ?, ?)',
                    ('admin', 'admin', 'admin123', 'admin', '["read", "write", "delete", "manage_users"]')
                )
                # 插入默认普通用户
                cursor.execute(
                    'INSERT INTO users (user_id, username, password, role, permissions) VALUES (?, ?, ?, ?, ?)',
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
            cursor = conn.cursor()
            cursor.execute(
                'SELECT user_id, username, password, role, permissions FROM users WHERE username = ?',
                (username,)
            )
            user = cursor.fetchone()
            
            if not user:
                return None
            
            # 验证明文密码
            if user['password'] != password:
                return None
            
            # 解析权限JSON字符串为列表
            import json
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
            cursor = conn.cursor()
            cursor.execute('SELECT permissions FROM users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return []
            
            import json
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
    
    def add_user(self, user_id: str, username: str, password: str, role: str = "user"):
        """
        添加新用户
        
        :param user_id: 用户 ID
        :param username: 用户名
        :param password: 密码（明文存储）
        :param role: 角色
        """
        import json
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 检查用户是否已存在
            cursor.execute('SELECT user_id FROM users WHERE user_id = ? OR username = ?', (user_id, username))
            if cursor.fetchone():
                raise ValueError(f"用户 {user_id} 或用户名 {username} 已存在")
            
            permissions = json.dumps(self._get_default_permissions(role))
            cursor.execute(
                'INSERT INTO users (user_id, username, password, role, permissions) VALUES (?, ?, ?, ?, ?)',
                (user_id, username, password, role, permissions)
            )
            conn.commit()
        except sqlite3.IntegrityError:
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
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            conn.commit()
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
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET password = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (new_password, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def update_user_role(self, user_id: str, new_role: str):
        """
        更新用户角色
        
        :param user_id: 用户 ID
        :param new_role: 新角色
        """
        import json
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 先检查用户是否存在
            cursor.execute('SELECT role, permissions FROM users WHERE user_id = ?', (user_id,))
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
                'UPDATE users SET role = ?, permissions = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (new_role, permissions_json, user_id)
            )
            conn.commit()
            
            # 验证更新是否成功
            cursor.execute('SELECT role, permissions FROM users WHERE user_id = ?', (user_id,))
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
        import json
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT user_id, username, role, permissions, created_at, updated_at FROM users WHERE user_id = ?',
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
                "created_at": user['created_at'],
                "updated_at": user['updated_at']
            }
        finally:
            conn.close()
    
    def list_users(self) -> List[dict]:
        """
        获取所有用户列表（不包含密码）
        
        :return: 用户列表
        """
        import json
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
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
                    "created_at": user['created_at'],
                    "updated_at": user['updated_at']
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
