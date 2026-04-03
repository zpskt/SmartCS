"""
飞书集成模块
支持从飞书云文档同步知识
"""
import requests
from typing import List, Dict, Any, Optional
from config.settings import settings


class FeishuClient:
    """飞书 API 客户端类"""
    
    def __init__(self):
        """初始化飞书客户端"""
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET
        self.base_url = settings.FEISHU_BASE_URL
        self.tenant_access_token: Optional[str] = None
    
    def _get_tenant_access_token(self) -> str:
        """
        获取租户访问令牌
        
        :return: 租户访问令牌
        """
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        response = requests.post(url, json=payload)
        result = response.json()
        if result.get("code") == 0:
            self.tenant_access_token = result.get("tenant_access_token")
            return self.tenant_access_token
        else:
            raise Exception(f"获取令牌失败：{result}")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        获取请求头
        
        :return: 请求头字典
        """
        if not self.tenant_access_token:
            self._get_tenant_access_token()
        return {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json"
        }
    
    def get_drive_files(
        self,
        folder_token: str,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取云盘文件列表
        
        :param folder_token: 文件夹 token
        :param page_size: 每页数量
        :return: 文件列表
        """
        url = f"{self.base_url}/drive/v1/files/search"
        params = {
            "folder_token": folder_token,
            "page_size": page_size
        }
        response = requests.get(url, params=params, headers=self._get_headers())
        result = response.json()
        if result.get("code") == 0:
            return result.get("data", {}).get("items", [])
        return []
    
    def get_docx_content(self, docx_token: str) -> Dict[str, Any]:
        """
        获取云文档内容
        
        :param docx_token: 文档 token
        :return: 文档内容和元数据
        """
        # 获取文档元数据
        meta_url = f"{self.base_url}/docs/v1/documents/{docx_token}"
        meta_response = requests.get(meta_url, headers=self._get_headers())
        meta_result = meta_response.json()
        
        if meta_result.get("code") != 0:
            raise Exception(f"获取文档元数据失败：{meta_result}")
        
        document = meta_result.get("data", {})
        
        # 获取文档正文内容
        content_url = f"{self.base_url}/docs/v1/documents/{docx_token}/raw_content"
        content_response = requests.get(content_url, headers=self._get_headers())
        content_result = content_response.json()
        
        if content_result.get("code") != 0:
            raise Exception(f"获取文档内容失败：{content_result}")
        
        return {
            "doc_id": docx_token,
            "title": document.get("title", "无标题"),
            "content": content_result.get("data", {}).get("content", ""),
            "url": f"https://your-company.feishu.cn/docs/{docx_token}",
            "creator": document.get("edit_user_id"),
            "create_time": document.get("create_time")
        }
    
    def sync_folder_documents(self, folder_token: str, created_by: str) -> List[Dict[str, Any]]:
        """
        同步整个文件夹的文档
        
        :param folder_token: 文件夹 token
        :param created_by: 同步操作者
        :return: 文档列表
        """
        files = self.get_drive_files(folder_token)
        documents = []
        
        for file in files:
            if file.get("file_type") == "docx":  # 只处理云文档
                try:
                    doc_content = self.get_docx_content(file.get("token"))
                    documents.append(doc_content)
                except Exception as e:
                    print(f"同步文档 {file.get('title')} 失败：{e}")
        
        return documents
