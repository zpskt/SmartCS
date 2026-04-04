"""
飞书集成模块
支持从飞书云文档同步知识
"""
import requests
import time
from typing import List, Dict, Any, Optional
from config.settings import settings
from utils.logger import get_logger

# 获取应用层日志记录器
logger = get_logger("app")

class FeishuClient:
    """飞书 API 客户端类"""
    
    def __init__(self):
        """初始化飞书客户端"""
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET
        self.base_url = settings.FEISHU_BASE_URL
        self.tenant_access_token: Optional[str] = None
        self.token_expire_time: float = 0  # token过期时间戳
    
    def _get_tenant_access_token(self) -> str:
        """
        获取租户访问令牌（带缓存和自动刷新）
        
        :return: 租户访问令牌
        """
        # 检查token是否有效（提前5分钟刷新）
        if self.tenant_access_token and time.time() < self.token_expire_time - 300:
            return self.tenant_access_token
        
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            result = response.json()
            
            if result.get("code") == 0:
                self.tenant_access_token = result.get("tenant_access_token")
                expire_seconds = result.get("expire", 7200)
                self.token_expire_time = time.time() + expire_seconds
                logger.info(f"✅ 飞书Token获取成功 | 有效期: {expire_seconds}秒")
                return self.tenant_access_token
            else:
                error_msg = result.get("msg", "未知错误")
                logger.error(f"❌ 飞书Token获取失败 | Code: {result.get('code')} | Msg: {error_msg}")
                raise Exception(f"获取令牌失败：{error_msg}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 飞书Token请求异常: {str(e)}")
            raise Exception(f"获取令牌网络异常：{str(e)}")
    
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
        获取云文档内容（需要认证）
        
        :param docx_token: 文档 token/ID
        :return: 文档内容和元数据
        """
        # 1. 获取title
        title_url = f"{self.base_url}/docx/v1/documents/{docx_token}"
        title_response = requests.get(title_url, headers=self._get_headers(), timeout=10)
        document = title_response.json().get("data", {}).get("document", {})
        # 2. 获取文档正文内容（markdown格式）
        content_url = f"{self.base_url}/docs/v1/content?content_type=markdown&doc_token={docx_token}&doc_type=docx&lang=zh"
        content_response = requests.get(content_url, headers=self._get_headers(), timeout=10)
        content_result = content_response.json()
        
        if content_result.get("code") != 0:
            error_msg = content_result.get("msg", "未知错误")
            logger.error(f"❌ 获取文档内容失败 | ID: {docx_token} | Code: {content_result.get('code')} | Msg: {error_msg}")
            raise Exception(f"获取文档内容失败：{error_msg}")
        
        content = content_result.get("data", {}).get("content", "")
        
        logger.info(f"✅ 成功获取飞书文档 | ID: {docx_token} | 标题: {document.get('title')} | 内容长度: {len(content)}")
        
        return {
            "doc_id": docx_token,
            "title": document.get('title'),
            "content": content,
            "url": f"https://xxx.feishu.cn/docs/{docx_token}",
            "creator": "飞书作者",
            "create_time": None
        }
    
    def get_public_doc_content(self, url: str) -> Dict[str, Any]:
        """
        获取公网可访问的飞书文档内容（无需认证）
        
        :param url: 飞书文档 URL
        :return: 文档内容和元数据
        """
        try:
            # 直接请求网页内容
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # 尝试从 HTML 中提取标题和内容
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "无标题"
            
            # 提取主要内容（尝试多个选择器）
            content_selectors = [
                'article',
                '.doc-content',
                '.wiki-content',
                '[role="main"]',
                'main'
            ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator='\n', strip=True)
                    break
            
            # 如果没找到特定元素，使用 body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            if not content:
                raise Exception("无法从网页中提取内容")
            
            logger.info(f"✅ 成功获取公网文档 | URL: {url} | 标题: {title}")
            
            return {
                "doc_id": url,
                "title": title,
                "content": content,
                "url": url,
                "creator": None,
                "create_time": None
            }
        except ImportError:
            # 如果没有安装 beautifulsoup4，返回简单文本
            logger.warning("⚠️ 未安装 beautifulsoup4，使用简单文本提取")
            return {
                "doc_id": url,
                "title": "飞书文档",
                "content": response.text[:5000],  # 限制长度
                "url": url,
                "creator": None,
                "create_time": None
            }
        except Exception as e:
            logger.error(f"❌ 获取公网文档失败 | URL: {url} | 错误: {str(e)}")
            raise Exception(f"获取公网文档失败：{str(e)}")
    
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
