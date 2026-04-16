"""
向量存储服务
封装 Chroma 向量数据库操作
"""
import os
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from config.settings import settings
import chromadb


class VectorStoreService:
    """向量存储服务类"""
    
    def __init__(self):
        """初始化向量存储服务"""
        # 根据配置选择嵌入模型
        if settings.MODEL_PROVIDER == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=settings.OLLAMA_EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
        else:  # dashscope
            self.embeddings = DashScopeEmbeddings(
                model=settings.EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY
            )
        
        # 根据配置选择 Chroma 连接方式
        if settings.CHROMA_USE_HTTP:
            # HTTP 客户端模式（远程服务器）
            chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embeddings
            )
        else:
            # 本地文件模式
            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name=settings.COLLECTION_NAME,
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        添加文本到向量库
        
        :param texts: 文本列表
        :param metadatas: 元数据列表（可选）
        :return: 文档 ID 列表
        """
        ids = self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        return ids
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量库
        
        :param documents: LangChain Document 列表
        :return: 文档 ID 列表
        """
        ids = self.vectorstore.add_documents(documents)
        return ids
    
    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度搜索
        
        :param query: 查询文本
        :param limit: 返回数量限制
        :param filters: 过滤条件
        :return: 文档列表
        """
        if filters:
            return self.vectorstore.similarity_search(
                query=query,
                k=limit,
                filter=filters
            )
        return self.vectorstore.similarity_search(query=query, k=limit)
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        获取检索器
        
        :param search_kwargs: 检索参数
        :return: 检索器对象
        """
        if search_kwargs:
            return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        return self.vectorstore.as_retriever()
    
    def delete_by_metadata(self, key: str, value: Any) -> bool:
        """
        通过元数据删除文档
        
        :param key: 元数据键名
        :param value: 元数据值
        :return: 是否成功删除
        """
        # Chroma 不直接支持按元数据删除，需要先搜索再删除
        docs = self.vectorstore.get()
        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(docs.get("ids", []), docs.get("metadatas", []))
            if metadata and metadata.get(key) == value
        ]
        
        if ids_to_delete:
            self.vectorstore.delete(ids_to_delete)
            return True
        return False
    
    def get_all_documents(self) -> Dict[str, Any]:
        """
        获取所有文档（用于管理界面）
        
        :return: 包含所有文档信息的字典
        """
        return self.vectorstore.get()
    
    def delete(self, ids: List[str]) -> bool:
        """
        批量删除文档
        
        :param ids: 文档 ID 列表
        :return: 是否成功
        """
        try:
            self.vectorstore.delete(ids)
            return True
        except Exception:
            return False
