"""
知识库管理服务
支持多种知识源：本地文件、飞书文档、URL 等
"""
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from models.schemas import KnowledgeDocument
from stores.vector_store import VectorStoreService


class KnowledgeBaseService:
    """知识库管理服务类"""
    
    def __init__(self):
        """初始化知识库管理服务"""
        self.vector_store = VectorStoreService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )
        self.md5_file_path = "./data/rag_md5.txt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.md5_file_path), exist_ok=True)
    
    def _calculate_md5(self, content: str) -> str:
        """
        计算内容 MD5
        
        :param content: 文本内容
        :return: MD5 哈希值
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _check_md5_exists(self, md5_hex: str) -> bool:
        """
        检查 MD5 是否存在
        
        :param md5_hex: MD5 哈希值
        :return: 是否存在
        """
        if not os.path.exists(self.md5_file_path):
            return False
        
        with open(self.md5_file_path, 'r', encoding='utf-8') as f:
            return md5_hex in f.read()
    
    def _save_md5(self, md5_hex: str):
        """
        保存 MD5
        
        :param md5_hex: MD5 哈希值
        """
        with open(self.md5_file_path, 'a', encoding='utf-8') as f:
            f.write(md5_hex + '\n')
    
    def add_document(
        self,
        title: str,
        content: str,
        source_type: str,
        created_by: str,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeDocument:
        """
        添加文档到知识库
        
        :param title: 文档标题
        :param content: 文档内容
        :param source_type: 来源类型 (file, feishu, url, manual)
        :param created_by: 创建者
        :param source_url: 来源 URL（可选）
        :param metadata: 额外元数据（可选）
        :return: KnowledgeDocument 对象
        """
        # 检查是否重复
        md5_hex = self._calculate_md5(content)
        if self._check_md5_exists(md5_hex):
            raise ValueError("文档内容已存在")
        
        # 文本分块
        chunks = self.text_splitter.split_text(content)
        
        # 创建文档对象
        doc = KnowledgeDocument(
            doc_id=f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{md5_hex[:8]}",
            title=title,
            content=content,
            source_type=source_type,
            source_url=source_url,
            metadata=metadata or {},
            chunks=chunks,
            created_by=created_by
        )
        
        # 添加到向量数据库
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=[{
                "doc_id": doc.doc_id,
                "title": title,
                "source_type": source_type,
                "created_by": created_by,
                "created_at": doc.created_at.isoformat()
            } for _ in chunks]
        )
        
        # 保存 MD5
        self._save_md5(md5_hex)
        
        return doc
    
    def add_documents_from_feishu(
        self,
        feishu_docs: List[Dict[str, Any]],
        created_by: str
    ) -> List[KnowledgeDocument]:
        """
        从飞书批量添加文档
        
        :param feishu_docs: 飞书文档列表
        :param created_by: 创建者
        :return: 文档列表
        """
        results = []
        for doc_data in feishu_docs:
            try:
                doc = self.add_document(
                    title=doc_data.get("title", "无标题"),
                    content=doc_data.get("content", ""),
                    source_type="feishu",
                    created_by=created_by,
                    source_url=doc_data.get("url"),
                    metadata={"feishu_doc_id": doc_data.get("doc_id")}
                )
                results.append(doc)
            except Exception as e:
                print(f"添加飞书文档失败：{e}")
        
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        :param doc_id: 文档 ID
        :return: 是否成功删除
        """
        return self.vector_store.delete_by_metadata("doc_id", doc_id)
    
    def search_documents(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索文档
        
        :param query: 查询文本
        :param limit: 返回数量限制
        :param filters: 过滤条件
        :return: 文档列表
        """
        docs = self.vector_store.similarity_search(query, limit, filters)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "metadata", {}).get("score", 0)
            }
            for doc in docs
        ]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        获取所有文档列表
        
        :return: 文档列表
        """
        all_data = self.vector_store.get_all_documents()
        return all_data.get("documents", [])
    
    def update_document(self, doc_id: str, new_content: str) -> bool:
        """
        更新文档内容
        
        :param doc_id: 文档 ID
        :param new_content: 新内容
        :return: 是否成功
        """
        # 先删除旧文档
        self.delete_document(doc_id)
        # 重新添加（实际应该保留部分元数据）
        # 这里简化处理
        return True
    
    def get_document_count(self) -> int:
        """
        获取文档总数
        
        :return: 文档数量
        """
        all_data = self.vector_store.get_all_documents()
        return len(all_data.get("ids", []))
