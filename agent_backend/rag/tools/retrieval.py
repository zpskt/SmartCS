"""
检索相关工具
包含知识库搜索、文档检索等功能
"""
import logging
from typing import Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def create_search_knowledge_base_tool(retriever: Any):
    """
    创建知识库检索工具
    
    :param retriever: 向量检索器实例
    :return: LangChain Tool
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """在企业知识库中搜索相关信息。当用户提问时,应该先调用此工具检索相关知识。"""
        logger.info(f"🔍 调用工具: search_knowledge_base | 查询: {query[:50]}...")
        docs = retriever.invoke(query)
        logger.info(f"📄 检索结果: {len(docs)} 个文档")
        
        if not docs:
            logger.warning("⚠️ 未找到相关参考资料")
            return "未找到相关参考资料"
            
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("title", "未知来源")
            formatted.append(f"[资料{i}] 来源:{source}\n内容:{doc.page_content}")
        
        result = "\n\n".join(formatted)
        logger.debug(f"✅ 返回检索结果 | 总长度: {len(result)} 字符")
        return result
    
    return search_knowledge_base
