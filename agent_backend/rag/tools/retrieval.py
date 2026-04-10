"""
检索相关工具
包含知识库搜索、文档检索等功能
"""
from typing import Any
from langchain_core.tools import tool


def create_search_knowledge_base_tool(retriever: Any):
    """
    创建知识库检索工具
    
    :param retriever: 向量检索器实例
    :return: LangChain Tool
    """
    @tool
    def search_knowledge_base(query: str) -> str:
        """在企业知识库中搜索相关信息。当用户提问时,应该先调用此工具检索相关知识。"""
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关参考资料"
            
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("title", "未知来源")
            formatted.append(f"[资料{i}] 来源:{source}\n内容:{doc.page_content}")
            
        return "\n\n".join(formatted)
    
    return search_knowledge_base
