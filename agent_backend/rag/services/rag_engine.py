"""
RAG 对话引擎
核心问答逻辑，结合检索和生成
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
from config.settings import settings
from stores.vector_store import VectorStoreService


class RAGEngine:
    """RAG 对话引擎类"""
    
    def __init__(self):
        """初始化 RAG 对话引擎"""
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever(
            search_kwargs={"k": settings.SIMILARITY_TOP_K}
        )
        self.chat_model = ChatTongyi(
            model=settings.CHAT_MODEL,
            temperature=settings.TEMPERATURE,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )
        
        # 构建 prompt 模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的企业知识库助手。请根据提供的参考资料回答用户问题。
            
回答规则：
1. 优先基于参考资料回答，不要编造信息
2. 如果参考资料不足，请如实说明
3. 回答简洁明了，专业准确
4. 必要时可以引用参考资料的来源

参考资料：
{context}

历史对话：
{history}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # 构建链
        self.chain = self._build_chain()
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        格式化检索到的文档
        
        :param docs: 文档列表
        :return: 格式化后的文本
        """
        if not docs:
            return "未找到相关参考资料"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("title", "未知来源")
            formatted.append(f"[资料{i}] 来源：{source}\n内容：{doc.page_content}")
        
        return "\n\n".join(formatted)
    
    def _build_chain(self):
        """
        构建 RAG 链
        
        :return: 执行链
        """
        
        def retrieve_and_format(input_dict: dict) -> dict:
            """检索并格式化上下文"""
            question = input_dict["question"]
            docs = self.retriever.invoke(question)
            context = self._format_docs(docs)
            return {
                "context": context,
                "question": question,
                "chat_history": input_dict.get("chat_history", []),
                "sources": [doc.metadata for doc in docs]
            }
        
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["question"]))
            )
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        执行查询
        
        :param question: 用户问题
        :param chat_history: 聊天历史（可选）
        :param include_sources: 是否包含来源信息
        :return: 包含回答和来源的字典
        """
        try:
            # 执行链
            result = self.chain.invoke({
                "question": question,
                "chat_history": chat_history or []
            })
            
            response = {
                "answer": result,
                "sources": []
            }
            
            if include_sources:
                # 获取来源信息
                docs = self.retriever.invoke(question)
                response["sources"] = [
                    {
                        "title": doc.metadata.get("title", "未知来源"),
                        "source_type": doc.metadata.get("source_type", "unknown"),
                        "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    for doc in docs[:3]  # 只显示前 3 个来源
                ]
            
            return response
            
        except Exception as e:
            return {
                "answer": f"抱歉，处理时出现错误：{str(e)}",
                "sources": []
            }
    
    def query_with_stream(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        流式查询（生成器）
        
        :param question: 用户问题
        :param chat_history: 聊天历史
        :yield: 流式输出的文本片段
        """
        # TODO: 实现流式输出
        pass
