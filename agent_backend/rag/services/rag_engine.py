"""
RAG 对话引擎
核心问答逻辑,结合检索和生成,使用 LangGraph Agent + Checkpointer 实现持久化短期记忆
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from config.settings import settings
from stores.vector_store import VectorStoreService
import sqlite3
import os


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
        
        # 初始化工具列表
        self.tools = [self._create_retrieval_tool()]
        # todo 暂时为null 后期添加
        self.system_prompt = ""
        self.middleware = []
        # 初始化 Checkpointer (使用 SQLite)
        self.checkpointer = self._init_checkpointer()
        
        # 创建 Agent
        self.agent = self._create_agent()
    
    def _create_retrieval_tool(self):
        """
        创建检索工具
            
        :return: LangChain Tool
        """
        @tool
        def search_knowledge_base(query: str) -> str:
            """在企业知识库中搜索相关信息。当用户提问时，应该先调用此工具检索相关知识。"""
            docs = self.retriever.invoke(query)
            if not docs:
                return "未找到相关参考资料"
                
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("title", "未知来源")
                formatted.append(f"[资料{i}] 来源:{source}\n内容:{doc.page_content}")
                
            return "\n\n".join(formatted)
            
        return search_knowledge_base
        
    def _init_checkpointer(self):
        """
        初始化 SQLite Checkpointer
            
        :return: SqliteSaver 实例
        """
        # 确保数据目录存在
        db_dir = os.path.dirname(settings.CHECKPOINTER_DB_PATH)
        os.makedirs(db_dir, exist_ok=True)
            
        # 创建并返回 checkpointer
        conn = sqlite3.connect(settings.CHECKPOINTER_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
            
        return checkpointer
        
    def _create_agent(self):
        """
        创建 ReAct Agent
            
        :return: Agent 实例
        """
        system_prompt = ""
        agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware,
        )
            
        return agent
    
    def query(
        self,
        question: str,
        session_id: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        执行查询 (使用 Agent + Checkpointer)
            
        :param question: 用户问题
        :param session_id: 会话 ID (用于持久化短期记忆)
        :param include_sources: 是否包含来源信息
        :return: 包含回答和来源的字典
        """
        try:
            # 配置线程 ID (用于 checkpointer 识别会话)
            config = {"configurable": {"thread_id": session_id}}
                
            # 调用 Agent
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config
            )
                
            # 提取最后一条消息作为回答
            messages = result.get("messages", [])
            if not messages:
                return {
                    "answer": "抱歉，未收到回复",
                    "sources": []
                }
                
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
            response = {
                "answer": answer,
                "sources": []
            }
                
            if include_sources:
                # 获取来源信息 (重新检索以获取元数据)
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
                "answer": f"抱歉，处理时出现错误:{str(e)}",
                "sources": []
            }
    
    def query_with_stream(
        self,
        question: str,
        session_id: str
    ):
        """
        流式查询 (生成器)
        
        :param question: 用户问题
        :param session_id: 会话 ID
        :yield: 流式输出的文本片段
        """
        try:
            # 配置线程 ID
            config = {"configurable": {"thread_id": session_id}}
            
            # 流式调用 Agent
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in chunk:
                    last_msg = chunk["messages"][-1]
                    if hasattr(last_msg, 'content') and last_msg.content:
                        yield last_msg.content
        except Exception as e:
            yield f"错误:{str(e)}"
