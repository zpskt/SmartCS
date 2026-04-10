"""
RAG 对话引擎
核心问答逻辑,结合检索和生成,使用 LangGraph Agent + Checkpointer 实现持久化短期记忆
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from config.settings import settings
from stores.vector_store import VectorStoreService
from tools.retrieval import create_search_knowledge_base_tool
from tools.model_adapter import MODEL_ADAPTER_TOOLS
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
        
        # 根据配置选择聊天模型
        if settings.MODEL_PROVIDER == "ollama":
            self.chat_model = ChatOllama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.TEMPERATURE
            )
        else:  # dashscope
            self.chat_model = ChatTongyi(
                model=settings.CHAT_MODEL,
                temperature=settings.TEMPERATURE,
                dashscope_api_key=settings.DASHSCOPE_API_KEY
            )
        
        # 初始化工具列表
        self.tools = [
            create_search_knowledge_base_tool(self.retriever),
            *MODEL_ADAPTER_TOOLS
        ]
        
        # 定义系统提示词：强调严格基于知识库，禁止捏造
        self.system_prompt = """你是一个企业知识库助手。请严格根据【检索到的参考资料】回答用户的问题。

回答准则：
1. **严格基于事实**：你的所有回答必须完全源自提供的参考资料。
2. **禁止捏造**：如果参考资料中没有包含回答问题所需的信息，请直接回答：“抱歉，我在当前的知识库中没有搜索到相关信息。”，不要尝试编造答案或使用你自己的通用知识。
3. **引用来源**：在回答时，尽量提及信息的来源文档标题。
4. **诚实原则**：如果问题与知识库内容无关，也请诚实告知用户你目前仅能回答知识库范围内的问题。"""
        
        self.middleware = []
        # 初始化 Checkpointer (使用 SQLite)
        self.checkpointer = self._init_checkpointer()
        
        # 创建 Agent
        self.agent = self._create_agent()
    

        
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
        agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            system_prompt=self.system_prompt, # 使用实例变量中定义的提示词
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
        from langchain_core.messages import AIMessage
        
        try:
            # 配置线程 ID
            config = {"configurable": {"thread_id": session_id}}
            
            # 流式调用 Agent
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
                stream_mode="values"
            ):
                last_msg = chunk['messages'][-1]
                # 确保是 AI 消息且有内容
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    logging.info(f"[Stream] {last_msg.content}")
                    yield last_msg.content

        except Exception as e:
            yield f"错误:{str(e)}"
