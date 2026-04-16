"""
型号适配专用 RAG 引擎
使用专门的系统提示词和工具集
"""
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.middleware import wrap_tool_call
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from config.settings import settings
from tools.model_adapter import MODEL_ADAPTER_TOOLS

# 获取日志记录器
logger = logging.getLogger(__name__)


# ==================== Middleware 装饰器 ====================

@wrap_tool_call
def log_tool_call(request, handler):
    """工具调用监控中间件"""
    tool_name = request.tool_call['name']
    tool_args = request.tool_call['args']
    
    logger.info(f"🔧 调用工具: {tool_name}")
    logger.debug(f"   参数: {tool_args}")
    
    # 执行工具
    result = handler(request)
    
    content_preview = str(result)[:200] if len(str(result)) > 200 else str(result)
    logger.info(f"✅ 工具完成: {tool_name}")
    logger.debug(f"   返回内容预览: {content_preview}")
    
    return result

from langchain.agents import create_agent, AgentState


class ModelAdapterEngine:
    """型号适配专用引擎"""
    
    def __init__(self):
        """初始化型号适配引擎"""
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
        
        # 型号适配专用工具列表
        self.tools = MODEL_ADAPTER_TOOLS
        
        # 型号适配专用系统提示词
        self.system_prompt = """你是一个专业的产品型号适配助手,专门帮助用户查询和管理产品型号信息。

你的核心能力:
1. **型号查询**:可以查询所有可用的产品型号、搜索特定型号
2. **功能适配**:可以查询某个型号适配了哪些功能
3. **详细信息**:可以提供型号的完整信息,包括成品编码、创建人、创建时间等
4. **SQL生成**:当用户明确要求生成开关门相关的 SQL 语句时,使用 generate_insert_sql 工具

回答准则:
1. **专业准确**:严格基于工具返回的信息回答,不要编造数据
2. **结构化输出**:尽量以清晰的格式展示信息(如列表、表格)
3. **主动引导**:如果用户的问题不明确,主动询问具体需求
4. **友好交互**:使用专业的语气,但保持友好易懂
5. **工具调用时机**:
   - 查询型号信息时:使用 list_all_models、search_models、get_model_features、get_model_details
   - **仅当用户明确要求生成开关门 SQL 时**:才使用 generate_insert_sql 工具
   - 其他场景不要调用 generate_insert_sql

可用工具:
- list_all_models: 列出所有型号
- search_models: 搜索型号(支持模糊匹配)
- get_model_features: 查询型号的功能适配
- get_model_details: 查询型号的详细信息
- generate_insert_sql: 生成开关门配置 SQL(仅在用户明确要求时使用)

当用户询问时,优先使用最合适的工具获取信息,然后组织成易读的格式返回。"""
        
        # 初始化 Checkpointer (使用 PostgreSQL)
        self.checkpointer = self._init_checkpointer()
        
        # 创建 Agent（传入 middleware）
        self.agent = self._create_agent()
    
    def _init_checkpointer(self):
        """初始化 PostgreSQL Checkpointer"""
        # 创建并返回 checkpointer
        import psycopg
        conn = psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DATABASE,
            autocommit=True
        )
        checkpointer = PostgresSaver(conn)
        
        # 创建所需的表结构
        checkpointer.setup()
        
        return checkpointer
    
    def _create_agent(self):
        """创建 ReAct Agent"""
        agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            system_prompt=self.system_prompt,  # 使用实例变量中定义的提示词
            checkpointer=self.checkpointer,
            middleware=[log_tool_call],  # 使用装饰器定义的 middleware
        )
        
        return agent
    
    def query(
        self,
        question: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        执行查询
        
        :param question: 用户问题
        :param session_id: 会话 ID
        :return: 包含回答的字典
        """
        try:
            logger.info(f"🔧 开始型号适配查询 | 会话: {session_id} | 问题长度: {len(question)}")
            
            # 配置线程 ID
            config = {"configurable": {"thread_id": session_id}}
            
            # 调用 Agent
            logger.debug("⚙️ 调用型号适配 Agent...")
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config
            )
            logger.debug("✅ Agent 调用完成")
            
            # 提取最后一条消息作为回答
            messages = result.get("messages", [])
            if not messages:
                logger.warning("⚠️ Agent 未返回任何消息")
                return {
                    "answer": "抱歉，未收到回复",
                    "sources": []
                }
            
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
            logger.info(f"💬 生成回答 | 长度: {len(answer)} 字符")
            
            logger.info(f"✅ 型号适配查询完成 | 会话: {session_id}")
            return {
                "answer": answer,
                "sources": []  # 型号适配不使用传统来源
            }
            
        except Exception as e:
            logger.error(f"❌ 型号适配查询失败 | 错误: {str(e)}", exc_info=True)
            return {
                "answer": f"抱歉，处理时出现错误: {str(e)}",
                "sources": []
            }
    
    def query_with_stream(
        self,
        question: str,
        session_id: str
    ):
        """
        流式查询
        
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
                    yield last_msg.content
                    
        except Exception as e:
            logging.error(f"型号适配流式查询错误: {e}", exc_info=True)
            yield f"错误: {str(e)}"
