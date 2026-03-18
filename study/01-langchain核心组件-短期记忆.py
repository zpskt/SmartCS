#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：01-langchain-上下文.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/18 23:26 
@Description： 
'''
from langchain.agents import create_agent, AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

'''
这里主要学习上下文context：
context相当于AI的临时记忆，包含对话历史、相关文档、用户信息等。
f
'''

# 内存存储：session_id -> 历史记录对象
store = {}
def get_session_history(session_id: str):
    """
    获取指定 session_id 的对话历史

    :param session_id: 会话 ID
    :return: 该会话的历史记录
    """
    # 实际应用中可以从数据库或缓存中读取
    # 这里简化为返回空列表
    print(f"获取会话 {session_id} 的历史记录")
    if session_id not in store:
        # 创建新的历史记录对象
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


def short_term_memory_old_version():
    '''
    短期记忆示例--老版本
    :return:
    '''
    # 新建模型
    chat_model = ChatTongyi(model="qwen3-max")
    #
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，记得用户之前说过的话。"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
        ("human", "{input}"),  # 当前用户输入
    ])

    # 先构建完整的链，然后包装历史管理
    chain = prompt | chat_model | StrOutputParser()
    # 创建一个有历史的可执行的chain
    conversation_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    # 配置会话 ID
    config = {
        "configurable": {
            "session_id": "user_zhangpeng_001"  # ✅ 添加会话 ID
        }
    }

    #第一次对话
    response_1 = conversation_chain.invoke({"input": "你好，我叫张鹏，我喜欢编程和喝可乐"},config)
    print(response_1)
    #第一次对话
    response_2 = conversation_chain.invoke({"input": "我的爱好是什么？"},config)
    print(response_2)

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """查看用户信息"""
    user_id = runtime.state["user_id"]
    return "用户是张鹏" if user_id == "zhangpeng" else "不认识的 user"

class CustomState(AgentState):
    """
    这里要提前定义好要存储的key，最终值会出现在runtime.state里面
    """
    user_id: str
    user_name: str

class CustomContext(BaseModel):
    """
    这里要提前定义好要存储的key，最终值会出现在runtime.context里面
    """
    user_id: str

def access_memory_by_tool():
    """
    通过工具访问记忆
    :return:
    """
    chat_model = ChatTongyi(model="qwen3-max")

    agent = create_agent(
        chat_model,
        [get_user_info],
        state_schema=CustomState,  # [!code highlight]
    )
    result = agent.invoke({
        "messages": "查看用户信息",
        "user_id": "zhangpeng"
    })
    print(result["messages"][-1].content)

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """查看并更新用户信息"""
    user_id = runtime.context.user_id  # [!code highlight]
    name = "张鹏" if user_id == "zhangpeng" else "我没有见过这个人！！！！"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "成功查看用户信息",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """一旦获取到用户信息，就使用下面的话问候用户"""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"

def edit_memory_by_tool():
    """
    通过工具修改agent的短期记忆
    :return:
    """
    '''首先生成一个记忆：userid '''


    chat_model = ChatTongyi(model="qwen3-max")

    agent = create_agent(
        chat_model,
        [update_user_info,greet],
        state_schema=CustomState,
        context_schema=CustomContext,  # [!code highlight]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "问候用户"}]},
        context=CustomContext(user_id="zhangpeng"),
    )
    print(result["messages"][-1].content)

if __name__ == '__main__':
    # access_memory_by_tool()
    edit_memory_by_tool()