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
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

'''
这里主要学习上下文context：
context相当于AI的临时记忆，包含对话历史、相关文档、用户信息等。
f
'''

# 内存存储：session_id -> 历史记录对象
store = {}
# 全局模型定义
chat_model = ChatTongyi(model="qwen3-max")

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

    # 第一次对话
    response_1 = conversation_chain.invoke({"input": "你好，我叫张鹏，我喜欢编程和喝可乐"}, config)
    print(response_1)
    # 第一次对话
    response_2 = conversation_chain.invoke({"input": "我的爱好是什么？"}, config)
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
    user_name: str

def read_memory_by_tool():
    """
    通过工具访问记忆
    这里state_schema里面传入一个自定义AgentState类，并且定义了里面的key值
    通过get_user_info工具去查询用户信息
    :return:
    """

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
    在这里面通过update_user_info工具，在state里面添加了user_name
    :return:
    """



    agent = create_agent(
        chat_model,
        [update_user_info, greet],
        state_schema=CustomState,
        context_schema=CustomContext,  # [!code highlight]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "问候用户"}]},
        context=CustomContext(user_id="zhangpeng"),
    )
    print(result["messages"][-1].content)

def get_weather(city: str) -> str:
    """获取城市天气"""
    return f" {city}的天气总是晴朗的！"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """
        通过此工具获取短期记忆里面的数据然后作为提示词返回了
        这是一个特殊的 wrapModelCall，专门用于动态生成系统提示词
        它在 beforeModel 阶段执行，但简化了接口
    """
    user_name = request.runtime.context.user_name
    system_prompt = f"你是一位乐于助人的助手。请以{user_name}称呼用户。"
    return system_prompt

def read_memory_by_middleware_prompt():
    """
    通过中间件访问短期记忆，
    :return:
    """
    agent = create_agent(
        model=chat_model,
        tools=[get_weather],
        middleware=[dynamic_system_prompt],
        context_schema=CustomContext,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "临沂的天气如何?"}]},
        context=CustomContext(user_name="张鹏",user_id="zhangpeng"),
    )
    for msg in result["messages"]:
        msg.pretty_print()


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """ 仅保留最后几条消息以适应上下文窗口。"""
    messages = state["messages"]

    print(f"=== trim_messages 执行 ===")
    print(f"当前消息数量：{len(messages)}")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg.__class__.__name__}: {str(msg)[:50]}...")
    print(f"========================")

    if len(messages) <= 3:
        return None  #返回none，即不做任何修改

    first_msg = messages[0]
    # 奇数保留三条，偶数保留四条
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    # 只保留第一条和最近几条消息
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 删除所有旧消息
            *new_messages                           # 插入精简后的新消息
        ]
    }
def read_memory_by_middleware_before_model():
    """
    在模型执行前获取短期记忆

    :return:
    """
    # 创建内存检查点保存器
    memory = MemorySaver()
    agent = create_agent(
        chat_model,
        tools=[get_weather],
        middleware=[trim_messages],
        checkpointer=memory,  # [!code highlight] 添加检查点保存器
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "你好，我的姓名是张鹏"}, config)
    agent.invoke({"messages": "你好，我的性别是男性"}, config)
    agent.invoke({"messages": "写一个关于猫的短诗"}, config)
    agent.invoke({"messages": "现在相同的事情弄个关于狗的"}, config)
    final_response = agent.invoke({"messages": "我的名字是什么?我都让你干了什么事？我的性别是什么？"}, config)

    final_response["messages"][-1].pretty_print()


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """删除聊天信息里面的敏感词"""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

def read_memory_by_middleware_after_model():
    agent = create_agent(
        model=chat_model,
        tools=[],
        middleware=[validate_response],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "你好，我的password是 123456，我的secret是zhangpeng"}, config)

    final_response = agent.invoke({"messages": "设置 password 和 secret 的原则是什么？"}, config)
    for msg in final_response["messages"]:
        msg.pretty_print()
    # 这里其实可以看到看不到最后一条的AI回答，因为已经被我们删除了

if __name__ == '__main__':
    # 通过工具访问短期记忆
    # read_memory_by_tool()
    # 通过工具修改agent的短期记忆
    # edit_memory_by_tool()
    # 通过中间件访问短期记忆
    # read_memory_by_middleware_prompt()
    # 在模型执行前获取记忆执行逻辑
    # read_memory_by_middleware_before_model()

    #在模型执行后获取记忆执行逻辑
    read_memory_by_middleware_after_model()