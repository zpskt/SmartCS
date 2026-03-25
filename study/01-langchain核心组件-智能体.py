#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：01-langchain核心组件-智能体.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/21 13:59 
@Description： 
'''
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent, AgentState
from langchain_core.runnables import RunnableConfig

chat_model = ChatTongyi(model="qwen3-max")


def static_model_agent():
    '''
    静态模型使用
    :return:
    '''
    agent = create_agent(model=chat_model)
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    response = agent.invoke({"messages": "你好，我是张鹏"}, config)
    for message in response["messages"]:
        print(message.pretty_print())



def dynamic_model_agent():
    '''
    静态模型使用
    :return:
    '''
    advanced_model = ChatTongyi(model="qwen3-max")
    basic_model = ChatTongyi(model="qwen3-max")
    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """根据对话复杂性选择模型。"""
        message_count = len(request.state["messages"])

        if message_count > 10:
            # 对较长的对话使用高级模型
            model = advanced_model
        else:
            model = basic_model

        request.model = model
        return handler(request)

    agent = create_agent(
        model=basic_model,  # 默认模型
        middleware=[dynamic_model_selection]
    )
    response = agent.invoke({"messages": "你好，我是张鹏"})
    for msg in response["messages"]:
        print(msg.pretty_print())

# 使用示例
if __name__ == "__main__":
    # static_model_agent()
    dynamic_model_agent()


