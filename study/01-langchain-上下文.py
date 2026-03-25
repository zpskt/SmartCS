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
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

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


def short_term_memory():
    '''
    短期记忆示例
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

if __name__ == '__main__':
    short_term_memory()