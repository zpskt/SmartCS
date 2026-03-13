#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：file_history_store.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/13 21:59 
@Description： 
'''
import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
import config_data as config


def get_history(session_id: str):
    return FileHistoryStore(session_id, config.history_file_path)


class FileHistoryStore(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence序列 类似list、tuple
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)  # 新的和已有的融合成一个list

        # 将数据同步写入到本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        # 官方message_to_dict：单个消息对象（BaseMessage类实例） -> 字典
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path+'.json', "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property  # @property装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path+'.json', "r", encoding="utf-8") as f:
                messages_data = json.load(f)  # 返回值就是：list[字典]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path+'.json', "w", encoding="utf-8") as f:
            json.dump([], f)
