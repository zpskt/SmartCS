#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/13 22:30 
@Description： 
'''
from core.rag import RagService

if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "我身高175cm，体重160斤，给我推荐尺码？"}, session_config)
    print(res)