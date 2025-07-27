#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/27 22:24
# @Author  : zhangpeng /zpskt
# @File    : base_config.py
# @Software: PyCharm
# 基础配置
# LLM相关配置
import os

LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))

# 数据库配置
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///smartcs.db')

# 知识库配置
KNOWLEDGE_BASE_PATH = os.environ.get('KNOWLEDGE_BASE_PATH', 'knowledge_base')

# 日志配置
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# 其他配置
MAX_DIALOG_TURNS = int(os.environ.get('MAX_DIALOG_TURNS', '10'))
SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '1800'))  # 30分钟