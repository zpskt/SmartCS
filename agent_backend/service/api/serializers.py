#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：serializers.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/22 00:14 
@Description： 
'''
from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    """对话请求序列化器"""
    message = serializers.CharField(required=True, help_text="用户输入的消息")
    session_id = serializers.CharField(required=False, help_text="会话ID，不传则自动创建")
    user_id = serializers.CharField(required=False, help_text="用户ID")


class ChatResponseSerializer(serializers.Serializer):
    """对话响应序列化器"""
    success = serializers.BooleanField()
    session_id = serializers.CharField()
    message = serializers.CharField()
    error = serializers.CharField(required=False)


class SessionCreateSerializer(serializers.Serializer):
    """创建会话请求序列化器"""
    user_id = serializers.CharField(required=False, help_text="用户ID")


class SessionResponseSerializer(serializers.Serializer):
    """会话响应序列化器"""
    session_id = serializers.CharField()
    created_at = serializers.DateTimeField()


class HistoryResponseSerializer(serializers.Serializer):
    """历史记录响应序列化器"""
    session_id = serializers.CharField()
    messages = serializers.ListField()
    count = serializers.IntegerField()