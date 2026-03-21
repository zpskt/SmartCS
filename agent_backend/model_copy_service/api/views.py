#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：views.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/22 00:15 
@Description： 
'''
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import (
    ChatRequestSerializer,
    ChatResponseSerializer,
    SessionCreateSerializer,
    SessionResponseSerializer,
    HistoryResponseSerializer,
)
from .services import get_model_copy_service

logger = logging.getLogger(__name__)


class ChatView(APIView):
    """对话接口"""

    def post(self, request):
        """
        对话接口

        请求体:
        {
            "message": "用户输入的消息",
            "session_id": "会话ID（可选）",
            "user_id": "用户ID（可选）"
        }

        响应:
        {
            "success": true,
            "session_id": "会话ID",
            "message": "AI响应内容"
        }
        """
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        data = serializer.validated_data
        service = get_model_copy_service()

        result = service.chat(
            message=data['message'],
            session_id=data.get('session_id'),
            user_id=data.get('user_id')
        )

        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionView(APIView):
    """会话管理接口"""

    def post(self, request):
        """
        创建新会话

        请求体:
        {
            "user_id": "用户ID（可选）"
        }

        响应:
        {
            "session_id": "新会话ID",
            "created_at": "创建时间"
        }
        """
        serializer = SessionCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        service = get_model_copy_service()
        session_id = service.create_session(
            user_id=serializer.validated_data.get('user_id')
        )

        return Response({
            "session_id": session_id,
            "created_at": "now"
        }, status=status.HTTP_201_CREATED)

    def get(self, request):
        """
        获取会话列表

        响应:
        {
            "sessions": [
                {
                    "session_id": "...",
                    "message_count": 5,
                    "last_message": "...",
                    "last_updated": "..."
                }
            ]
        }
        """
        service = get_model_copy_service()
        limit = int(request.query_params.get('limit', 20))
        sessions = service.list_sessions(limit)
        return Response({"sessions": sessions}, status=status.HTTP_200_OK)

    def delete(self, request, session_id):
        """
        删除会话

        :param session_id: 会话ID
        """
        service = get_model_copy_service()
        service.delete_session(session_id)
        return Response({"success": True}, status=status.HTTP_200_OK)


class HistoryView(APIView):
    """历史记录接口"""

    def get(self, request, session_id):
        """
        获取会话历史

        :param session_id: 会话ID
        响应:
        {
            "session_id": "...",
            "messages": [...],
            "count": 5
        }
        """
        service = get_model_copy_service()
        history = service.get_history(session_id)
        return Response(history, status=status.HTTP_200_OK)


class HealthView(APIView):
    """健康检查接口"""

    def get(self, request):
        return Response({"status": "ok", "service": "model-copy-agent"}, status=status.HTTP_200_OK)