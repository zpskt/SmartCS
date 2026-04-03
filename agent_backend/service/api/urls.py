#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：urls.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/22 00:16 
@Description： 
'''
from django.urls import path
from . import views

urlpatterns = [
    # 健康检查
    path('health/', views.HealthView.as_view(), name='health'),

    # 对话接口
    path('chat/', views.ChatView.as_view(), name='chat'),

    # 会话管理
    path('sessions/', views.SessionView.as_view(), name='session-list'),
    path('sessions/<str:session_id>/', views.SessionView.as_view(), name='session-detail'),
    path('sessions/<str:session_id>/history/', views.HistoryView.as_view(), name='history'),
]
