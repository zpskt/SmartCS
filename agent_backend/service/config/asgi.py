"""
ASGI (Asynchronous Server Gateway Interface) 配置。

此模块为项目设置 ASGI 应用程序。它将 Django 设置暴露给 ASGI 兼容的 Web 服务器，以处理异步请求。
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_asgi_application()
