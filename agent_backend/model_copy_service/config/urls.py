"""
项目的根URL配置。

该模块定义了顶层的URL路由。所有以 `/api/` 开头的请求都将被转发到 `api` 应用的路由配置中进行处理，实现了URL的模块化管理。
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # API 路由
]
