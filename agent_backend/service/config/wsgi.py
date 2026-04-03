"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# 设置Django项目的配置模块
# 该环境变量指定了Django应用使用的settings文件路径
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# 创建WSGI应用实例
# get_wsgi_application()函数返回一个可调用的WSGI应用程序对象，
# 该对象作为Web服务器与Django应用之间的接口
application = get_wsgi_application()
