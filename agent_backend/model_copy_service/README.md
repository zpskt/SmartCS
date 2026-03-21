#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：README.txt.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/22 00:51 
@Description： 
'''

```shell
# 1. 创建项目
cd /path/to/your/project
django-admin startproject config .

# 2. 进入项目目录
cd config

# 3. 创建 API 应用
python manage.py startapp api

# 4. 创建 agents 目录并放入你的智能体代码
mkdir agents
cp /path/to/your/model_copy_agent.py agents/

# 5. 运行迁移
python manage.py migrate


# 6. 启动服务
python manage.py runserver 0.0.0.0:8000

python manage.py createsuperuser
```
默认账号密码均是 zhangepng