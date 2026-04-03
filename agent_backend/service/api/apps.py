from django.apps import AppConfig


class ApiConfig(AppConfig):
    """
    API应用配置类

    该类定义了api应用的配置信息，继承自Django的AppConfig。
    name: 指定应用的Python导入路径
    default_auto_field: 指定默认的自动主键字段类型
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
