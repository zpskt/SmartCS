"""
日志配置工具
提供统一的日志记录器
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logger(name: str = None) -> logging.Logger:
    """
    创建并配置日志记录器
    
    :param name: 日志记录器名称，默认为 root logger
    :return: 配置好的 Logger 实例
    """
    logger = logging.getLogger(name or __name__)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 创建日志目录
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # 日志格式
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # ========== 控制台处理器 ==========
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ========== 文件处理器（所有日志）==========
    all_log_file = os.path.join(settings.LOG_DIR, "app.log")
    file_handler = RotatingFileHandler(
        all_log_file,
        maxBytes=settings.LOG_FILE_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # ========== 错误日志文件处理器 ==========
    error_log_file = os.path.join(settings.LOG_DIR, "error.log")
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=settings.LOG_FILE_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # ========== API 请求日志文件处理器 ==========
    api_log_file = os.path.join(settings.LOG_DIR, "api.log")
    api_handler = RotatingFileHandler(
        api_log_file,
        maxBytes=settings.LOG_FILE_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(formatter)
    logger.addHandler(api_handler)
    
    return logger


# 创建默认的全局日志记录器
logger = setup_logger("rag_system")


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    :param name: 模块名称
    :return: Logger 实例
    """
    if name:
        return setup_logger(name)
    return logger
