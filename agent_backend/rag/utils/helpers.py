"""
通用工具函数
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict


def calculate_md5(content: str) -> str:
    """
    计算字符串的 MD5 哈希值
    
    :param content: 输入字符串
    :return: MD5 哈希值（32 位十六进制）
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间戳
    
    :param timestamp: 时间对象
    :param format_str: 格式字符串
    :return: 格式化后的时间字符串
    """
    return timestamp.strftime(format_str)


def parse_json_safe(json_str: str) -> Dict[str, Any]:
    """
    安全地解析 JSON 字符串
    
    :param json_str: JSON 字符串
    :return: 解析后的字典，失败返回空字典
    """
    try:
        return json.loads(json_str)
    except Exception:
        return {}


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    截断文本到指定长度
    
    :param text: 原始文本
    :param max_length: 最大长度
    :param suffix: 超出时添加的后缀
    :return: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix
