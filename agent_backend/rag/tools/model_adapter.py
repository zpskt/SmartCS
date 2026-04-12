"""
型号适配相关工具
提供产品型号查询、功能适配、成品编码等信息检索功能
"""
import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ==================== 数据管理层 ====================

DATA_FILE = "./data/models.json"
_models_cache: Dict[str, Any] = {}



def _save_models():
    """保存型号数据"""
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(_models_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存型号数据失败: {e}")


def _get_all_models(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取所有型号"""
    # todo 调用外部接口
    models = []
    if status:
        models = [m for m in models if m.get('status') == status]
    return models


def _get_model_by_code(model_code: str) -> Optional[Dict[str, Any]]:
    """根据型号编码获取型号"""
    # todo 调用外部接口
    return []


def _search_models(keyword: str) -> List[Dict[str, Any]]:
    """搜索型号"""
    keyword_lower = keyword.lower()
    models = []# todo 调用外部接口
    results = []
    for model in models:
        if (keyword_lower in model.get('model_code', '').lower() or
            keyword_lower in model.get('model_name', '').lower() or
            keyword_lower in model.get('product_code', '').lower()):
            results.append(model)
    return results


# ==================== 工具定义 ====================

@tool
def list_all_models(status: str = "active") -> str:
    """列出所有可用的产品型号。当用户询问有哪些型号时使用此工具。
    
    Args:
        status: 型号状态，可选值：active（活跃）、inactive（停用），默认为 active
    """
    logger.info(f"🔧 调用工具: list_all_models | 状态: {status}")
    try:
        models = _get_all_models(status=status)
        logger.info(f"📄 查询结果: {len(models)} 个型号")
        
        if not models:
            return f"未找到状态为 {status} 的型号"
        
        result = []
        for model in models:
            result.append(
                f"- 型号编码: {model['model_code']}\n"
                f"  型号名称: {model['model_name']}\n"
                f"  成品编码: {model['product_code']}\n"
                f"  状态: {model['status']}"
            )
        
        return f"共找到 {len(models)} 个型号:\n\n" + "\n\n".join(result)
    except Exception as e:
        logger.error(f"❌ 查询型号列表失败: {str(e)}")
        return f"查询型号列表失败: {str(e)}"


@tool
def get_model_features(model_code: str) -> str:
    """查询某个型号适配了哪些功能。当用户询问某个型号的功能时使用此工具。
    
    Args:
        model_code: 型号编码，例如 "M001"
    """
    logger.info(f"🔧 调用工具: get_model_features | 型号: {model_code}")
    try:
        model = _get_model_by_code(model_code)
        if not model:
            logger.warning(f"⚠️ 型号不存在: {model_code}")
            return f"型号 {model_code} 不存在"
        
        features = model.get('features', [])
        logger.info(f"📄 查询结果: {len(features)} 个功能")
        
        if not features:
            return f"型号 {model_code} 未配置功能"
        
        result = f"型号 {model_code} ({model['model_name']}) 适配的功能:\n"
        for i, feature in enumerate(features, 1):
            result += f"{i}. {feature}\n"
        
        return result
    except Exception as e:
        logger.error(f"❌ 查询型号功能失败: {str(e)}")
        return f"查询型号功能失败: {str(e)}"


@tool
def get_model_details(model_code: str) -> str:
    """查询某个型号的详细信息，包括成品编码、创建人、创建时间等。
    
    Args:
        model_code: 型号编码，例如 "M001"
    """
    logger.info(f"🔧 调用工具: get_model_details | 型号: {model_code}")
    try:
        model = _get_model_by_code(model_code)
        if not model:
            logger.warning(f"⚠️ 未找到型号: {model_code}")
            return f"未找到型号 {model_code} 的信息"
        
        result = (
            f"型号详细信息:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"型号编码: {model['model_code']}\n"
            f"型号名称: {model['model_name']}\n"
            f"成品编码: {model['product_code']}\n"
            f"创建人: {model['creator']}\n"
            f"创建时间: {model.get('created_at', '未知')}\n"
            f"更新时间: {model.get('updated_at', '未知')}\n"
            f"状态: {model['status']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )
        
        features = model.get('features', [])
        if features:
            result += f"\n适配功能 ({len(features)} 个):\n"
            for feature in features:
                result += f"  • {feature}\n"
        
        logger.info(f"✅ 返回型号详情")
        return result
    except Exception as e:
        logger.error(f"❌ 查询型号详情失败: {str(e)}")
        return f"查询型号详情失败: {str(e)}"


@tool
def search_models(keyword: str) -> str:
    """搜索型号，支持按型号编码、名称或成品编码进行模糊搜索。
    
    Args:
        keyword: 搜索关键词
    """
    logger.info(f"🔧 调用工具: search_models | 关键词: {keyword}")
    try:
        models = _search_models(keyword)
        logger.info(f"📄 搜索结果: {len(models)} 个匹配")
        
        if not models:
            return f"未找到包含 '{keyword}' 的型号"
        
        result = f"找到 {len(models)} 个匹配的型号:\n\n"
        for model in models:
            result += (
                f"- {model['model_code']} ({model['model_name']})\n"
                f"  成品编码: {model['product_code']}\n"
                f"  创建人: {model['creator']}\n"
            )
        
        return result
    except Exception as e:
        logger.error(f"❌ 搜索型号失败: {str(e)}")
        return f"搜索型号失败: {str(e)}"


@tool
def generate_insert_sql(
    device_model: str,
    product_code: str,
    is_special: bool,
    status_role: str | Dict[str, str],
    alarm_role: str | Dict[str, str]
) -> str:
    """根据设备信息生成 INSERT SQL 语句，用于插入门体配置数据。
    
    Args:
        device_model: 设备型号，例如 "M001"
        product_code: 设备编码（成品编码），例如 "PC001"
        is_special: 是否是特殊型号，true 或 false
        status_role: 门体上报字段对应关系，支持两种格式：
            - 字典格式：{"门状态": "door_status", "锁状态": "lock_status"}
            - 文本格式："门体1 door1status 门体2 door2status"（空格分隔）
        alarm_role: 门体告警字段对应关系，支持两种格式：
            - 字典格式：{"入侵告警": "intrusion_alarm", "故障告警": "fault_alarm"}
            - 文本格式："门体1 door1alarm 门体2 door2alarm"（空格分隔）
    
    Returns:
        生成的 INSERT SQL 语句
    
    Example:
        >>> # 使用字典格式
        >>> generate_insert_sql(
        ...     device_model="M001",
        ...     product_code="PC001",
        ...     is_special=True,
        ...     status_role={"门状态": "door_status", "锁状态": "lock_status"},
        ...     alarm_role={"入侵告警": "intrusion_alarm"}
        ... )
        
        >>> # 使用文本格式（更常用）
        >>> generate_insert_sql(
        ...     device_model="M001",
        ...     product_code="PC001",
        ...     is_special=True,
        ...     status_role="门体1 door1status 门体2 door2status",
        ...     alarm_role="门体1 door1alarm 门体2 door2alarm"
        ... )
    """
    logger.info(f"🔧 调用工具: generate_insert_sql | 型号: {device_model} | 编码: {product_code}")
    try:
        import json
        import re
        
        def parse_role_mapping(role_data: str | Dict[str, str]) -> Dict[str, str]:
            """解析字段映射关系，支持字典和文本两种格式"""
            if isinstance(role_data, dict):
                return role_data
            
            # 文本格式解析：按空格分割，成对提取
            if isinstance(role_data, str):
                # 去除多余空格
                text = role_data.strip()
                if not text:
                    return {}
                
                # 按空格分割
                parts = text.split()
                
                # 检查是否为偶数个元素（key-value 成对）
                if len(parts) % 2 != 0:
                    raise ValueError(f"字段映射格式错误：需要成对的 key-value，但收到 {len(parts)} 个元素")
                
                # 构建字典
                result = {}
                for i in range(0, len(parts), 2):
                    key = parts[i]
                    value = parts[i + 1]
                    result[key] = value
                
                return result
            
            raise ValueError("不支持的数据类型")
        
        # 解析两种格式的输入
        status_dict = parse_role_mapping(status_role)
        alarm_dict = parse_role_mapping(alarm_role)
        
        logger.info(f"📊 解析结果 | 上报字段: {len(status_dict)} 个 | 告警字段: {len(alarm_dict)} 个")
        
        # 将字典转换为 JSON 字符串
        status_role_json = json.dumps(status_dict, ensure_ascii=False)
        alarm_role_json = json.dumps(alarm_dict, ensure_ascii=False)
        
        # 构建 SQL 语句
        sql = (
            f"insert into door (device_model, product_code, is_special, status_role, alarm_role) "
            f"values ('{device_model}', '{product_code}', {str(is_special).lower()}, "
            f"'{status_role_json}', '{alarm_role_json}');"
        )
        
        logger.info(f"✅ SQL 生成成功")
        
        # 构建友好的返回信息
        result = f"生成的 SQL 语句:\n\n```sql\n{sql}\n```\n\n"
        result += f"**参数说明:**\n"
        result += f"- 设备型号: {device_model}\n"
        result += f"- 设备编码: {product_code}\n"
        result += f"- 特殊型号: {is_special}\n"
        result += f"- 上报字段 ({len(status_dict)} 个):\n"
        for key, value in status_dict.items():
            result += f"  • {key} → {value}\n"
        result += f"- 告警字段 ({len(alarm_dict)} 个):\n"
        for key, value in alarm_dict.items():
            result += f"  • {key} → {value}\n"
        
        return result
    
    except Exception as e:
        logger.error(f"❌ 生成 SQL 失败: {str(e)}")
        return f"生成 SQL 语句失败: {str(e)}"


# ==================== 统一导出 ====================

MODEL_ADAPTER_TOOLS = [
    list_all_models,
    get_model_features,
    get_model_details,
    search_models
]
