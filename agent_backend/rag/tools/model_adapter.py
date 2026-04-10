"""
型号适配相关工具
提供产品型号查询、功能适配、成品编码等信息检索功能
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool


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
    try:
        models = _get_all_models(status=status)
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
        return f"查询型号列表失败: {str(e)}"


@tool
def get_model_features(model_code: str) -> str:
    """查询某个型号适配了哪些功能。当用户询问某个型号的功能时使用此工具。
    
    Args:
        model_code: 型号编码，例如 "M001"
    """
    try:
        model = _get_model_by_code(model_code)
        if not model:
            return f"型号 {model_code} 不存在"
        
        features = model.get('features', [])
        if not features:
            return f"型号 {model_code} 未配置功能"
        
        result = f"型号 {model_code} ({model['model_name']}) 适配的功能:\n"
        for i, feature in enumerate(features, 1):
            result += f"{i}. {feature}\n"
        
        return result
    except Exception as e:
        return f"查询型号功能失败: {str(e)}"


@tool
def get_model_details(model_code: str) -> str:
    """查询某个型号的详细信息，包括成品编码、创建人、创建时间等。
    
    Args:
        model_code: 型号编码，例如 "M001"
    """
    try:
        model = _get_model_by_code(model_code)
        if not model:
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
        
        return result
    except Exception as e:
        return f"查询型号详情失败: {str(e)}"


@tool
def search_models(keyword: str) -> str:
    """搜索型号，支持按型号编码、名称或成品编码进行模糊搜索。
    
    Args:
        keyword: 搜索关键词
    """
    try:
        models = _search_models(keyword)
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
        return f"搜索型号失败: {str(e)}"


# ==================== 统一导出 ====================

MODEL_ADAPTER_TOOLS = [
    list_all_models,
    get_model_features,
    get_model_details,
    search_models
]
