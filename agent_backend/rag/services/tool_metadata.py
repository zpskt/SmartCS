"""
工具元数据管理
定义每个工具的表单Schema，用于前端动态生成表单
"""
from typing import List, Dict, Any, Optional


# 工具元数据注册表
TOOL_METADATA_REGISTRY = {
    "generate_insert_sql": {
        "tool_name": "generate_insert_sql",
        "display_name": "生成门体配置SQL",
        "description": "根据设备信息生成INSERT SQL语句，用于插入门体配置数据",
        "category": "database",
        "icon": "📊",
        "fields": [
            {
                "name": "device_model",
                "label": "设备型号",
                "type": "text",
                "placeholder": "例如: M001",
                "required": True,
                "help_text": "请输入设备型号编码"
            },
            {
                "name": "product_code",
                "label": "成品编码",
                "type": "text",
                "placeholder": "例如: PC001",
                "required": True,
                "help_text": "请输入设备的成品编码"
            },
            {
                "name": "is_special",
                "label": "是否特殊型号",
                "type": "select",
                "options": [
                    {"label": "是", "value": True},
                    {"label": "否", "value": False}
                ],
                "required": True,
                "default": False,
                "help_text": "标记是否为特殊型号"
            },
            {
                "name": "status_role",
                "label": "门体上报字段映射",
                "type": "textarea",
                "placeholder": "例如: 门体1 door1status 门体2 door2status",
                "required": True,
                "help_text": "格式：门体名称 字段名（空格分隔，多个门体用空格隔开）",
                "rows": 3
            },
            {
                "name": "alarm_role",
                "label": "门体告警字段映射",
                "type": "textarea",
                "placeholder": "例如: 门体1 door1alarm 门体2 door2alarm",
                "required": True,
                "help_text": "格式：门体名称 字段名（空格分隔，多个门体用空格隔开）",
                "rows": 3
            }
        ]
    },
    
    # 可以添加更多工具的元数据
    # "create_branch": {
    #     "tool_name": "create_branch",
    #     "display_name": "创建代码分支",
    #     "description": "在代码仓库中创建新的分支",
    #     "category": "git",
    #     "icon": "🌿",
    #     "fields": [
    #         {
    #             "name": "branch_name",
    #             "label": "分支名称",
    #             "type": "text",
    #             "placeholder": "例如: feature/new-feature",
    #             "required": True
    #         },
    #         {
    #             "name": "base_branch",
    #             "label": "基于哪个分支",
    #             "type": "text",
    #             "placeholder": "例如: main",
    #             "required": True,
    #             "default": "main"
    #         }
    #     ]
    # }
}


def get_tool_metadata(tool_name: str) -> Optional[Dict[str, Any]]:
    """获取工具的元数据
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具元数据字典，如果不存在返回None
    """
    return TOOL_METADATA_REGISTRY.get(tool_name)


def list_available_tools() -> List[Dict[str, Any]]:
    """列出所有可用的工具元数据
    
    Returns:
        工具元数据列表
    """
    return list(TOOL_METADATA_REGISTRY.values())


def register_tool_metadata(metadata: Dict[str, Any]):
    """注册新的工具元数据
    
    Args:
        metadata: 工具元数据字典
    """
    tool_name = metadata.get("tool_name")
    if tool_name:
        TOOL_METADATA_REGISTRY[tool_name] = metadata
