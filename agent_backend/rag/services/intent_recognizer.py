"""
意图识别服务
识别用户输入是否需要调用工具表单
"""
import re
from typing import Dict, Any, Optional, List
from services.tool_metadata import get_tool_metadata, list_available_tools


class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self):
        # 关键词映射表：关键词 -> 工具名称
        self.keyword_mapping = {
            # SQL生成相关
            "生成sql": "generate_insert_sql",
            "创建sql": "generate_insert_sql",
            "插入sql": "generate_insert_sql",
            "门体配置": "generate_insert_sql",
            "门体数据": "generate_insert_sql",
            "insert": "generate_insert_sql",
            
            # 可以添加更多映射
            # "创建分支": "create_branch",
            # "新建分支": "create_branch",
            # "branch": "create_branch",
        }
        
        # 编译正则表达式模式
        self.patterns = {}
        for keyword in self.keyword_mapping.keys():
            # 使用不区分大小写的正则
            self.patterns[keyword] = re.compile(re.escape(keyword), re.IGNORECASE)
    
    def recognize(self, message: str) -> Dict[str, Any]:
        """识别用户意图
        
        Args:
            message: 用户输入的消息
            
        Returns:
            识别结果字典，包含：
            - intent_type: "tool_form" | "chat" | "query"
            - tool_name: 工具名称（如果是tool_form）
            - confidence: 置信度 (0-1)
            - matched_keyword: 匹配的关键词
        """
        message_lower = message.lower()
        
        # 1. 检查是否匹配已知工具的关键词
        best_match = None
        highest_confidence = 0.0
        
        for keyword, tool_name in self.keyword_mapping.items():
            if self.patterns[keyword].search(message_lower):
                # 计算置信度：基于关键词长度和匹配位置
                confidence = self._calculate_confidence(keyword, message_lower)
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = {
                        "tool_name": tool_name,
                        "keyword": keyword,
                        "confidence": confidence
                    }
        
        # 2. 如果找到匹配，返回工具表单意图
        if best_match and best_match["confidence"] > 0.5:
            tool_metadata = get_tool_metadata(best_match["tool_name"])
            
            if tool_metadata:
                return {
                    "intent_type": "tool_form",
                    "tool_name": best_match["tool_name"],
                    "form_schema": tool_metadata,
                    "confidence": best_match["confidence"],
                    "message": f"检测到您需要「{tool_metadata['display_name']}」，请填写以下信息："
                }
        
        # 3. 默认返回聊天意图
        return {
            "intent_type": "chat",
            "tool_name": None,
            "form_schema": None,
            "confidence": 0.0,
            "message": ""
        }
    
    def _calculate_confidence(self, keyword: str, message: str) -> float:
        """计算匹配置信度
        
        Args:
            keyword: 匹配的关键词
            message: 用户消息（小写）
            
        Returns:
            置信度 (0-1)
        """
        # 基础置信度
        base_confidence = 0.6
        
        # 关键词长度权重：更长的关键词通常更准确
        length_weight = min(len(keyword) / 20.0, 0.2)
        
        # 位置权重：在开头匹配权重更高
        position_weight = 0.0
        if message.startswith(keyword):
            position_weight = 0.2
        elif keyword in message[:len(message)//2]:
            position_weight = 0.1
        
        confidence = base_confidence + length_weight + position_weight
        return min(confidence, 1.0)
    
    def add_keyword_mapping(self, keyword: str, tool_name: str):
        """动态添加关键词映射
        
        Args:
            keyword: 关键词
            tool_name: 对应的工具名称
        """
        self.keyword_mapping[keyword] = tool_name
        self.patterns[keyword] = re.compile(re.escape(keyword), re.IGNORECASE)
    
    def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """获取所有可用工具的信息
        
        Returns:
            工具信息列表
        """
        return list_available_tools()


# 全局单例
_intent_recognizer = None


def get_intent_recognizer() -> IntentRecognizer:
    """获取意图识别器单例"""
    global _intent_recognizer
    if _intent_recognizer is None:
        _intent_recognizer = IntentRecognizer()
    return _intent_recognizer
