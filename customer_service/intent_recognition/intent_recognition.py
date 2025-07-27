"""
意图识别器
识别用户输入的意图和关键信息
"""
from transformers import BertForSequenceClassification, BertTokenizer

class IntentRecognizer:
    """
    意图识别器类
    """
    
    def __init__(self,model_name='model/bert-base-chinese'):
        """
        初始化意图识别器
        """
        # 预定义的意图类型
        self.intents = {
            'greeting': ['你好', '您好', 'hello', 'hi'],
            'goodbye': ['再见', '拜拜', 'bye'],
            'question': ['什么', '如何', '怎么', '为什么', '?'],
            'complaint': ['投诉', '抱怨', '不满', '生气'],
            'praise': ['表扬', '夸奖', '满意', '很好']
        }
        self.model= BertForSequenceClassification.from_pretrained(model_name)
    
    def recognize(self, text):
        """
        识别用户输入的意图
        
        Args:
            text (str): 用户输入文本
            
        Returns:
            dict: 意图识别结果
        """
        # 简单的关键词匹配实现
        # 实际应用中应该使用更复杂的NLP模型
        
        text_lower = text.lower()
        
        for intent_name, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'name': intent_name,
                        'confidence': 0.8,
                        'entities': self._extract_entities(text)
                    }
        
        # 默认意图
        return {
            'name': 'unknown',
            'confidence': 0.5,
            'entities': self._extract_entities(text)
        }
    
    def _extract_entities(self, text):
        """
        提取文本中的实体信息
        
        Args:
            text (str): 输入文本
            
        Returns:
            list: 实体列表
        """
        # 简单实现，实际应该使用NER模型
        return []