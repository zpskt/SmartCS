"""
情感分析器
分析用户输入的情感倾向
"""

class SentimentAnalyzer:
    """
    情感分析器类
    """
    
    def __init__(self):
        """
        初始化情感分析器
        """
        # 情感词典（简化版）
        self.positive_words = ['好', '棒', '喜欢', '满意', '高兴', '开心', '赞']
        self.negative_words = ['差', '坏', '讨厌', '不满', '生气', '愤怒', '失望']
    
    def analyze(self, text):
        """
        分析文本情感
        
        Args:
            text (str): 待分析文本
            
        Returns:
            dict: 情感分析结果
        """
        # 简单的情感词匹配实现
        # 实际应用中应该使用更复杂的情感分析模型
        
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment_label = 'positive'
            confidence = positive_count / (positive_count + negative_count + 1)  # 加1避免除零
        elif negative_count > positive_count:
            sentiment_label = 'negative'
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment_label = 'neutral'
            confidence = 0.5
            
        return {
            'label': sentiment_label,
            'confidence': confidence
        }