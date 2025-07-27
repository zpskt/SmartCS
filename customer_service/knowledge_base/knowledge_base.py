"""
知识库管理器
维护和查询客服知识库
"""

class KnowledgeBase:
    """
    知识库管理器类
    """
    
    def __init__(self):
        """
        初始化知识库
        """
        # 示例知识库数据
        self.knowledge = [
            {
                'id': 1,
                'question': '如何退货',
                'answer': '您可以在订单页面申请退货，填写退货原因并提交申请。我们的客服会在1-3个工作日内处理您的申请。',
                'category': '售后服务'
            },
            {
                'id': 2,
                'question': '运费多少',
                'answer': '我们的运费标准是：满99元免运费，不满99元收取8元运费。',
                'category': '购物咨询'
            },
            {
                'id': 3,
                'question': '发货时间',
                'answer': '我们通常在您下单后24小时内发货，节假日可能会有延迟。',
                'category': '购物咨询'
            }
        ]
    
    def query(self, question):
        """
        查询知识库
        
        Args:
            question (str): 用户问题
            
        Returns:
            list: 匹配的知识条目
        """
        # 简单的关键词匹配实现
        # 实际应用中应该使用向量检索或语义匹配
        
        results = []
        question_lower = question.lower()
        
        for item in self.knowledge:
            # 检查问题是否包含知识库中的关键词
            if any(keyword in question_lower for keyword in item['question'].lower().split()):
                results.append(item)
        
        return results
    
    def add_knowledge(self, question, answer, category=''):
        """
        添加新知识到知识库
        
        Args:
            question (str): 问题
            answer (str): 答案
            category (str): 分类
        """
        new_id = max([item['id'] for item in self.knowledge], default=0) + 1
        self.knowledge.append({
            'id': new_id,
            'question': question,
            'answer': answer,
            'category': category
        })
    
    def update_knowledge(self, knowledge_id, question=None, answer=None, category=None):
        """
        更新知识库条目
        
        Args:
            knowledge_id (int): 知识条目ID
            question (str, optional): 新问题
            answer (str, optional): 新答案
            category (str, optional): 新分类
        """
        for item in self.knowledge:
            if item['id'] == knowledge_id:
                if question is not None:
                    item['question'] = question
                if answer is not None:
                    item['answer'] = answer
                if category is not None:
                    item['category'] = category
                break