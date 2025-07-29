"""
意图识别器
识别用户输入的意图和关键信息
"""
from typing import List, Dict, Tuple

import torch  # PyTorch，用于深度学习推理
from transformers import pipeline


class IntentRecognizer:
    """
    意图识别器类
    """

    def __init__(self, model_name='facebook/bart-large-mnli', intents: List[str] = None):
        """
        初始化意图识别器

        Args:
            model_name (str): 预训练模型名称或路径
            intent_mapping (List[str]): 意图ID到意图名称的映射
        """
        # 1. 加载零样本分类模型
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        # 定义默认客服场景意图（可根据实际需求修改）
        self.default_intents = [
            "查询订单状态",
            "咨询物流信息",
            "申请退货退款",
            "修改收货地址",
            "咨询产品功能",
            "投诉或反馈问题",
            "查询优惠活动",
            "其他问题"
        ]
        # 2. 定义意图（可根据场景自定义）
        self.intents = intents if intents else self.default_intents
        # 3. 定义关键词规则（作为模型的补充）
        self.keyword_rules = {
            "查询订单发货时间或物流状态": ["订单", "发货", "物流", "快递", "单号", "运送"],
            "申请退货、退款或换货": ["退款", "退货", "换货", "退钱", "取消订单"],
            "修改订单信息（包括收货地址、联系方式等）": ["地址", "修改", "更换", "电话", "信息"],
            "咨询产品功能、特性或使用方法": ["功能", "怎么用", "特性", "方法", "操作"],
            "投诉服务质量或反馈问题": ["差", "不好", "垃圾", "投诉", "反馈", "问题"],
            "查询优惠券、促销活动或使用方法": ["优惠券", "优惠", "活动", "折扣", "满减"]
        }
        # 4. 定义否定词（避免误判）
        self.negation_words = {"不", "没", "无", "不是", "没有"}

    def _rule_based_match(self, text: str) -> Tuple[str, float]:
        """基于关键词的规则匹配，作为模型的补充"""
        text_lower = text.lower()
        max_matches = 0
        best_intent = None

        # 检查每个意图的关键词
        for intent, keywords in self.keyword_rules.items():
            matches = 0
            for keyword in keywords:
                # 避免否定结构导致的误判（如"不退款"不应匹配"申请退款"）
                if keyword in text_lower:
                    # 检查关键词前是否有否定词
                    keyword_index = text_lower.index(keyword)
                    has_negation = False
                    for neg_word in self.negation_words:
                        if neg_word in text_lower[:keyword_index]:
                            has_negation = True
                            break
                    if not has_negation:
                        matches += 1

            # 找到匹配最多关键词的意图
            if matches > max_matches:
                max_matches = matches
                best_intent = intent

        # 如果有明确匹配，返回高置信度
        if best_intent and max_matches > 0:
            confidence = 0.7 + (max_matches * 0.05)  # 匹配越多，置信度越高
            return best_intent, min(confidence, 0.95)

        return None, 0.0

    def recognize(self, text: str, top_k: int = 3) -> Dict:
        """优化的意图识别方法：规则匹配 + 模型预测"""
        if not text.strip():
            return {"intents": [], "error": "输入文本不能为空"}

        # 1. 先尝试规则匹配（高优先级）
        rule_intent, rule_confidence = self._rule_based_match(text)
        if rule_intent:
            # 规则匹配到的结果优先返回
            model_result = self.classifier(text, self.intents, multi_label=False)
            return {
                "intents": [
                    {"intent": rule_intent, "confidence": rule_confidence},
                    {"intent": model_result["labels"][0], "confidence": round(model_result["scores"][0], 4)}
                ],
                "best_intent": rule_intent,
                "best_confidence": rule_confidence,
                "method": "rule_based"
            }

        # 2. 规则匹配失败，使用模型预测（带优化提示）
        # 优化提示词，给模型更多上下文
        prompt = f"用户在电商平台咨询客服，这句话的意图是：{text}"
        model_result = self.classifier(
            prompt,  # 使用优化后的提示词
            self.intents,
            multi_label=False,
            hypothesis_template="这句话的意图是{}。"  # 更明确的假设模板
        )

        # 3. 整理结果
        top_indices = range(min(top_k, len(model_result["labels"])))
        return {
            "intents": [
                {
                    "intent": model_result["labels"][i],
                    "confidence": round(model_result["scores"][i], 4)
                }
                for i in top_indices
            ],
            "best_intent": model_result["labels"][0],
            "best_confidence": round(model_result["scores"][0], 4),
            "method": "model_based"
        }


    def _extract_entities(self, text):
        """
        提取文本中的实体信息

        Args:
            text (str): 输入文本

        Returns:
            list: 实体列表
        """
        # 这里可以集成NER模型，当前使用简单的关键词匹配
        entities = []

        # 示例实体类型和关键词
        entity_keywords = {
            'person': ['小明', '张三', '李四'],
            'location': ['北京', '上海', '深圳'],
            'time': ['今天', '明天', '后天']
        }

        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    start = text.index(keyword)
                    entities.append({
                        'type': entity_type,
                        'value': keyword,
                        'start': start,
                        'end': start + len(keyword)
                    })

        return entities

    def _get_all_intents(self, probabilities):
        """
        获取所有意图的概率分布

        Args:
            probabilities (torch.Tensor): 概率分布张量

        Returns:
            list: 所有意图的概率分布
        """
        probs = probabilities.cpu().numpy()[0]
        intent_probs = []

        for i, prob in enumerate(probs):
            intent_name = self.intent_mapping.get(i, f"unknown_{i}")
            intent_probs.append({
                'name': intent_name,
                'confidence': float(prob)
            })

        # 按置信度排序
        intent_probs.sort(key=lambda x: x['confidence'], reverse=True)

        return intent_probs

    def batch_recognize(self, texts):
        """
        批量识别意图

        Args:
            texts (list): 用户输入文本列表

        Returns:
            list: 意图识别结果列表
        """
        results = []
        batch_size = 16  # 可配置的批处理大小

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            for j in range(len(batch_texts)):
                confidence, intent_id = torch.max(probabilities[j], dim=0)
                confidence = confidence.item()
                intent_id = intent_id.item()

                intent_name = self.intent_mapping.get(intent_id, f"unknown_{intent_id}")
                entities = self._extract_entities(batch_texts[j])

                results.append({
                    'name': intent_name,
                    'confidence': confidence,
                    'entities': entities
                })

        return results

# 使用示例
if __name__ == "__main__":
    # 假设有以下意图映射
    intents = [
    "查询订单状态",
    "申请退货退款",
    "修改收货地址",
    "咨询产品功能",
    "投诉或反馈问题",
    "其他问题",
    "咨询天气、日期、新闻等与客服业务完全无关的问题"
]

    # 初始化意图识别器
    recognizer = IntentRecognizer(
        model_name="facebook/bart-large-mnli",
        intents=intents
    )
    # 测试不同用户输入
    test_queries = [
        "我的订单什么时候能发货？",
        "这个商品怎么退款啊？",
        "你们的优惠券怎么用",
        "我想换个收货地址",
        "这个手机有指纹解锁功能吗？",
        "你们的服务太差了！",
        "今天天气怎么样"  # 不属于客服场景的问题
    ]
    for query in test_queries:
        print(f"\n用户输入：{query}")
        result = recognizer.recognize(query)
        print(f"最可能的意图：{result['best_intent']}（置信度：{result['best_confidence']}）")
        print("其他可能的意图：", result["intents"][1:])