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

    def __init__(self, model_name='facebook/bart-large-mnli',
                 model_path: str = None,
                 intents: List[str] = None):
        """
        初始化意图识别器

        Args:
            model_name (str): 预训练模型名称或路径
            model_path (str): 本地微调模型路径（优先于model_name）
            intents (List[str]): 自定义意图列表（可选）
        """
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
        # 1. 优先加载本地微调模型（序列分类任务）
        if model_path:
            self.classifier = pipeline(
                "text-classification",  # 微调模型是序列分类任务
                model=model_path,
                tokenizer=model_path,  # 分词器与模型同路径
                return_all_scores=True  # 返回所有类别的预测分数
            )
            # 从模型配置中自动获取意图标签（7个，与训练时一致）
            self.intents = [
                self.classifier.model.config.id2label[i]
                for i in range(len(self.classifier.model.config.id2label))
            ]
        # 2. 否则加载预训练零样本模型
        else:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name
            )
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
            # 区分模型类型调用（修复参数不匹配问题）
            if hasattr(self.classifier, "task") and self.classifier.task == "text-classification":
                # 本地微调模型：仅需传入文本，返回所有类别分数
                model_output = self.classifier(text)[0]
                model_best = sorted(model_output, key=lambda x: x["score"], reverse=True)[0]
                model_best_intent = model_best["label"]
                model_best_confidence = round(model_best["score"], 4)
            else:
                # 零样本模型：需传入文本、意图列表和multi_label
                model_output = self.classifier(text, self.intents, multi_label=False)
                model_best_intent = model_output["labels"][0]
                model_best_confidence = round(model_output["scores"][0], 4)
            return {
                "intents": [
                    {"intent": rule_intent, "confidence": rule_confidence},
                    {"intent": model_best_intent, "confidence": model_best_confidence}
                ],
                "best_intent": rule_intent,
                "best_confidence": rule_confidence,
                "method": "rule_based"
            }


        # 2. 模型预测（区分本地模型和预训练模型）
        prompt = f"用户在电商平台咨询客服，这句话的意图是：{text}"
        if hasattr(self.classifier, "task") and self.classifier.task == "text-classification":
            # 本地微调模型：输出格式为 list[dict]（每个类别含label和score）
            model_result = self.classifier(prompt)[0]
            # 排序并提取top_k意图
            intent_scores = sorted(
                [{"intent": item["label"], "confidence": round(item["score"], 4)}
                 for item in model_result],
                key=lambda x: x["confidence"], reverse=True
            )
        else:
            # 预训练零样本模型：输出格式为含labels和scores的dict
            model_result = self.classifier(
                prompt, self.intents, multi_label=False,
                hypothesis_template="这句话的意图是{}。"
            )
            intent_scores = [
                {"intent": model_result["labels"][i], "confidence": round(model_result["scores"][i], 4)}
                for i in range(min(top_k, len(model_result["labels"])))
            ]

        # 3. 整理结果（保持统一输出格式）
        return {
            "intents": intent_scores[:top_k],
            "best_intent": intent_scores[0]["intent"],
            "best_confidence": intent_scores[0]["confidence"],
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

    # 方式1：加载本地微调模型（优先推荐）
    # 注意：路径需根据实际目录结构调整（相对于当前文件的位置）
    recognizer = IntentRecognizer(
        model_path="../../retraining/final_intent_model"  # 本地模型路径
    )

    # # 使用零样本-初始化意图识别器
    # recognizer = IntentRecognizer(
    #     model_name="facebook/bart-large-mnli",
    #     intents=intents
    # )
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