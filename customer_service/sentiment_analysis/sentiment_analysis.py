"""
情感分析器
分析用户输入的情感倾向（基于预训练模型）
"""
from typing import Dict
from transformers import pipeline  # 复用意图识别中使用的Transformers库


class SentimentAnalyzer:
    """
    情感分析器类（基于预训练模型）
    """

    def __init__(self, model_name: str = "hfl/chinese-roberta-wwm-ext-emotion"):
        """
        初始化情感分析器

        Args:
            model_name (str): 预训练情感分析模型名称或路径
                              推荐模型：
                              - hfl/chinese-roberta-wwm-ext-emotion (中文情感分类)
                              *推荐三分类模型*：
                              - hfl/chinese-roberta-wwm-ext-emotion (中文情感分类，原生支持 positive/negative/neutral)
                              二分类模型（不推荐）：
                              — uer/roberta-base-finetuned-dianping-chinese (仅支持 positive/negative)
                              - uer/roberta-base-finetuned-dianping-chinese (中文电商评论)

        """
        # 加载情感分析pipeline（自动下载模型）
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=False  # 仅返回最高置信度结果
        )
        # 模型标签映射（不同模型标签可能不同，需根据实际模型调整）
        # 当前模型(uer/dianping)标签：'positive'->积极, 'negative'->消极
        self.label_mapping = {
            "positive": "positive",   # 积极情感
            "negative": "negative",   # 消极情感
            "neutral": "neutral"      # 中性情感（部分模型支持）
        }

    def analyze(self, text: str) -> Dict[str, float]:
        """
        分析文本情感

        Args:
            text (str): 待分析文本（用户输入）

        Returns:
            dict: 情感分析结果，包含'label'(情感标签)和'confidence'(置信度)
        """
        if not text.strip():
            return {"label": "neutral", "confidence": 0.5}

        # 使用预训练模型预测情感
        result = self.classifier(text)[0]

        # 映射模型输出标签（优化：提取核心情感倾向，支持带后缀的标签）
        raw_label = result["label"].lower()  # 转为小写，避免大小写问题
        if "positive" in raw_label:
            label = "positive"
        elif "negative" in raw_label:
            label = "negative"
        elif "neutral" in raw_label:
            label = "neutral"
        else:
            label = "neutral"  # 未知标签默认中性

        confidence = round(result["score"], 4)  # 保留4位小数

        # 处理模型未明确返回中性的情况（通过置信度阈值判断）
        if confidence < 0.6:  # 若最高置信度低于阈值，判定为中性
            label = "neutral"
            confidence = 0.5

        return {
            "label": label,
            "confidence": confidence
        }


# 验证用main方法
if __name__ == "__main__":
    # 初始化情感分析器
    analyzer = SentimentAnalyzer()

    # 测试用例（覆盖客服场景常见情感）
    test_cases = [
        # 消极情感（投诉/不满）
        "你们的服务太差了！订单三天都没发货",
        "商品有质量问题，客服还不处理",
        "退款申请提交一周了还没到账，太差劲了",

        # 积极情感（满意/表扬）
        "商品很好用，物流也快，点赞！",
        "客服很耐心，问题解决了，谢谢",
        "优惠券使用很方便，活动很划算",

        # 中性情感（咨询/查询）
        "请问我的订单什么时候能到？",
        "修改收货地址需要什么手续？",
        "这个产品有哪些功能特性？"
    ]

    # 执行测试并打印结果
    print("===== 情感分析验证结果 =====")
    for i, text in enumerate(test_cases, 1):
        result = analyzer.analyze(text)
        print(f"测试用例 {i}: {text}")
        print(f"情感分析结果: {result}\n")