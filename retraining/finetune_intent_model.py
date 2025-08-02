#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/29 23:58
# @Author  : zhangpeng /zpskt
# @File    : finetune_intent_model.py.py
# @Software: PyCharm
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# 1. 加载数据
def load_data(file_path):
    """加载标注数据并分割为训练集和验证集"""
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 数据清洗：移除包含NaN的行
    data = data.dropna(subset=['text', 'intent'])

    # 提取文本和标签
    texts = data['text'].tolist()
    labels = data['intent'].tolist()

    # 为标签分配ID
    unique_intents = list(set(labels))
    print(f"数据中实际唯一意图数量: {len(unique_intents)}")
    print(f"具体意图列表: {unique_intents}")  # 检查是否有重复或错误标签

    intent2id = {intent: i for i, intent in enumerate(unique_intents)}
    id2intent = {i: intent for intent, i in intent2id.items()}

    # 转换为ID
    label_ids = [intent2id[label] for label in labels]

    # 分割训练集和验证集（8:2）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
    )

    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    }), intent2id, id2intent


# 2. 加载模型和分词器
model_name = "facebook/bart-large-mnli"  # 基于原有零样本模型微调
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 需要确保unique_intents在load_data函数执行后可用
dataset, intent2id, id2intent = load_data("../data/intent_train.csv")
num_labels = len(intent2id)  # 动态获取类别数

# 加载模型（序列分类任务）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels, # 一共七个
    ignore_mismatched_sizes=True,  # 关键参数：允许加载不匹配的参数，会自动初始化新的分类层
    id2label=id2intent,  # 确保键是整数类型
    label2id=intent2id   # 确保值是整数类型
)



# 3. 数据预处理（关键修改：添加padding="max_length"强制统一长度）
def preprocess_function(examples):
    """对文本进行分词处理，强制统一长度"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"  # 强制所有样本都pad到max_length长度
    )

# 应用预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)
# 验证数据形状（修正打印方式，确保每个样本长度一致）
print("训练集样本数:", len(tokenized_dataset["train"]))
print("单个样本input_ids长度:", len(tokenized_dataset["train"][0]["input_ids"]))  # 应输出128
print("标签示例（确保是整数）:", tokenized_dataset["train"][0]["label"])  # 应输出整数

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 定义评估指标
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    # 从元组中提取 logits（模型输出的第一个元素）和标签
    predictions, labels = eval_pred
    # 若 predictions 是元组（如包含 logits 和其他输出），取第一个元素（logits）
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # 关键修复：提取 logits 张量

    # # 处理张量转 numpy
    # if isinstance(predictions, torch.Tensor):
    #     predictions = predictions.cpu().numpy()
    # if isinstance(labels, torch.Tensor):
    #     labels = labels.cpu().numpy()

    # 确保形状正确
    assert len(predictions.shape) == 2, f"预期2维预测结果，实际为{len(predictions.shape)}维"
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./intent_model_finetuned",  # 模型保存路径
    learning_rate=2e-5,  # 学习率（较小值避免破坏预训练知识）
    per_device_train_batch_size=8,  # 训练批次大小
    per_device_eval_batch_size=8,  # 验证批次大小
    num_train_epochs=3,  # 训练轮次（数据少时3-5轮即可）
    weight_decay=0.01,  # 权重衰减防止过拟合
    save_strategy="epoch",  # 每轮结束后保存模型
    eval_strategy="epoch",  # 每轮结束后进行评估（与保存策略匹配）
    load_best_model_at_end=True,  # 最后加载最佳模型
    logging_dir="./logs",  # 日志路径
    # no_cuda=True,  # 强制使用CPU，规避MPS问题
)

# 6. 初始化Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 7. 保存最终模型和分词器
model.save_pretrained("./final_intent_model")
tokenizer.save_pretrained("./final_intent_model")
print("模型训练完成并保存至 ./final_intent_model")