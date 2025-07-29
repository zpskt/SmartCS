#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/29 23:58
# @Author  : zhangpeng /zpskt
# @File    : finetune_intent_model.py.py
# @Software: PyCharm
import evaluate
import numpy as np
import pandas as pd
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
    
    # 提取文本和标签
    texts = data['text'].tolist()
    labels = data['intent'].tolist()

    # 为标签分配ID
    unique_intents = list(set(labels))
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
# todo 这里报错了
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2intent,
    label2id=intent2id
)


# 3. 数据预处理
def preprocess_function(examples):
    """对文本进行分词处理"""
    return tokenizer(examples["text"], truncation=True, max_length=128)


# 应用预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 定义评估指标
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
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
    evaluation_strategy="epoch",  # 每轮结束后评估
    save_strategy="epoch",  # 每轮结束后保存模型
    load_best_model_at_end=True,  # 最后加载最佳模型
    logging_dir="./logs",  # 日志路径
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