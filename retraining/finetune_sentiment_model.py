#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""情感分析模型微调脚本（生成本地模型）"""
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


def load_sentiment_data(file_path):
    """加载情感标注数据（格式：text,sentiment）"""
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['text', 'sentiment'])  # 清洗空值
    
    # 提取唯一情感标签并映射ID
    unique_sentiments = list(set(data['sentiment'].tolist()))
    sentiment2id = {s: i for i, s in enumerate(unique_sentiments)}
    id2sentiment = {i: s for s, i in sentiment2id.items()}
    
    # 转换标签为ID
    texts = data['text'].tolist()
    labels = [sentiment2id[s] for s in data['sentiment'].tolist()]
    
    # 分割训练集/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 转换为Dataset格式
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    return DatasetDict({"train": train_dataset, "validation": val_dataset}), sentiment2id, id2sentiment


# 1. 加载数据和模型
model_name = "hfl/chinese-roberta-wwm-ext-emotion"  # 基于三分类预训练模型微调
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset, sentiment2id, id2sentiment = load_sentiment_data("../data/sentiment_train.csv")  # 情感数据路径
num_labels = len(sentiment2id)
print(f"情感标签: {id2sentiment} (共{num_labels}类)")


# 2. 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 3. 加载模型（序列分类任务）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2sentiment,
    label2id=sentiment2id,
    ignore_mismatched_sizes=True  # 初始化新分类层
)


# 4. 定义评估指标
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1) if isinstance(predictions, tuple) else np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./sentiment_model_finetuned",  # 中间checkpoint路径
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./sentiment_logs"
)


# 6. 训练并保存模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 保存最终模型（用于本地加载）
trainer.model.save_pretrained("./final_sentiment_model")
tokenizer.save_pretrained("./final_sentiment_model")
print("本地情感模型保存至: ./final_sentiment_model")
