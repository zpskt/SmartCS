# SmartCS - 智能客服系统
Smart Customer Service
基于大语言模型（LLM）构建的智能客服系统，提供7*24小时智能问答服务。

## 功能特性

- 多场景客户接待解决方案
- 对话管理模块
- 意图识别模块
- 情感分析模块
- 知识库动态更新
- 多轮回话优化

## 项目结构
### 意图分析

默认的意图分析使用facebook/bart-large-mnli模型ing，因为这个模型是零样本模型。
当然模型效果针对特定领域肯定是不完美的，所以在retraining/finetune_intent_model.py中可以对其进行微调。
## 环境配置
```shell
conda create -n smart-cs --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
conda activate smart-cs
pip install -r requirements.txt
```