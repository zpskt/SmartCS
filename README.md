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
## 环境配置
```shell
conda create -n smart-cs --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.10
conda activate smart-cs
pip install -r requirements.txt
```
```shell
pip install langchain_chroma -i https://mirrors.aliyun.com/pypi/simple/
pip install langchain_community -i https://mirrors.aliyun.com/pypi/simple/
pip install langchain_core -i https://mirrors.aliyun.com/pypi/simple/
pip install langchain_text_splitters -i https://mirrors.aliyun.com/pypi/simple/
pip install dashscope -i https://mirrors.aliyun.com/pypi/simple/
pip install streamlit -i https://mirrors.aliyun.com/pypi/simple/

```