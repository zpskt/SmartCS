#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：01-langchain核心组件-智能体.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/21 13:59 
@Description： 
'''
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.prompts import SystemMessagePromptTemplate

chat_model = ChatTongyi(model="qwen3-max")

# 存储临时数据的字典（实际项目中可以用数据库或缓存）
temp_data_store = {}

@tool(description="获取型号信息")
def get_model_info(model: str) -> str:
    """获取型号信息"""
    return f" 型号：{model}有以下功能：节能，定时开关机，AI对话"

@tool(description="复制型号功能 - 将源型号的功能复制到目标型号")
def copy_model_info(source_model: str, target_model: str) -> str:
    """
    执行复制操作：将源型号的功能复制到目标型号
    :param source_model: 源型号
    :param target_model: 目标型号
    :return: 复制结果
    """
    # TODO: 这里替换为实际的复制逻辑
    return f" 型号：{source_model}的功能已经复制给型号{target_model}"

@tool(description="判断目标型号是否存在")
def model_exists(model: str) -> dict:
    """
    检查型号是否存在
    :param model: 要检查的型号
    :return: {"exists": True/False, "message": "提示信息"}
    """
    # 调用外部接口
    # TODO: 这里替换为实际的数据库查询或 API 调用
    # 示例：假设只有特定型号存在
    existing_models = ["X1", "X2", "X3", "Pro1", "Pro2"]

    if model in existing_models:
        return {"exists": True, "message": f"型号 {model} 存在"}
    else:
        return {"exists": False, "message": f"型号 {model} 不存在，请先创建该型号"}

# 工具列表
tools = [get_model_info, copy_model_info, model_exists]


def create_copy_model_agent():
    """
    创建型号复制智能体

    流程：
    1. 收集信息：询问用户源型号和目标型号
    2. 验证检查：检查目标型号是否存在
    3. 确认操作：让用户确认复制
    4. 执行复制：执行复制逻辑
    """

    # 自定义提示词 - 控制对话流程
    system_prompt = """你是一个型号复制助手，帮助用户完成型号功能的复制操作。

请严格按照以下流程与用户交互：

【第 1 步：信息收集】
- 当用户提出要复制型号时，如果用户没有提供完整信息，询问用户提供：
  * 源型号（要复制功能的型号）
  * 目标型号（要复制到的型号）
- 示例："请告诉我源型号和目标型号，例如：把 X1 的功能复制到 X2"

【第 2 步：验证检查】
- 收到两个型号后，先调用 check_model_exists 工具检查目标型号是否存在
- 如果不存在，告知用户："目标型号 {model} 不存在，请先创建该型号"
- 如果存在，进入下一步

【第 3 步：确认操作】
- 目标型号存在时，输出确认信息让用户确认：
  "确认要将 {source_model} 的功能复制到 {target_model} 吗？请回复'确认'或'取消'"
- 等待用户明确确认

【第 4 步：执行复制】
- 用户确认后，调用 copy_model_function 工具执行复制
- 完成后告知用户："✓ 复制完成！{source_model} 的所有功能已复制到 {target_model}"

注意事项：
- 必须按顺序执行，不能跳过任何步骤
- 用户未确认前，不要执行复制操作
- 如果目标型号不存在，停止流程并提示用户
"""

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,  # 关键修正点！
        state_schema=AgentState,
    )
    # prompt_template = SystemMessagePromptTemplate(system_prompt)
    # agent.invoke(prompt_template)

    return agent


def run_copy_model_flow(user_input: str):
    """
    运行型号复制流程

    :param user_input: 用户初始输入
    :return: 最终结果
    """
    agent = create_copy_model_agent()

    # 初始化状态
    initial_state = AgentState(
        messages=[],
        source_model="",
        target_model="",
        step="collect_info"
    )

    # 运行智能体
    result = agent.invoke({
        "messages": [user_input],
        **initial_state
    })

    return result["messages"][-1]


# 使用示例
if __name__ == "__main__":
    # 示例 1：用户直接提供完整信息
    print("=== 示例 1：用户直接提供完整信息 ===")
    response = run_copy_model_flow("我想把 X1 的功能复制到 X2")
    print(response)
    print("\n")


def use_agent_by_static():
    """
    静态模型在创建智能体时配置一次，并在整个执行过程中保持不变

    :return:
    """
    agent = create_agent(
        "openai:gpt-5",
        tools= tools,
    )
