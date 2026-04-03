#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS 
@File    ：model_copy_agent.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/3/21
@Description：型号复制智能体 - 封装类
'''
from typing import List, Dict, Optional

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langchain.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime


class ModelCopyAgent:
    """
    型号复制智能体类

    功能：
    - 帮助用户完成型号功能的复制操作
    - 包含完整的信息收集、验证、确认、执行流程
    """

    def __init__(
            self,
            model_name: str = "qwen3-max",
            temperature: float = 0.7,
            existing_models: Optional[List[str]] = None,
            model_info_map: Optional[Dict[str, str]] = None
    ):
        """
        初始化型号复制智能体

        :param model_name: 模型名称
        :param temperature: 温度参数
        :param existing_models: 已存在的型号列表（用于验证）
        :param model_info_map: 型号功能映射字典
        """
        self.model_name = model_name
        self.temperature = temperature
        self.existing_models = existing_models or ["X1", "X2", "X3", "Pro1", "Pro2"]
        self.model_info_map = model_info_map or {
            "X1": "节能模式、定时开关机、语音控制",
            "X2": "智能温控、远程控制、能耗统计",
            "X3": "AI学习、自动调节、场景模式",
            "Pro1": "专业模式、数据导出、多用户管理",
            "Pro2": "企业级功能、API接口、实时监控"
        }

        # 初始化模型
        self.chat_model = ChatTongyi(
            model=self.model_name,
            temperature=self.temperature
        )

        self.middleware = self._create_middleware()

        # 创建工具
        self.tools = self._create_tools()

        # 创建内存记忆
        self.memory = InMemorySaver()

        # 创建智能体
        self.agent = self._create_agent()


        # 会话配置
        self.config = {"configurable": {"thread_id": "default_user"}}


    def _create_middleware(self):
        @after_model
        def check_message(state: AgentState, runtime: ToolRuntime) -> dict | None:
            """删除聊天信息里面的敏感词"""
            STOP_WORDS = ["password", "secret"]
            last_message = state["messages"][-1]
            print("last_message: ", last_message)
            return None
        return [check_message]

    def _create_tools(self):
        """创建工具"""

        @tool(description="获取型号信息")
        def get_model_info(model: str) -> str:
            """获取型号信息"""
            info = self.model_info_map.get(
                model,
                f"基础功能：开关机、模式切换、温度调节"
            )
            return f"型号 {model} 有以下功能：{info}"

        @tool(description="复制型号功能 - 将源型号的功能复制到目标型号")
        def copy_model_info(source_model: str, target_model: str) -> str:
            """
            执行复制操作：将源型号的功能复制到目标型号
            """
            # 这里可以调用实际的复制逻辑
            return f"✓ 复制完成！{source_model} 的所有功能已复制到 {target_model}"

        @tool(description="判断目标型号是否存在")
        def model_exists(model: str) -> dict:
            """
            检查型号是否存在
            """
            if model in self.existing_models:
                return {"exists": True, "message": f"✅ 型号 {model} 存在"}
            else:
                return {"exists": False, "message": f"❌ 型号 {model} 不存在，请先创建该型号"}

        return [get_model_info, copy_model_info, model_exists]

    def _create_agent(self):
        """创建智能体"""

        system_prompt = """你是一个型号复制助手，帮助用户完成型号功能的复制操作。

        请严格按照以下流程与用户交互，**记住你已经收集到的信息**：

        【第 1 步：信息收集】
        - 当用户提出要复制型号时，如果用户没有提供完整信息，询问用户提供：
          * 源型号（要复制功能的型号）
          * 目标型号（要复制到的型号）
        - 示例："请告诉我源型号和目标型号，例如：把 X1 的功能复制到 X2"
        - **重要**：一旦用户提供了型号信息，必须记住这两个型号，不要在后续步骤中忘记！

        【第 2 步：验证检查】
        - 收到两个型号后，先调用 model_exists 工具检查目标型号是否存在
        - 如果不存在，告知用户目标型号不存在，请先创建该型号
        - 如果存在，进入下一步

        【第 3 步：确认操作】
        - 目标型号存在时，输出确认信息让用户确认：
          "确认要将 {source_model} 的功能复制到 {target_model} 吗？请回复'确认'或'取消'"
        - **重要**：这里的 {source_model} 和 {target_model} 必须使用第1步中用户提供的型号
        - 等待用户明确确认

        【第 4 步：执行复制】
        - 用户确认后，**立即调用 copy_model_info 工具**执行复制
        - 工具调用时必须传入正确的 source_model 和 target_model（来自第1步）
        - 完成后告知用户："✓ 复制完成！{source_model} 的所有功能已复制到 {target_model}"

        【关键规则 - 必须遵守】：
        1. 当用户回复"确认"时，你必须从对话历史中找到之前提到的源型号和目标型号
        2. 不要在确认后重新询问型号信息，而是直接执行复制
        3. 如果用户回复"确认"但你找不到型号信息，可以从历史消息中查找：例如"把 X1 的功能复制到 X2"这样的语句
        4. 执行复制后，流程结束

        【示例对话流程】：
        用户：把 X1 的功能复制到 X2
        助手：（调用 model_exists 检查 X2）确认要将 X1 的功能复制到 X2 吗？请回复'确认'或'取消'
        用户：确认
        助手：（调用 copy_model_info 工具，传入 source_model="X1", target_model="X2"）✓ 复制完成！X1 的所有功能已复制到 X2

        注意事项：
        - 必须按顺序执行，不能跳过任何步骤
        - 用户未确认前，不要执行复制操作
        - 如果目标型号不存在，停止流程并提示用户
        - 如果用户想查看型号信息，可以使用 get_model_info 工具
        """
        agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            system_prompt=system_prompt,
            state_schema=AgentState,
            checkpointer=self.memory,
            middleware=self.middleware,

        )

        return agent

    def set_session_id(self, session_id: str):
        """
        设置会话ID（支持多会话）

        :param session_id: 会话ID
        """
        self.config = {"configurable": {"thread_id": session_id}}

    def get_session_id(self) -> str:
        """获取当前会话ID"""
        return self.config["configurable"]["thread_id"]

    def invoke(self, user_input: str) -> str:
        """
        调用智能体处理用户输入

        :param user_input: 用户输入
        :return: AI响应内容
        """
        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=self.config
            )
            # return response
            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                return last_message.content
            else:
                return "抱歉，我暂时无法处理你的请求。"

        except Exception as e:
            return f"处理时出现错误：{str(e)}"

    def invoke_with_history(self, messages: List[Dict[str, str]]) -> str:
        """
        带历史消息的调用

        :param messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
        :return: AI响应内容
        """
        try:
            # 转换消息格式
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # 调用智能体
            response = self.agent.invoke(
                {"messages": langchain_messages},
                config=self.config
            )

            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                return last_message.content
            else:
                return "抱歉，我暂时无法处理你的请求。"

        except Exception as e:
            return f"处理时出现错误：{str(e)}"

    def reset_session(self):
        """重置当前会话"""
        self.config = {"configurable": {"thread_id": "default_user"}}

    def add_model(self, model_name: str, info: str = ""):
        """
        添加新型号

        :param model_name: 型号名称
        :param info: 型号功能描述
        """
        if model_name not in self.existing_models:
            self.existing_models.append(model_name)

        if info and model_name not in self.model_info_map:
            self.model_info_map[model_name] = info

    def remove_model(self, model_name: str):
        """
        删除型号

        :param model_name: 型号名称
        """
        if model_name in self.existing_models:
            self.existing_models.remove(model_name)

        if model_name in self.model_info_map:
            del self.model_info_map[model_name]

    def get_available_models(self) -> List[str]:
        """获取可用型号列表"""
        return self.existing_models.copy()

    def get_model_info_dict(self) -> Dict[str, str]:
        """获取型号信息字典"""
        return self.model_info_map.copy()

    def update_model_info(self, model_name: str, info: str):
        """
        更新型号功能信息

        :param model_name: 型号名称
        :param info: 新的功能描述
        """
        self.model_info_map[model_name] = info


# 单例模式，避免重复创建
_agent_instance = None


def get_model_copy_agent(
        model_name: str = "qwen3-max",
        temperature: float = 0.7
) -> ModelCopyAgent:
    """
    获取型号复制智能体实例（单例）

    :param model_name: 模型名称
    :param temperature: 温度参数
    :return: ModelCopyAgent 实例
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ModelCopyAgent(
            model_name=model_name,
            temperature=temperature
        )
    return _agent_instance

if __name__ == '__main__':
    agent = get_model_copy_agent(model_name="qwen3-max", temperature=0.7)
    agent.invoke("将 X1 的功能复制到 X2 中。")
    response = agent.invoke("确认")
    for msg in response["messages"]:
        msg.pretty_print()

