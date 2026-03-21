#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：SmartCS
@File    ：copy_model_agent_app.py
@IDE     ：PyCharm
@Author  ：张鹏
@Date    ：2026/3/21
@Description：型号复制智能体 - Streamlit 交互界面
'''
import streamlit as st
import time

# 导入封装好的智能体类
from model_copy_agent import get_model_copy_agent, ModelCopyAgent

# ==================== 配置页面 ====================
st.set_page_config(
    page_title="型号复制智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 会话状态管理 ====================
def init_session_state():
    """初始化会话状态"""

    # 初始化智能体（使用单例模式）
    if "agent" not in st.session_state:
        st.session_state.agent = get_model_copy_agent(
            model_name="qwen3-max",
            temperature=0.7
        )

    # 初始化消息历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 初始化会话ID
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_user"
        st.session_state.agent.set_session_id(st.session_state.session_id)

    # 初始化设置
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "show_typing_effect": True,
            "typing_speed": 0.05,
            "auto_scroll": True
        }


# ==================== 清空对话 ====================
def clear_conversation():
    """清空对话历史"""
    st.session_state.messages = []
    st.rerun()


# ==================== 重置会话 ====================
def reset_session():
    """重置当前会话"""
    st.session_state.agent.reset_session()
    st.session_state.messages = []
    st.success("✅ 会话已重置")
    time.sleep(1)
    st.rerun()


# ==================== 切换会话 ====================
def switch_session(session_id: str):
    """切换会话"""
    st.session_state.session_id = session_id
    st.session_state.agent.set_session_id(session_id)

    # 可选：从数据库加载历史消息
    # st.session_state.messages = load_messages_from_db(session_id)

    st.rerun()


# ==================== 处理用户输入 ====================
def process_user_input(user_input: str):
    """处理用户输入"""
    if not user_input or not user_input.strip():
        return

    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 获取 AI 响应
    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            # 调用智能体
            response_content = st.session_state.agent.invoke(user_input)

        # 显示响应
        if st.session_state.settings["show_typing_effect"]:
            # 打字机效果
            message_placeholder = st.empty()
            full_response = ""

            for chunk in response_content.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(st.session_state.settings["typing_speed"])

            message_placeholder.markdown(full_response)
        else:
            st.markdown(response_content)

        # 保存到历史
        st.session_state.messages.append({"role": "assistant", "content": response_content})

        # 检查是否完成复制
        if "复制完成" in response_content:
            st.balloons()


# ==================== 示例查询 ====================
def load_example(example: str):
    """加载示例查询"""
    process_user_input(example)


# ==================== 导出对话 ====================
def export_conversation():
    """导出对话为 Markdown"""
    md_content = "# 型号复制助手对话记录\n\n"
    md_content += f"**会话ID**: {st.session_state.session_id}\n\n"
    md_content += f"**导出时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"

    for msg in st.session_state.messages:
        role = "👤 **用户**" if msg["role"] == "user" else "🤖 **助手**"
        md_content += f"{role}:\n\n{msg['content']}\n\n---\n\n"

    return md_content


# ==================== 主界面 ====================
def main():
    """主函数"""

    # 初始化
    init_session_state()

    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.title("⚙️ 控制面板")

        # 会话管理
        st.subheader("💬 会话管理")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                clear_conversation()
        with col2:
            if st.button("🔄 重置会话", use_container_width=True):
                reset_session()

        st.divider()

        # 会话切换
        st.subheader("📁 切换会话")
        new_session = st.text_input(
            "会话ID",
            value=st.session_state.session_id,
            placeholder="输入会话ID",
            key="session_input"
        )
        if st.button("切换", use_container_width=True):
            if new_session:
                switch_session(new_session)

        st.divider()

        # 示例查询
        st.subheader("📝 示例查询")
        examples = [
            "📋 我想复制型号",
            "🔄 把 X1 的功能复制到 X2",
            "❓ X5 型号存在吗？",
            "ℹ️ 查看 X1 型号的功能"
        ]

        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                load_example(example.split(" ", 1)[1])

        st.divider()

        # 可用型号
        st.subheader("📦 可用型号")

        # 获取智能体的型号信息
        agent = st.session_state.agent
        available_models = agent.get_available_models()
        model_info = agent.get_model_info_dict()

        for model in available_models:
            with st.expander(f"🔹 {model}"):
                st.markdown(f"**功能**: {model_info.get(model, '基础功能')}")

        st.divider()

        # 设置
        st.subheader("⚙️ 显示设置")

        show_typing = st.toggle(
            "打字机效果",
            value=st.session_state.settings["show_typing_effect"],
            key="typing_toggle"
        )
        st.session_state.settings["show_typing_effect"] = show_typing

        if show_typing:
            typing_speed = st.slider(
                "打字速度",
                min_value=0.01,
                max_value=0.2,
                value=st.session_state.settings["typing_speed"],
                step=0.01,
                key="speed_slider"
            )
            st.session_state.settings["typing_speed"] = typing_speed

        st.divider()

        # 导出功能
        st.subheader("📤 导出")

        if st.button("📥 导出对话", use_container_width=True):
            md_content = export_conversation()
            st.download_button(
                label="下载 Markdown",
                data=md_content,
                file_name=f"conversation_{st.session_state.session_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.divider()

        # 状态信息
        st.subheader("📊 状态")
        st.info(f"**会话ID**: {st.session_state.session_id}")
        st.info(f"**消息数**: {len(st.session_state.messages)}")
        st.info(f"**模型**: qwen3-max")

    # ==================== 主聊天区域 ====================
    st.title("🤖 型号复制智能助手")

    # 描述
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;'>
        <h4 style='margin: 0;'>📋 功能介绍</h4>
        <p style='margin: 10px 0 0 0;'>我可以帮助你完成型号功能的复制操作，流程如下：</p>
        <ol style='margin: 10px 0 0 20px;'>
            <li><strong>收集信息</strong> - 询问源型号和目标型号</li>
            <li><strong>验证检查</strong> - 检查目标型号是否存在</li>
            <li><strong>确认操作</strong> - 让用户确认复制操作</li>
            <li><strong>执行复制</strong> - 执行复制逻辑并反馈结果</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # 聊天历史容器
    chat_container = st.container()

    # 显示历史消息
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 输入框
    if prompt := st.chat_input("请输入你的问题..."):
        process_user_input(prompt)

    # 欢迎消息（首次访问）
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            welcome_msg = """你好！我是型号复制智能助手 🤖

我可以帮你完成型号功能的复制操作。请告诉我你想要做什么，例如：

- 🔄 "把 X1 的功能复制到 X2"
- 📋 "我想复制型号"
- ℹ️ "查看 X1 型号的功能"
- ❓ "X5 型号存在吗？"

我会按照标准流程引导你完成操作。让我们开始吧！"""
            st.markdown(welcome_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )

    # 自动滚动到最新消息
    if st.session_state.settings["auto_scroll"]:
        st.markdown(
            """
            <style>
            .stChatContainer {
                scroll-behavior: smooth;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


# ==================== 运行应用 ====================
if __name__ == "__main__":
    main()