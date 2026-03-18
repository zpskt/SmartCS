from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        """
        执行agent流式查询返回响应内容

        :param query: 用户的查询字符串
        :return: 生成器对象，逐个产出 Agent 响应消息的内容（已去除首尾空白）
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        # stream_model里面是设置输出的类型，例如chunk的类型，values：看完整状态值（返回所有，包括所有累计的消息历史） updates：只看变化；messages：实时看生成文字的过程 debug：调试使用，看所有细节
        # context是一个运行时的上下文容器，用于在Agent执行过程中传递和共享自定义数据
        '''
        contex 使用模式：初始化，读取，修改，使用。支持多层嵌套数据结构。不可以存储超大对象，不然占用内存太多了，不要存储token。
        在实际使用过程中，是全局共享的，所有middlewar、tools、nodes都能f昂文和修改。生命周期全程保持。
        常用的实际应用场景举例。
        
        '''
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
