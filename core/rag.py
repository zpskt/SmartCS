'''核心'''
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import config_data as config
from core.vector_stores import VectorService


class RagService(object):
    def __init__(self):
        self.vector_service = VectorService(DashScopeEmbeddings(
            model=config.embedding_model_name))
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主,"
                           "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model_name=config.chat_model_name)
        # 主链
        self.chain = self.__get_chain()

    def __get_chain(self):
        '''获取最终执行链'''
        retriever = self.vector_service.get_retriever()

        def format_for_retriever(value: dict) -> str:
            """
            将retriever的数据格式转换
            :param value:
            :return:
            """
            print(f"Retriever: {value}")
            return value["input"]

        def format_document(docs: list[Document]):
            """对数据库返回的文档进行格式处理"""
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n 文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_prompt_template(value):
            """将用户输入的格式转换成prompt模板需要的格式"""
            # {input, context, history}
            print(f"Prompt Template: {value}")
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | retriever | format_document
                } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )
