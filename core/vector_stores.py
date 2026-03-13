"""向量服务"""
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

class VectorService(object):
    def __init__(self, embedding):
       """
       :param embedding: 嵌入模型的传入
       """
       self.embedding = embedding

       self.vector_store = Chroma(
           collection_name=config.collection_name,
           persist_directory=config.persist_directory,
           embedding_function=self.embedding,
       )

    def get_retriever(self):
        """
        返回向量检索器
        :return:
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

if __name__ == '__main__':
    vector_service = VectorService(DashScopeEmbeddings(model=config.embedding_model_name))
    retriever = vector_service.get_retriever()
    res = retriever.invoke("开关门管理生产数据库地址?")
    print(res)
