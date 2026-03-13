import os
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


def get_md5_by_str(input_str: str, encoding='utf-8'):
    '''生成md5'''
    import hashlib
    md5 = hashlib.md5()
    md5.update(input_str.encode(encoding))
    return md5.hexdigest()


def check_md5_exist(md5_str: str) -> bool:
    if not os.path.exists(config.md5_file_path):
        # 不存在文件，说明肯定没有
        open(config.md5_file_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_file_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # 去掉换行符 空格
            if line == md5_str:
                return True
        return False


def save_md5(md5_hex: str):
    with open(config.md5_file_path, 'a', encoding='utf-8') as f:
        f.write(md5_hex + '\n')


class KnowledgeBaseService(object):
    def __init__(self):
        # 创建文件目录
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chorma = Chroma(
            collection_name=config.collection_name,  # 数据库的表名
            persist_directory=config.persist_directory,  # 数据库本地存储文件夹
            embedding_function=DashScopeEmbeddings(
                model=config.embedding_model_name,
                dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))  # 嵌入模型 )
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,  # 连续文本段之间的字符重叠数量
            separators=config.separators,  # 自然段落划分符号
            length_function=len  # 使用python自带的len函数做长度统计的依据
        )

    def upload_by_str(self, data: str, filename: str):
        '''上传'''
        # 判断md5
        md5_hex = get_md5_by_str(data)
        # 判断是否存在
        if check_md5_exist(md5_hex):
            # 存在，不重复添加
            return '[略过]已添加'

        if len(data) > config.max_split_chunk_size:
            knowldege_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowldege_chunks = [data]

        # 元组数据
        metadata = {
            'source': filename,
            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'creator': "zpaskt"
        }
        print(f"[添加] {filename} {len(knowldege_chunks)}")
        # 保存到数据库
        self.chorma.add_texts(
            texts=knowldege_chunks,
            metadatas=[metadata for _ in knowldege_chunks]
        )
        # 保存md5
        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"

if __name__ == '__main__':
    kbs = KnowledgeBaseService()
    result = kbs.upload_by_str("123456", "test.txt")
    print(result)
