'''配置'''
persist_directory = './data'
md5_file_path = './data/md5.txt'
max_split_chunk_size = 1000
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n",'.', '?', '!', '。', '？', '！', ';', '；', ':', '：', ',']

collection_name = 'rag'
embedding_model_name = "text-embedding-v4"

chat_model_name = "qwen3-max"

similarity_threshold = 1            # 检索返回匹配的文档数量
history_file_path = './data/history'